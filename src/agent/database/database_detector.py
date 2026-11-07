"""Passive database driver and service discovery.

This module deliberately does not connect to databases, read credentials,
install drivers, start/stop services, or modify configuration.
"""

from __future__ import annotations

import importlib.util
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DatabaseDriverStatus:
    database: str
    module: str
    available: bool
    version: Optional[str] = None


@dataclass(frozen=True)
class DatabaseServiceStatus:
    database: str
    source: str
    identifier: str
    running: bool
    details: Optional[str] = None


@dataclass(frozen=True)
class DatabaseDetectionReport:
    detected_at: str
    drivers: Tuple[DatabaseDriverStatus, ...]
    services: Tuple[DatabaseServiceStatus, ...]
    docker_available: bool
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class DatabaseDetector:
    """Detect installed Python drivers and already-running DB services."""

    DRIVER_MODULES: Tuple[Tuple[str, str], ...] = (
        ("sqlite", "sqlite3"),
        ("postgresql", "psycopg"),
        ("postgresql", "psycopg2"),
        ("mysql", "pymysql"),
        ("mysql", "mysql.connector"),
        ("sqlserver", "pyodbc"),
        ("oracle", "oracledb"),
        ("oracle", "cx_Oracle"),
        ("mongodb", "pymongo"),
        ("duckdb", "duckdb"),
        ("sqlalchemy", "sqlalchemy"),
        ("sqlite_async", "aiosqlite"),
    )

    DATABASE_PATTERN = re.compile(
        r"(postgres|postgre|mysql|maria|mongo|sql[ _-]?server|mssql|oracle|duckdb|sqlite)", re.IGNORECASE
    )

    def detect(
        self,
        include_services: bool = True,
        include_docker_containers: bool = True,
    ) -> DatabaseDetectionReport:
        warnings: List[str] = []
        drivers = tuple(self._detect_drivers())
        services: List[DatabaseServiceStatus] = []
        docker_available = shutil.which("docker") is not None

        if include_services:
            services.extend(self._detect_host_services(warnings))

        if include_services and include_docker_containers and docker_available:
            services.extend(self._detect_docker_services(warnings))

        return DatabaseDetectionReport(
            detected_at=datetime.now(timezone.utc).isoformat(),
            drivers=drivers,
            services=tuple(services),
            docker_available=docker_available,
            warnings=tuple(warnings),
        )

    @classmethod
    def _detect_drivers(cls) -> List[DatabaseDriverStatus]:
        results: List[DatabaseDriverStatus] = []

        for database, module in cls.DRIVER_MODULES:
            try:
                available = importlib.util.find_spec(module) is not None
            except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
                available = False

            results.append(
                DatabaseDriverStatus(
                    database=database,
                    module=module,
                    available=available,
                )
            )

        return results

    @classmethod
    def _detect_host_services(
        cls, warnings: List[str]
    ) -> List[DatabaseServiceStatus]:
        system = platform.system().lower()
        if system == "windows":
            return cls._detect_windows_services(warnings)
        if system == "linux":
            return cls._detect_systemd_services(warnings)
        if system == "darwin":
            return cls._detect_macos_services(warnings)
        return []

    @classmethod
    def _detect_windows_services(
        cls, warnings: List[str]
    ) -> List[DatabaseServiceStatus]:
        executable = shutil.which("sc.exe") or shutil.which("sc")
        if not executable:
            warnings.append("Windows service query unavailable: sc.exe not found")
            return []

        try:
            result = subprocess.run(
                [executable, "query", "state=", "all"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"Windows service query failed: {type(exc).__name__}: {exc}")
            return []

        if result.returncode != 0:
            warnings.append("Windows service query returned a non-zero exit code")
            return []

        return cls._parse_windows_services(result.stdout)

    @classmethod
    def _parse_windows_services(cls, output: str) -> List[DatabaseServiceStatus]:
        services: List[DatabaseServiceStatus] = []
        current_name: Optional[str] = None
        current_state: Optional[str] = None

        def append_current() -> None:
            if not current_name or not cls.DATABASE_PATTERN.search(current_name):
                return
            state = current_state or "UNKNOWN"
            database = cls._database_from_text(current_name)
            services.append(
                DatabaseServiceStatus(
                    database=database,
                    source="windows_service",
                    identifier=current_name,
                    running=state.upper().startswith("4"),
                    details=state,
                )
            )

        for line in output.splitlines():
            name_match = re.search(r"SERVICE_NAME\s*:\s*(.+)", line, re.IGNORECASE)
            state_match = re.search(r"STATE\s*:\s*(.+)", line, re.IGNORECASE)

            if name_match:
                append_current()
                current_name = name_match.group(1).strip()
                current_state = None
            elif state_match and current_name:
                current_state = state_match.group(1).strip()

        append_current()
        return services

    @classmethod
    def _detect_systemd_services(
        cls, warnings: List[str]
    ) -> List[DatabaseServiceStatus]:
        executable = shutil.which("systemctl")
        if not executable:
            return []

        try:
            result = subprocess.run(
                [
                    executable,
                    "list-units",
                    "--type=service",
                    "--state=running",
                    "--no-legend",
                    "--plain",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"systemd service query failed: {type(exc).__name__}: {exc}")
            return []

        if result.returncode != 0:
            warnings.append("systemd service query returned a non-zero exit code")
            return []

        services: List[DatabaseServiceStatus] = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if not fields:
                continue
            identifier = fields[0]
            if cls.DATABASE_PATTERN.search(identifier):
                services.append(
                    DatabaseServiceStatus(
                        database=cls._database_from_text(identifier),
                        source="systemd",
                        identifier=identifier,
                        running=True,
                        details=" ".join(fields[1:]) or None,
                    )
                )
        return services

    @classmethod
    def _detect_macos_services(
        cls, warnings: List[str]
    ) -> List[DatabaseServiceStatus]:
        executable = shutil.which("brew")
        if not executable:
            return []

        try:
            result = subprocess.run(
                [executable, "services", "list"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"Homebrew service query failed: {type(exc).__name__}: {exc}")
            return []

        if result.returncode != 0:
            warnings.append("Homebrew service query returned a non-zero exit code")
            return []

        services: List[DatabaseServiceStatus] = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) < 2 or not cls.DATABASE_PATTERN.search(fields[0]):
                continue
            state = fields[1].lower()
            services.append(
                DatabaseServiceStatus(
                    database=cls._database_from_text(fields[0]),
                    source="homebrew",
                    identifier=fields[0],
                    running=state == "started",
                    details=" ".join(fields[1:]),
                )
            )
        return services

    @classmethod
    def _detect_docker_services(
        cls, warnings: List[str]
    ) -> List[DatabaseServiceStatus]:
        executable = shutil.which("docker")
        if not executable:
            return []

        try:
            result = subprocess.run(
                [
                    executable,
                    "ps",
                    "--format",
                    "{{.Names}}\t{{.Image}}\t{{.Status}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"Docker container query failed: {type(exc).__name__}: {exc}")
            return []

        if result.returncode != 0:
            warnings.append(
                "Docker container query failed; Docker may be unavailable or inaccessible"
            )
            return []

        services: List[DatabaseServiceStatus] = []
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            name, image = fields[0].strip(), fields[1].strip()
            status = fields[2].strip() if len(fields) > 2 else "running"
            identity = f"{name} {image}"
            if not cls.DATABASE_PATTERN.search(identity):
                continue
            services.append(
                DatabaseServiceStatus(
                    database=cls._database_from_text(identity),
                    source="docker_container",
                    identifier=name,
                    running=True,
                    details=f"{image} | {status}",
                )
            )
        return services

    @staticmethod
    def _database_from_text(value: str) -> str:
        lowered = value.lower()
        if "postgres" in lowered or "postgre" in lowered:
            return "postgresql"
        if "mysql" in lowered:
            return "mysql"
        if "maria" in lowered:
            return "mariadb"
        if "mongo" in lowered:
            return "mongodb"
        if "sql server" in lowered or "sqlserver" in lowered or "mssql" in lowered:
            return "sqlserver"
        if "oracle" in lowered:
            return "oracle"
        if "duckdb" in lowered:
            return "duckdb"
        if "sqlite" in lowered:
            return "sqlite"
        return "unknown"
