"""Read-only, dependency-free device profiling.

The profiler uses Python's standard library only. It does not install software,
start services, inspect credentials, or connect to external systems.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


PathLike = Union[str, os.PathLike[str]]


@dataclass(frozen=True)
class CpuProfile:
    architecture: str
    processor: Optional[str]
    logical_processors: Optional[int]


@dataclass(frozen=True)
class MemoryProfile:
    total_bytes: Optional[int]
    available_bytes: Optional[int]
    source: Optional[str]


@dataclass(frozen=True)
class DiskProfile:
    requested_path: str
    measured_path: Optional[str]
    total_bytes: Optional[int]
    free_bytes: Optional[int]
    used_bytes: Optional[int]


@dataclass(frozen=True)
class PythonProfile:
    version: str
    implementation: str
    executable: str
    prefix: str
    is_venv: bool


@dataclass(frozen=True)
class DockerProfile:
    available_on_path: bool
    executable: Optional[str]


@dataclass(frozen=True)
class DeviceProfile:
    captured_at: str
    operating_system: str
    operating_system_release: str
    operating_system_version: str
    machine: str
    cpu: CpuProfile
    memory: MemoryProfile
    disk: DiskProfile
    python: PythonProfile
    docker: Optional[DockerProfile]
    warnings: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class DeviceProfiler:
    """Collect a best-effort snapshot of the current host."""

    def profile(
        self,
        target_path: Optional[PathLike] = None,
        include_docker: bool = True,
    ) -> DeviceProfile:
        warnings: List[str] = []
        requested_path = str(target_path if target_path is not None else Path.cwd())

        return DeviceProfile(
            captured_at=datetime.now(timezone.utc).isoformat(),
            operating_system=platform.system() or "unknown",
            operating_system_release=platform.release() or "unknown",
            operating_system_version=platform.version() or "unknown",
            machine=platform.machine() or "unknown",
            cpu=self._profile_cpu(),
            memory=self._profile_memory(warnings),
            disk=self._profile_disk(requested_path, warnings),
            python=self._profile_python(),
            docker=self._profile_docker() if include_docker else None,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _profile_cpu() -> CpuProfile:
        logical = os.cpu_count()
        if logical is not None and logical < 1:
            logical = None

        processor = platform.processor().strip() or None
        return CpuProfile(
            architecture=platform.machine() or "unknown",
            processor=processor,
            logical_processors=logical,
        )

    @classmethod
    def _profile_memory(cls, warnings: List[str]) -> MemoryProfile:
        system = platform.system().lower()

        try:
            if system == "windows":
                total, available = cls._windows_memory()
                source = "windows_globalmemorystatusex"
            elif system == "linux":
                total, available = cls._linux_memory()
                source = "/proc/meminfo"
            elif system == "darwin":
                total, available = cls._posix_memory()
                source = "os.sysconf"
            else:
                total, available = cls._posix_memory()
                source = "os.sysconf"

            if total is None:
                raise RuntimeError("total physical memory is unavailable")

            return MemoryProfile(
                total_bytes=total,
                available_bytes=available,
                source=source,
            )
        except Exception as exc:  # best-effort telemetry must not abort profiling
            warnings.append(f"memory profiling unavailable: {type(exc).__name__}: {exc}")
            return MemoryProfile(total_bytes=None, available_bytes=None, source=None)

    @staticmethod
    def _windows_memory() -> tuple[int, int]:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        function = ctypes.windll.kernel32.GlobalMemoryStatusEx
        function.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        function.restype = ctypes.c_int

        if not function(ctypes.byref(status)):
            raise ctypes.WinError()

        return int(status.ullTotalPhys), int(status.ullAvailPhys)

    @staticmethod
    def _linux_memory() -> tuple[Optional[int], Optional[int]]:
        values: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[0].rstrip(":") in {
                    "MemTotal",
                    "MemAvailable",
                }:
                    values[parts[0].rstrip(":")] = int(parts[1]) * 1024

        return values.get("MemTotal"), values.get("MemAvailable")

    @staticmethod
    def _posix_memory() -> tuple[Optional[int], Optional[int]]:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * total_pages, page_size * available_pages

    @staticmethod
    def _profile_disk(requested_path: str, warnings: List[str]) -> DiskProfile:
        candidate = Path(requested_path).expanduser()

        try:
            resolved = candidate.resolve(strict=False)
            probe = resolved
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent

            usage = shutil.disk_usage(str(probe))
            return DiskProfile(
                requested_path=requested_path,
                measured_path=str(probe),
                total_bytes=int(usage.total),
                free_bytes=int(usage.free),
                used_bytes=int(usage.used),
            )
        except Exception as exc:
            warnings.append(f"disk profiling unavailable: {type(exc).__name__}: {exc}")
            return DiskProfile(
                requested_path=requested_path,
                measured_path=None,
                total_bytes=None,
                free_bytes=None,
                used_bytes=None,
            )

    @staticmethod
    def _profile_python() -> PythonProfile:
        return PythonProfile(
            version=sys.version,
            implementation=platform.python_implementation(),
            executable=sys.executable,
            prefix=sys.prefix,
            is_venv=(sys.prefix != getattr(sys, "base_prefix", sys.prefix)),
        )

    @staticmethod
    def _profile_docker() -> DockerProfile:
        executable = shutil.which("docker")
        return DockerProfile(
            available_on_path=executable is not None,
            executable=executable,
        )
