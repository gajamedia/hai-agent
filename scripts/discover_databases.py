"""Find the database servers installed on this computer and (optionally) register them as profiles.

    python scripts/discover_databases.py                      # only look and report
    python scripts/discover_databases.py --add                # also offer to add new MySQL/PostgreSQL profiles
    python scripts/discover_databases.py --sqlite-dir D:\\data  # also look for SQLite files in a folder
    python scripts/discover_databases.py --host 10.0.0.5      # also probe one more address you name yourself

What it does, and what it deliberately does NOT do:
  * It probes only 127.0.0.1 plus the addresses you pass with --host. It never scans a network range.
  * It identifies a server by its first bytes on the standard ports (MySQL/MariaDB 3306, PostgreSQL 5432,
    SQL Server 1433) and by Windows service names. It sends no login, no password, no query.
  * With --add it writes a profile into profiles.json (host, port, database name, account NAMES and
    the name of the environment variable that will hold the password). It never writes a password and
    never asks for one. The next step it prints creates the read-only account (setup_mysql_readonly_user.py
    asks for the administrator password once, in the terminal, and does not keep it).
  * SQL Server is detected and reported only: the agent has no SQL Server engine yet, so no profile is written.
  * SQLite files are listed, with whether they sit inside HAI_DATABASE_ROOT (the only place the agent reads).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


PORTS = {"mysql": 3306, "postgresql": 5432, "sqlserver": 1433}
SERVICE_WORDS = {"mysql": ("mysql", "mariadb"), "postgresql": ("postgresql",), "sqlserver": ("mssql",)}
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_HOST = re.compile(r"^[A-Za-z0-9_.\-]{1,100}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_DB = re.compile(r"^[A-Za-z0-9_]{1,64}$")


@dataclass(frozen=True)
class Found:
    kind: str       # mysql | postgresql | sqlserver
    host: str
    port: int
    detail: str     # what the server said about itself, if anything


# ------------------------------------------------------------------ probing
def _connect(host: str, port: int, timeout: float) -> socket.socket | None:
    try:
        return socket.create_connection((host, port), timeout=timeout)
    except OSError:
        return None


def probe_mysql(host: str, port: int, timeout: float = 1.0) -> Found | None:
    """MySQL and MariaDB speak first: a greeting packet with protocol byte 10 and a version string."""
    sock = _connect(host, port, timeout)
    if sock is None:
        return None
    with sock:
        try:
            sock.settimeout(timeout)
            data = sock.recv(256)
        except OSError:
            return None
    if len(data) > 6 and data[4] == 10:
        version = data[5:].split(b"\x00", 1)[0].decode("ascii", "replace")
        return Found("mysql", host, port, ("MariaDB " if "mariadb" in version.lower() else "MySQL ") + version)
    return None


def probe_postgresql(host: str, port: int, timeout: float = 1.0) -> Found | None:
    """An SSLRequest (8 bytes) is answered with one byte, 'S' or 'N', by a PostgreSQL server. No login."""
    sock = _connect(host, port, timeout)
    if sock is None:
        return None
    with sock:
        try:
            sock.settimeout(timeout)
            sock.sendall(b"\x00\x00\x00\x08\x04\xd2\x16\x2f")
            answer = sock.recv(1)
        except OSError:
            return None
    if answer in (b"S", b"N"):
        return Found("postgresql", host, port, "PostgreSQL (SSL " + ("available" if answer == b"S" else "off") + ")")
    return None


def probe_sqlserver(host: str, port: int, timeout: float = 1.0) -> Found | None:
    """A TDS pre-login packet is answered with a TDS response (type 4). No login."""
    prelogin = (b"\x12\x01\x00\x2f\x00\x00\x01\x00"            # header: type 0x12, length 47
                b"\x00\x00\x1a\x00\x06\x01\x00\x20\x00\x01\x02\x00\x21\x00\x01\x03\x00\x22\x00\x04\x04\x00\x26\x00\x01\xff"
                b"\x09\x00\x00\x00\x00\x00\x01\x00\xb8\x0d\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    sock = _connect(host, port, timeout)
    if sock is None:
        return None
    with sock:
        try:
            sock.settimeout(timeout)
            sock.sendall(prelogin)
            answer = sock.recv(8)
        except OSError:
            return None
    if len(answer) >= 8 and answer[0] == 4:
        return Found("sqlserver", host, port, "Microsoft SQL Server")
    return None


PROBES: dict[str, Callable[..., Found | None]] = {
    "mysql": probe_mysql, "postgresql": probe_postgresql, "sqlserver": probe_sqlserver}


def probe_hosts(hosts: Iterable[str], timeout: float = 1.0) -> list[Found]:
    found = []
    for host in hosts:
        for kind, port in PORTS.items():
            hit = PROBES[kind](host, port, timeout)
            if hit:
                found.append(hit)
    return found


# ------------------------------------------------------------------ Windows services
def windows_services(run: Callable[..., str] | None = None) -> dict[str, list[str]]:
    """Installed database services by kind. Informational: a service can be stopped or on a custom port."""
    if run is None:
        if os.name != "nt":
            return {}
        def run(*cmd):  # noqa: E306
            return subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False).stdout
    try:
        text = run("sc", "query", "type=", "service", "state=", "all")
    except (OSError, subprocess.SubprocessError):
        return {}
    names = re.findall(r"SERVICE_NAME:\s*(\S+)", text or "")
    out: dict[str, list[str]] = {}
    for name in names:
        low = name.lower()
        for kind, words in SERVICE_WORDS.items():
            if any(low.startswith(w) or f"${w}" in low for w in words) and "sqlserveragent" not in low \
                    and "browser" not in low and "writer" not in low:
                out.setdefault(kind, []).append(name)
    return out


# ------------------------------------------------------------------ SQLite
def find_sqlite_files(folder: Path, *, limit: int = 200) -> list[Path]:
    """SQLite files directly in a folder or one level below it, checked by their 16-byte header."""
    found: list[Path] = []
    try:
        candidates = [p for p in folder.glob("*") if p.is_file()] + [p for p in folder.glob("*/*") if p.is_file()]
    except OSError:
        return []
    for path in candidates:
        if path.suffix.lower() not in SQLITE_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue    # hidden folders such as .hai-work hold the agent's own files, never offer those
        try:
            with path.open("rb") as handle:
                if handle.read(16) == b"SQLite format 3\x00":
                    found.append(path)
        except OSError:
            continue
        if len(found) >= limit:
            break
    return sorted(found)


def inside_root(path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


# ------------------------------------------------------------------ profiles
def load_profiles_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise SystemExit(f"error: {path} must hold a JSON object")
    return data


def is_registered(profiles: dict, found: Found) -> str | None:
    local = {"127.0.0.1", "localhost", "::1"}
    for name, raw in profiles.items():
        if not isinstance(raw, dict) or raw.get("type") != found.kind:
            continue
        same_host = str(raw.get("host", "")).lower() == found.host.lower() or (
            str(raw.get("host", "")).lower() in local and found.host in local)
        if same_host and int(raw.get("port", PORTS[found.kind])) == found.port:
            return name
    return None


def draft_profile(found: Found, *, name: str, database: str, user: str = "hai_reader") -> dict:
    """host, port, database and NAMES only. The password lives in an environment variable."""
    if found.kind not in ("mysql", "postgresql"):
        raise ValueError(f"{found.kind} is not supported by the agent yet")
    if not _NAME.match(name):
        raise ValueError("profile name: 2-32 characters, lowercase letters, digits and _, starting with a letter")
    if not _DB.match(database):
        raise ValueError("database name: letters, digits and _ only")
    profile = {"type": found.kind, "host": found.host, "port": found.port, "database": database,
               "user": user, "password_env": f"HAI_{name.upper()}_PASSWORD"}
    profile["ssl" if found.kind == "mysql" else "sslmode"] = "require"
    return profile


def next_step(found: Found, name: str) -> str:
    if found.kind == "mysql":
        return f"python scripts\\setup_mysql_readonly_user.py {name}"
    return (f"create a read-only PostgreSQL account yourself, put its password in .env as "
            f"HAI_{name.upper()}_PASSWORD, then run: python scripts\\check_database_profile.py {name}")


# ------------------------------------------------------------------ main
def build_report(found: list[Found], services: dict[str, list[str]], profiles: dict,
                 sqlite_files: list[Path], root: Path | None) -> list[str]:
    lines = ["Database servers on this computer:"]
    if not found:
        lines.append("  (none answered on the standard ports)")
    for hit in found:
        name = is_registered(profiles, hit)
        status = f"registered as profile '{name}'" if name else "NOT registered"
        if hit.kind == "sqlserver":
            status = "detected; the agent has no SQL Server engine yet, so it cannot be registered"
        lines.append(f"  - {hit.detail} at {hit.host}:{hit.port}: {status}")
    for kind, names in sorted(services.items()):
        if not any(f.kind == kind for f in found):
            lines.append(f"  - {kind} service installed ({', '.join(names)}) but not answering on {PORTS[kind]}: "
                         "stopped, or on another port")
    if sqlite_files:
        lines.append("SQLite files:")
        for path in sqlite_files:
            where = "inside HAI_DATABASE_ROOT: can be connected" if inside_root(path, root) \
                else "outside HAI_DATABASE_ROOT: copy it there first, the agent reads only that folder"
            lines.append(f"  - {path} ({where})")
    return lines


def main(argv: list[str] | None = None, *, hosts_probe: Callable[..., list[Found]] = probe_hosts,
         services_probe: Callable[[], dict[str, list[str]]] = windows_services,
         ask: Callable[[str], str] = input, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", action="append", default=[], help="an extra address to probe (repeatable)")
    parser.add_argument("--sqlite-dir", action="append", default=[], help="a folder to search for SQLite files")
    parser.add_argument("--add", action="store_true", help="offer to add new MySQL/PostgreSQL profiles")
    parser.add_argument("--profiles", default="", help="profiles JSON (default: HAI_DATABASE_PROFILES)")
    parser.add_argument("--env", default=str(ROOT / ".env"), help=".env file to read HAI_* values from")
    args = parser.parse_args(argv)

    env_values: dict[str, str] = {}
    env_path = Path(args.env)
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                env_values[key.strip()] = value.strip().strip('"')
    get = lambda k: os.getenv(k) or env_values.get(k, "")  # noqa: E731
    profiles_path = Path(args.profiles or get("HAI_DATABASE_PROFILES") or ROOT / "data" / "reports" / "profiles.json")
    root_text = get("HAI_DATABASE_ROOT")
    root = Path(root_text) if root_text else None

    hosts = ["127.0.0.1"]
    for host in args.host:
        if not _HOST.match(host):
            out(f"error: '{host}' is not a plain host name or address")
            return 2
        if host not in hosts:
            hosts.append(host)

    try:
        profiles = load_profiles_json(profiles_path)
    except (OSError, ValueError) as exc:
        out(f"error: cannot read {profiles_path}: {exc}")
        return 2

    found = hosts_probe(hosts)
    sqlite_files: list[Path] = []
    folders = [Path(p) for p in args.sqlite_dir] + ([root] if root and root.is_dir() else [])
    for folder in folders:
        for path in find_sqlite_files(folder):
            if path not in sqlite_files:
                sqlite_files.append(path)
    for line in build_report(found, services_probe(), profiles, sqlite_files, root):
        out(line)

    if not args.add:
        if any(not is_registered(profiles, f) and f.kind != "sqlserver" for f in found):
            out("\nRun again with --add to register the new ones (no password is asked or written).")
        return 0

    changed = False
    for hit in found:
        if hit.kind == "sqlserver" or is_registered(profiles, hit):
            continue
        if ask(f"\nRegister {hit.detail} at {hit.host}:{hit.port}? [y/N] ").strip().lower() != "y":
            continue
        default_name = f"{'local' if hit.host == '127.0.0.1' else hit.host.replace('.', '_').replace('-', '_')}_{hit.kind}"
        name = ask(f"Profile name [{default_name[:32]}]: ").strip() or default_name[:32]
        database = ask("Database the agent may read (name only): ").strip()
        try:
            if name in profiles:
                raise ValueError(f"a profile named '{name}' already exists")
            profile = draft_profile(hit, name=name, database=database)
        except ValueError as exc:
            out(f"  skipped: {exc}")
            continue
        profiles[name] = profile
        changed = True
        out(f"  added '{name}' (no password stored; it will read it from {profile['password_env']}).")
        out(f"  next: {next_step(hit, name)}")
    if changed:
        profiles_path.parent.mkdir(parents=True, exist_ok=True)
        profiles_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        out(f"\nSaved {profiles_path}. Restart the backend to use the new profile(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
