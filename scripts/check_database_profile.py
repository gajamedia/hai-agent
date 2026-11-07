"""Check a database profile from your own terminal and see the real reason when it fails.

    python scripts/check_database_profile.py local_mysql

It loads your .env first (like ``uvicorn --env-file .env``), so it sees what the server sees.

The web API deliberately hides driver error details from browsers. This tool runs the very
same code path the server uses (same read-only session, same SELECT-only account check,
same schema discovery) but prints the actual cause on *your* console. It never prints the
password. Profiles come from the JSON file named by HAI_DATABASE_PROFILES (or --profiles);
the password comes from the environment variable that the profile names.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent.database import ConnectionManager  # noqa: E402
from api.database_engines import (  # noqa: E402
    EngineError, MySQLEngine, PostgresEngine, load_profiles,
)

_ENGINES = {"mysql": MySQLEngine, "postgresql": PostgresEngine}
_DRIVER_HINT = {"mysql": 'pip install "PyMySQL>=1.1,<2"', "postgresql": 'pip install "psycopg[binary]>=3.2,<4"'}

# MySQL / MariaDB client error numbers -> what to do about them.
_MYSQL_HINTS = {
    1045: "wrong user or password, OR the account exists for a different host than the one the "
          "server saw: the message above names the host it saw as 'user'@'host'. List your "
          "accounts with: SELECT user, host FROM mysql.user;",
    1044: "the account has no access to this database (GRANT SELECT ON <database>.* ...)",
    1049: "that database does not exist on the server (check the name; names can be case sensitive)",
    2003: "the server is not reachable at that host and port (is MySQL running, and is the port right?)",
    2002: "the server is not reachable (is MySQL running? check host and port)",
    2013: "the connection dropped while connecting (a firewall or the wrong port?)",
    2026: "the TLS handshake failed: the server may not offer encryption. This tool never sends "
          "a password in clear text, so enable TLS on the server (MySQL 8 does by default)",
}


def _hint(exc: BaseException) -> str:
    code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
    if code in _MYSQL_HINTS:
        return _MYSQL_HINTS[code]
    text = str(exc).lower()
    if "ssl" in text or "tls" in text:
        return "a TLS problem: check the profile's ssl/sslmode and that the server offers encryption"
    if "password authentication failed" in text:
        return "wrong user or password"
    if "could not connect" in text or "connection refused" in text:
        return "the server is not reachable (is it running? check host and port)"
    return ""


def load_env_file(path: Path) -> list[str]:
    """Load KEY=VALUE pairs from a .env file the way ``uvicorn --env-file`` does.

    Variables already set in the terminal win. Returns the NAMES that were loaded
    (never the values), so the tool sees the same settings as the server.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return []
    if not path.is_file():
        return []
    loaded = []
    for key, value in dotenv_values(path).items():
        if value is not None and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def check_profile(name: str, profiles: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return (ok, report lines). Never includes the password."""
    if name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none)"
        return False, [f"[FAIL] no profile named '{name}'. Profiles in the file: {available}"]
    profile = profiles[name]
    lines = [f"[ok]   profile '{name}': {profile.type} at {profile.host}:{profile.port}, "
             f"database '{profile.database}', user '{profile.user}'"]
    if not os.environ.get(profile.password_env):
        lines.append(f"[FAIL] environment variable {profile.password_env} is not set or empty. "
                     "Put it in .env (or set it in this terminal), then restart the server.")
        return False, lines
    secret = os.environ[profile.password_env]
    # The length is not a secret but lets you spot a value that .env cut short (a `#`
    # starts a comment) without the tool ever printing the password itself.
    lines.append(f"[ok]   password variable {profile.password_env} is set ({len(secret)} characters)")
    if secret[:1] in {'"', "'"} or secret[-1:] in {'"', "'"}:
        lines.append("[WARN] the password starts or ends with a quote character. In .env write the "
                     "value without quotes, or the quote becomes part of the password.")
    if secret != secret.strip():
        lines.append("[WARN] the password starts or ends with a space.")
    engine = _ENGINES[profile.type]({name: profile})
    manager = ConnectionManager()
    try:
        schema, warnings = engine.open(manager, "check", name, "diagnostic:check_database_profile")
    except ImportError:
        lines.append(f"[FAIL] the database driver is not installed. Run: {_DRIVER_HINT[profile.type]}")
        return False, lines
    except EngineError as exc:
        lines.append(f"[FAIL] refused by the safety rules: {exc}")
        return False, lines
    except Exception as exc:  # the driver's own error: shown here, never by the web API
        lines.append(f"[FAIL] could not connect: {type(exc).__name__}: {exc}")
        hint = _hint(exc)
        if hint:
            lines.append(f"       hint: {hint}")
        return False, lines
    finally:
        manager.shutdown()
    lines.append("[ok]   connected; the session is read-only and the account holds SELECT only")
    tables = list(schema.tables)
    if not tables:
        lines.append("[WARN] connected, but this account can see no tables "
                     "(grant SELECT on the tables, or check the database name)")
    else:
        lines.append(f"[ok]   {len(tables)} table(s) visible:")
        for table in tables[:25]:
            lines.append(f"         {table.name} ({len(table.columns)} columns)")
        if len(tables) > 25:
            lines.append(f"         ... and {len(tables) - 25} more")
    lines.extend(f"[WARN] {w}" for w in warnings)
    return True, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", help="profile name from the profiles file")
    parser.add_argument("--profiles", default="",
                        help="path to the profiles JSON (default: HAI_DATABASE_PROFILES)")
    parser.add_argument("--env-file", default=str(ROOT / ".env"),
                        help="settings file to load first, like the server does (default: .env)")
    args = parser.parse_args(argv)
    loaded = load_env_file(Path(args.env_file))
    if loaded:
        print(f"[ok]   loaded {len(loaded)} setting(s) from {args.env_file}")
    args.profiles = args.profiles or os.getenv("HAI_DATABASE_PROFILES", "")
    if not args.profiles:
        print("[FAIL] no profiles file: set HAI_DATABASE_PROFILES or pass --profiles <path>", file=sys.stderr)
        return 1
    try:
        profiles = load_profiles(args.profiles)
    except EngineError as exc:
        print(f"[FAIL] the profiles file is not valid: {exc}", file=sys.stderr)
        return 1
    ok, lines = check_profile(args.profile, profiles)
    print("\n".join(lines))
    print("\nResult: " + ("this profile will work in the web app." if ok else "fix the [FAIL] line above, then run this again."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
