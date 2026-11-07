from __future__ import annotations

"""Step by step check of a database profile; the logic behind scripts/check_database_profile.py.

Runs the very same code path as the server (read-only session, SELECT-only account check,
schema discovery) and reports the real cause of a failure. Never includes the password.
"""

import os
from typing import Any, Mapping

from agent.database import ConnectionManager
from api.database_engines import EngineError, MySQLEngine, PostgresEngine

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


__all__ = ["check_profile"]
