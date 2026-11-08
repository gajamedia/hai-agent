from __future__ import annotations

"""Create the three limited accounts of a server from the browser, with an administrator password used once.

The administrator password arrives in one request, lives in local variables for the length of that request
(connect, run the statements, close) and is then dropped. It is never written to a file, to ``os.environ``, to
the profiles file, to the audit trail or to a log, and never returned. What IS stored are the three NEW
accounts' generated passwords, in ``.env`` exactly like the command line scripts do.

The accounts are named after the profile (``hai_rd_<name>``, ``hai_cr_<name>``, ``hai_wr_<name>``), so setting up
a second profile on the same server never resets the passwords another profile relies on.

Rights match ``scripts/setup_mysql_*_user.py`` and ``scripts/connect_sqlserver.py``:
  reader   SELECT only (MySQL) / db_datareader + db_denydatawriter (SQL Server)
  creator  CREATE on databases named hai_... only / CREATE ANY DATABASE
  writer   SELECT, INSERT, UPDATE, DELETE / db_datareader + db_datawriter
"""

import re
import secrets
import string
from typing import Any, Callable

from api.database_engines import SQLServerEngine, SQLServerProfile, _odbc_value

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
ROLES = ("reader", "creator", "writer")
_SYSTEM_MYSQL = {"information_schema", "mysql", "performance_schema", "sys"}
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_USER = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_PREFIX = "hai_"


class AdminSetupError(ValueError):
    """Something the user can fix; the message is safe to show (never contains a password)."""


def generate_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.islower() for c in candidate) and any(c.isupper() for c in candidate) \
                and any(c.isdigit() for c in candidate):
            return candidate


def account_names(name: str, kind: str) -> dict[str, str]:
    """Accounts derived from the profile name; MySQL user names are at most 32 characters."""
    keep = 25 if kind == "mysql" else 40
    return {"reader": f"hai_rd_{name[:keep]}", "creator": f"hai_cr_{name[:keep]}", "writer": f"hai_wr_{name[:keep]}"}


# ------------------------------------------------------------------------------- connecting
def _port(kind: str, port: Any) -> int:
    if port in (None, ""):
        return 3306 if kind == "mysql" else 0
    try:
        number = int(port)
    except (TypeError, ValueError):
        raise AdminSetupError("port must be a number") from None
    if not 0 <= number <= 65535:
        raise AdminSetupError("port must be between 0 and 65535")
    return number


def _admin_hint(exc: BaseException) -> str:
    """Why the administrator login failed, from the driver's code; never its raw text."""
    from api.database_profile_check import _hint
    code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
    if code == 1045:
        return "the administrator user or password is wrong"
    text = str(exc).lower()
    if "18456" in text or "login failed" in text:
        return "the administrator user or password is wrong, or this instance accepts Windows logins only"
    return _hint(exc) or "could not connect: check host, port, administrator user and password"


def open_admin(kind: str, host: str, port: int, user: str, password: str, *, database: str = "master",
               connect: Callable | None = None):
    """An autocommit administrator connection. ``connect`` is injectable for tests."""
    if kind == "mysql":
        if connect is None:
            import pymysql
            connect = pymysql.connect
        return connect(host=host, port=port or 3306, user=user, password=password, connect_timeout=5,
                       read_timeout=15, write_timeout=15, charset="utf8mb4", autocommit=True,
                       ssl={"check_hostname": False}, ssl_verify_cert=False)
    profile = SQLServerProfile(name="admin", host=host, port=port, database=database, user=user,
                               password_env="UNUSED")
    driver_connect, driver = SQLServerEngine({}, connect=connect)._driver(profile)
    server = f"{host},{port}" if port else host
    text = (f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={user};PWD={_odbc_value(password)};"
            "Encrypt=yes;TrustServerCertificate=yes;APP=hai-agent-setup")
    conn = driver_connect(text, timeout=5)
    conn.autocommit = True
    return conn


def _check_target(kind: str, host: Any, user: Any) -> tuple[str, str]:
    if kind not in {"mysql", "sqlserver"}:
        raise AdminSetupError("automatic account setup supports MySQL and SQL Server")
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.\-\\()]{1,100}", host.strip()):
        raise AdminSetupError("host is not valid")
    if not isinstance(user, str) or not _USER.match(user.strip()):
        raise AdminSetupError("administrator user is not valid")
    return host.strip(), user.strip()


def _run(kind, host, port, admin_user, admin_password, work, connect):
    if not isinstance(admin_password, str) or not admin_password:
        raise AdminSetupError("the administrator password is required")
    try:
        conn = open_admin(kind, host, port, admin_user, admin_password, connect=connect)
    except ImportError:
        raise AdminSetupError("the database driver is not installed on the server") from None
    except Exception as exc:
        raise AdminSetupError(_admin_hint(exc)) from None
    try:
        return work(conn)
    except AdminSetupError:
        raise
    except Exception as exc:
        raise AdminSetupError(_admin_hint(exc)) from None
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------- inspecting
def inspect(kind: str, host: Any, port: Any, admin_user: Any, admin_password: Any, name: str, *,
            connect: Callable | None = None) -> dict[str, Any]:
    """What is on the server and what would be done for this profile name. Changes nothing."""
    host, admin_user = _check_target(kind, host, admin_user)
    port = _port(kind, port)
    wanted = account_names(name, kind)

    def work(conn):
        cur = conn.cursor()
        if kind == "mysql":
            cur.execute("SELECT DISTINCT user FROM mysql.user WHERE user LIKE 'hai\\_%'")
            accounts = sorted(r[0] for r in cur.fetchall())
            cur.execute("SHOW DATABASES")
            databases = sorted(r[0] for r in cur.fetchall() if r[0] not in _SYSTEM_MYSQL)
        else:
            cur.execute("SELECT CAST(SERVERPROPERTY('IsIntegratedSecurityOnly') AS int), "
                        "CAST(IS_SRVROLEMEMBER('sysadmin') AS int)")
            windows_only, is_admin = cur.fetchone()
            if windows_only:
                raise AdminSetupError("this instance accepts Windows logins only, so SQL logins cannot be created; "
                                      "enable 'SQL Server and Windows Authentication mode' and restart the service")
            if not is_admin:
                raise AdminSetupError("that login is not a sysadmin of this instance")
            cur.execute("SELECT name FROM sys.server_principals WHERE type = 'S' AND name LIKE 'hai[_]%' ORDER BY name")
            accounts = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT name FROM sys.databases WHERE database_id > 4 AND state = 0 ORDER BY name")
            databases = [r[0] for r in cur.fetchall()]
        cur.close()
        return {"accounts": accounts, "databases": databases,
                "plan": {role: {"user": user, "exists": user in accounts} for role, user in wanted.items()}}

    return _run(kind, host, port, admin_user, admin_password, work, connect)


# ------------------------------------------------------------------------------ provisioning
def _mysql_statements(*, roles, names, passwords, database, hosts) -> list[tuple[str, tuple]]:
    pattern = "`" + _PREFIX[:-1] + "\\_%%`.*"          # `hai\_%`.* ; % doubled because values are parameters
    out: list[tuple[str, tuple]] = []
    for host in hosts:
        for role in roles:
            user = names[role]
            out += [("CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s", (user, host, passwords[role])),
                    ("ALTER USER %s@%s IDENTIFIED BY %s", (user, host, passwords[role])),
                    ("REVOKE ALL PRIVILEGES, GRANT OPTION FROM %s@%s", (user, host))]
        if "reader" in roles:
            if database:
                out.append((f"GRANT SELECT ON `{database}`.* TO %s@%s", (names["reader"], host)))
            if "creator" in roles:
                out.append((f"GRANT SELECT ON {pattern} TO %s@%s", (names["reader"], host)))
        if "creator" in roles:
            out.append((f"GRANT CREATE ON {pattern} TO %s@%s", (names["creator"], host)))
        if "writer" in roles:
            if database:
                out.append((f"GRANT SELECT, INSERT, UPDATE, DELETE ON `{database}`.* TO %s@%s", (names["writer"], host)))
            if "creator" in roles:
                out.append((f"GRANT SELECT, INSERT, UPDATE, DELETE ON {pattern} TO %s@%s", (names["writer"], host)))
    out.append(("FLUSH PRIVILEGES", ()))
    return out


def _login(user: str, password: str) -> str:
    return (f"IF SUSER_ID(N'{user}') IS NULL CREATE LOGIN [{user}] WITH PASSWORD = N'{password}', "
            f"CHECK_POLICY = OFF, DEFAULT_DATABASE = [master] ELSE ALTER LOGIN [{user}] WITH PASSWORD = N'{password}'")


def provision(kind: str, host: Any, port: Any, admin_user: Any, admin_password: Any, name: str, *,
              database: str = "", roles: tuple[str, ...] = ROLES, connect: Callable | None = None) -> dict[str, Any]:
    """Create or reset the chosen accounts. Returns ``{"names", "passwords", "reset"}``.

    The passwords are brand new and generated here; nothing about the administrator survives this call."""
    host, admin_user = _check_target(kind, host, admin_user)
    port = _port(kind, port)
    roles = tuple(r for r in ROLES if r in set(roles))
    if "reader" not in roles:
        raise AdminSetupError("the read account is always created")
    if database and not _IDENT.match(database):
        raise AdminSetupError("database: letters, digits and _ only")
    if not database and "creator" not in roles:
        raise AdminSetupError("give a database to read, or also create the account that creates databases")
    names = account_names(name, kind)
    passwords = {role: generate_password() for role in roles}

    def work(conn):
        existing = set(inspect_names(conn, kind))
        cur = conn.cursor()
        if kind == "mysql":
            hosts = ("localhost", "127.0.0.1") if host.lower() in LOCAL_HOSTS else ("%",)
            for sql, args in _mysql_statements(roles=roles, names=names, passwords=passwords,
                                               database=database, hosts=hosts):
                cur.execute(sql, args)
        else:
            for role in roles:
                cur.execute(_login(names[role], passwords[role]))
            if "creator" in roles:
                cur.execute(f"IF USER_ID(N'{names['creator']}') IS NULL CREATE USER [{names['creator']}] "
                            f"FOR LOGIN [{names['creator']}]")
                cur.execute(f"GRANT CREATE ANY DATABASE TO [{names['creator']}]")
            if database:
                inner = open_admin(kind, host, port, admin_user, admin_password, database=database, connect=connect)
                try:
                    c2 = inner.cursor()
                    c2.execute(f"IF USER_ID(N'{names['reader']}') IS NULL CREATE USER [{names['reader']}] "
                               f"FOR LOGIN [{names['reader']}]")
                    c2.execute(f"ALTER ROLE [db_datareader] ADD MEMBER [{names['reader']}]")
                    c2.execute(f"ALTER ROLE [db_denydatawriter] ADD MEMBER [{names['reader']}]")
                    if "writer" in roles:
                        c2.execute(f"IF USER_ID(N'{names['writer']}') IS NULL CREATE USER [{names['writer']}] "
                                   f"FOR LOGIN [{names['writer']}]")
                        c2.execute(f"ALTER ROLE [db_datareader] ADD MEMBER [{names['writer']}]")
                        c2.execute(f"ALTER ROLE [db_datawriter] ADD MEMBER [{names['writer']}]")
                finally:
                    inner.close()
        cur.close()
        return sorted(names[r] for r in roles if names[r] in existing)

    reset = _run(kind, host, port, admin_user, admin_password, work, connect)
    return {"names": {r: names[r] for r in roles}, "passwords": passwords, "reset": reset}


def inspect_names(conn, kind: str) -> list[str]:
    """Names of the existing hai_ accounts on the server."""
    cur = conn.cursor()
    if kind == "mysql":
        cur.execute("SELECT DISTINCT user FROM mysql.user WHERE user LIKE 'hai\\_%'")
    else:
        cur.execute("SELECT name FROM sys.server_principals WHERE type = 'S' AND name LIKE 'hai[_]%'")
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    return rows


__all__ = ["AdminSetupError", "ROLES", "account_names", "generate_password", "inspect", "provision"]
