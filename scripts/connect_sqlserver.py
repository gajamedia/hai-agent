"""Check the SQL Server installed on this computer, and (optionally) create a read-only account for the agent.

    python scripts/connect_sqlserver.py                                 # diagnose and test the connection
    python scripts/connect_sqlserver.py --create-reader --database dbsales
    python scripts/connect_sqlserver.py --create-creator                # lets the agent create databases
    python scripts/connect_sqlserver.py --server "10.62.10.34,1433" --admin-user sa --create-reader --database db1

Diagnose (changes nothing):
  * lists the SQL Server instances installed here (Windows registry), whether each service is running,
    whether TCP/IP is enabled and on which port, and which authentication modes it accepts;
  * checks the ODBC driver and the Python package (pyodbc) the agent needs;
  * connects with YOUR Windows login (no password) and reports the version, whether you are an administrator
    of the instance, and the user databases.

--create-reader (after showing the plan and asking for a yes):
  * creates a SQL login with a random password, a user for it in ONE database, and gives it read-only access
    (db_datareader) while explicitly denying writes and schema changes;
  * writes the password into .env (variable HAI_<PROFILE>_PASSWORD) and a profile, without the password, into
    profiles.json (HAI_DATABASE_PROFILES), which makes it appear in the Data Reports profile list after the
    backend restarts;
  * verifies the account: it can read, and a write is refused.
  It needs an instance that accepts SQL logins (mixed mode) and you being an administrator of it. It never
  changes server settings, never enables TCP/IP and never touches any other database.

--create-creator (same checks, same confirmation) lets the "Buat database" page create databases on this server:
  * creates a login that holds ONE server permission, CREATE ANY DATABASE (not the dbcreator role, which also
    allows altering other databases, and never sysadmin). It owns only what it creates;
  * writes its password into .env and a "creator" block into the profile that --create-reader made.
  SQL Server cannot limit that permission to names starting with a prefix, so the application enforces the
  prefix (default hai_) and the databases it makes give the read-only login read access, nothing more.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import secrets
import string
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.dotenv_tools import set_env_value  # noqa: E402

DRIVER_PREFERENCE = ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server", "ODBC Driver 13 for SQL Server")
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_PROFILE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_INSTANCE = re.compile(r"^[A-Za-z0-9_$]{1,64}$")
SQL_ROOT = r"SOFTWARE\Microsoft\Microsoft SQL Server"
SYSTEM_DATABASES = {"master", "tempdb", "model", "msdb"}


class SetupError(RuntimeError):
    """Something the user can fix; the message says what."""


def generate_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.islower() for c in candidate) and any(c.isupper() for c in candidate) \
                and any(c.isdigit() for c in candidate):
            return candidate


# ------------------------------------------------------------------ what is installed
def read_instances(winreg_module: Any = None) -> list[dict[str, Any]]:
    """Instances from the registry. Empty on a computer without SQL Server (or not Windows)."""
    if winreg_module is None:
        if os.name != "nt":
            return []
        import winreg as winreg_module  # type: ignore[no-redef]
    w = winreg_module

    def open_key(path: str):
        return w.OpenKey(w.HKEY_LOCAL_MACHINE, path)

    def value(key, name, default=None):
        try:
            return w.QueryValueEx(key, name)[0]
        except OSError:
            return default

    found: list[dict[str, Any]] = []
    try:
        with open_key(rf"{SQL_ROOT}\Instance Names\SQL") as names:
            index, pairs = 0, []
            while True:
                try:
                    pairs.append(w.EnumValue(names, index)[:2])
                    index += 1
                except OSError:
                    break
    except OSError:
        return []
    for name, folder in pairs:
        info: dict[str, Any] = {"name": name, "folder": folder, "tcp_enabled": None, "tcp_port": None,
                                "login_mode": None}
        try:
            with open_key(rf"{SQL_ROOT}\{folder}\MSSQLServer") as key:
                info["login_mode"] = value(key, "LoginMode")          # 1 = Windows only, 2 = mixed
            with open_key(rf"{SQL_ROOT}\{folder}\MSSQLServer\SuperSocketNetLib\Tcp") as key:
                info["tcp_enabled"] = bool(value(key, "Enabled", 0))
            with open_key(rf"{SQL_ROOT}\{folder}\MSSQLServer\SuperSocketNetLib\Tcp\IPAll") as key:
                port = str(value(key, "TcpPort", "") or "").strip()
                dynamic = str(value(key, "TcpDynamicPorts", "") or "").strip()
                info["tcp_port"] = port or (dynamic if dynamic not in ("", "0") else None)
        except OSError:
            pass
        found.append(info)
    return found


def server_string(instance: str) -> str:
    """How to reach an instance on this computer (shared memory, no TCP/IP needed)."""
    return "(local)" if instance.upper() == "MSSQLSERVER" else rf"(local)\{instance}"


def service_name(instance: str) -> str:
    return "MSSQLSERVER" if instance.upper() == "MSSQLSERVER" else f"MSSQL${instance}"


def pick_driver(drivers: list[str]) -> str | None:
    for wanted in DRIVER_PREFERENCE:
        if wanted in drivers:
            return wanted
    return None


def connection_string(driver: str, server: str, *, database: str = "master", user: str = "", password: str = "") -> str:
    # Braces make the password literal, so a ; or = in it cannot add connection keywords.
    login = f"UID={user};PWD={{{password.replace('}', '}}')}}}" if user else "Trusted_Connection=yes"
    # Encrypted, but the server certificate is not verified (the profile gets ssl "require", the same).
    return f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};{login};TrustServerCertificate=yes;Encrypt=yes"


def split_server(server: str) -> tuple[str, int]:
    """'10.62.10.34,1433' -> ('10.62.10.34', 1433); '(local)\\SQLEXPRESS' -> ('(local)\\SQLEXPRESS', 0)."""
    host, comma, port = server.partition(",")
    if not comma:
        return server, 0
    if not port.strip().isdigit() or not 1 <= int(port) <= 65535:
        raise SetupError(f"'{server}': the port after the comma must be a number from 1 to 65535")
    return host.strip(), int(port)


def profile_entry(server: str, database: str, login: str, env_var: str) -> dict[str, Any]:
    """The profile as the app will read it, validated here so a bad one is never written."""
    from api.database_engines import EngineError, parse_profiles
    host, port = split_server(server)
    entry: dict[str, Any] = {"type": "sqlserver", "server": host, "database": database, "user": login,
                             "password_env": env_var, "ssl": "require"}
    if port:
        entry["port"] = port
    try:
        parse_profiles({"check_only": entry})
    except EngineError as exc:
        raise SetupError(f"the server or names cannot be used in a profile: {exc}") from exc
    return entry


# ------------------------------------------------------------------ diagnose
def diagnose(conn, out: Callable[[str], None]) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT CAST(SERVERPROPERTY('ProductVersion') AS varchar(40)), "
                "CAST(SERVERPROPERTY('Edition') AS varchar(80)), "
                "CAST(SERVERPROPERTY('IsIntegratedSecurityOnly') AS int), "
                "CAST(IS_SRVROLEMEMBER('sysadmin') AS int), SUSER_SNAME()")
    version, edition, windows_only, is_admin, who = cur.fetchone()
    cur.execute("SELECT name FROM sys.databases WHERE database_id > 4 AND state = 0 ORDER BY name")
    databases = [row[0] for row in cur.fetchall()]
    out(f"  connected as {who}: SQL Server {version} ({edition})")
    out(f"  accepts SQL logins: {'no (Windows login only)' if windows_only else 'yes (mixed mode)'}; "
        f"you are {'an administrator' if is_admin else 'NOT an administrator'} of this instance")
    out("  user databases: " + (", ".join(databases) if databases else "(none)"))
    return {"windows_only": bool(windows_only), "is_admin": bool(is_admin), "databases": databases}


# ------------------------------------------------------------------ create the read-only account
def plan_statements(login: str, database: str, password: str) -> dict[str, list[str]]:
    """Every name is validated and quoted; the password is letters and digits only."""
    for name in (login, database):
        if not _IDENT.match(name):
            raise SetupError(f"'{name}': letters, digits and _ only, starting with a letter")
    return {
        "master": [f"IF SUSER_ID(N'{login}') IS NULL CREATE LOGIN [{login}] WITH PASSWORD = N'{password}', "
                   "CHECK_POLICY = OFF, DEFAULT_DATABASE = [master] "
                   f"ELSE ALTER LOGIN [{login}] WITH PASSWORD = N'{password}'"],
        database: [
            f"IF USER_ID(N'{login}') IS NULL CREATE USER [{login}] FOR LOGIN [{login}]",
            f"ALTER ROLE [db_datareader] ADD MEMBER [{login}]",
            f"ALTER ROLE [db_denydatawriter] ADD MEMBER [{login}]",
        ],
    }


def create_reader(*, admin_connect: Callable[[str], Any], reader_connect: Callable[[str, str, str], Any],
                  database: str, login: str, profile: str, server: str, info: dict[str, Any],
                  env_path: Path, profiles_path: Path, ask: Callable[[str], str],
                  out: Callable[[str], None]) -> int:
    if info["windows_only"]:
        raise SetupError("This instance accepts Windows logins only, so a SQL login cannot be created. Enable "
                         "'SQL Server and Windows Authentication mode' (SSMS: server Properties > Security) and "
                         "restart the SQL Server service, then run this again. This script never changes it.")
    if not info["is_admin"]:
        raise SetupError("Your Windows login is not an administrator (sysadmin) of this instance.")
    if database not in info["databases"]:
        raise SetupError(f"database '{database}' not found. Available: {', '.join(info['databases']) or '(none)'}")
    if not _PROFILE.match(profile):
        raise SetupError("profile: 2-32 characters, lowercase letters, digits and _, starting with a letter")
    password = generate_password()
    plan = plan_statements(login, database, password)
    out(f"\nPlan on {server}:")
    out(f"  1. create (or reset the password of) the SQL login '{login}' with a random password")
    out(f"  2. give it read-only access to the database '{database}' only (read yes; insert, update, delete,")
    out("     schema changes denied)")
    out(f"  3. write the password to {env_path.name} as HAI_{profile.upper()}_PASSWORD and the profile '{profile}'")
    out(f"     (no password) to {profiles_path.name}")
    if ask("Continue? [y/N] ").strip().lower() != "y":
        out("Nothing changed.")
        return 1
    conn = admin_connect("master")
    conn.autocommit = True
    cur = conn.cursor()
    for sql in plan["master"]:
        cur.execute(sql)
    conn.close()
    conn = admin_connect(database)
    conn.autocommit = True
    cur = conn.cursor()
    for sql in plan[database]:
        cur.execute(sql)
    conn.close()
    env_var = f"HAI_{profile.upper()}_PASSWORD"
    set_env_value(env_path, env_var, password)
    profiles = json.loads(profiles_path.read_text("utf-8-sig")) if profiles_path.exists() else {}
    profiles[profile] = profile_entry(server, database, login, env_var)
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")
    out(f"[ok] account created; password stored in {env_path.name}, profile '{profile}' in {profiles_path.name}")

    reader = reader_connect(database, login, password)
    cur = reader.cursor()
    cur.execute("SELECT COUNT(*) FROM sys.tables")
    out(f"[ok] the account can read ({cur.fetchone()[0]} tables visible)")
    try:
        cur.execute("CREATE TABLE hai_should_fail (x int)")
    except Exception:  # noqa: BLE001 - any refusal proves the point
        out("[ok] a write attempt was refused, as intended")
    else:
        raise SetupError("the account could write; remove it ('DROP LOGIN') and investigate before using it")
    finally:
        reader.close()
    return 0


def plan_creator_statements(login: str, password: str) -> list[str]:
    if not _IDENT.match(login):
        raise SetupError(f"'{login}': letters, digits and _ only, starting with a letter")
    return [
        f"IF SUSER_ID(N'{login}') IS NULL CREATE LOGIN [{login}] WITH PASSWORD = N'{password}', "
        "CHECK_POLICY = OFF, DEFAULT_DATABASE = [master] "
        f"ELSE ALTER LOGIN [{login}] WITH PASSWORD = N'{password}'",
        f"IF USER_ID(N'{login}') IS NULL CREATE USER [{login}] FOR LOGIN [{login}]",   # CREATE ANY DATABASE is
        f"GRANT CREATE ANY DATABASE TO [{login}]",                                    # granted to the master user
    ]


def create_creator(*, admin_connect: Callable[[str], Any], creator_connect: Callable[[str, str], Any],
                   login: str, profile: str, info: dict[str, Any], env_path: Path, profiles_path: Path,
                   ask: Callable[[str], str], out: Callable[[str], None]) -> int:
    from api.database_engines import check_sqlserver_creator_permissions
    if info["windows_only"]:
        raise SetupError("This instance accepts Windows logins only, so a SQL login cannot be created. Enable "
                         "'SQL Server and Windows Authentication mode' and restart the SQL Server service first.")
    if not info["is_admin"]:
        raise SetupError("Your Windows login is not an administrator (sysadmin) of this instance.")
    profiles = json.loads(profiles_path.read_text("utf-8-sig")) if profiles_path.exists() else {}
    existing = profiles.get(profile)
    if not isinstance(existing, dict) or existing.get("type") != "sqlserver":
        raise SetupError(f"profile '{profile}' is not a SQL Server profile in {profiles_path.name}. Run "
                         "--create-reader first (or pass --profile with the name it created).")
    if login == existing.get("user"):
        raise SetupError("the creator must be a different login than the read-only one")
    password = generate_password()
    statements = plan_creator_statements(login, password)
    env_var = f"HAI_{profile.upper()}_CREATOR_PASSWORD"
    out(f"\nPlan on this server for profile '{profile}':")
    out(f"  1. create (or reset the password of) the SQL login '{login}' with a random password")
    out("  2. give it ONE server permission: CREATE ANY DATABASE (it owns only the databases it creates; it")
    out("     cannot touch, alter or drop any other database)")
    out(f"  3. write the password to {env_path.name} as {env_var} and a creator block to {profiles_path.name}")
    if ask("Continue? [y/N] ").strip().lower() != "y":
        out("Nothing changed.")
        return 1
    conn = admin_connect("master")
    conn.autocommit = True
    cur = conn.cursor()
    for sql in statements:
        cur.execute(sql)
    conn.close()
    set_env_value(env_path, env_var, password)
    existing["creator"] = {"user": login, "password_env": env_var, "prefix": existing.get("creator", {}).get("prefix", "hai_")}
    profiles_path.write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")
    out(f"[ok] login created; password stored in {env_path.name}, creator block added to profile '{profile}'")
    check = creator_connect(login, password)
    cur = check.cursor()
    cur.execute("SELECT permission_name FROM sys.fn_my_permissions(NULL, 'SERVER')")
    perms = [row[0] for row in cur.fetchall()]
    from api.database_engines import _server_roles
    roles = _server_roles(check)
    check.close()
    try:
        check_sqlserver_creator_permissions(perms, roles)
    except Exception as exc:  # noqa: BLE001
        raise SetupError(f"the new login did not pass the safety check: {exc}") from exc
    out("[ok] the login can create databases and holds nothing else. Restart the backend; the page "
        "'Buat database' now lists this server.")
    return 0


# ------------------------------------------------------------------ main
def _default_profiles(env_path: Path) -> Path:
    value = os.getenv("HAI_DATABASE_PROFILES", "")
    if not value and env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("HAI_DATABASE_PROFILES="):
                value = line.partition("=")[2].strip().strip('"')
    return Path(value) if value else ROOT / "data" / "reports" / "profiles.json"


def main(argv: list[str] | None = None, *, pyodbc_module: Any = None, instances_reader=read_instances,
         service_state: Callable[[str], str] | None = None, ask: Callable[[str], str] = input,
         ask_secret: Callable[[str], str] = getpass.getpass,
         out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instance", default="", help="instance name (default: the first one found)")
    parser.add_argument("--server", default="", help="override the server string, e.g. (local)\\SQLEXPRESS or host,1433")
    parser.add_argument("--admin-user", default="", help="administrator SQL login (e.g. sa) for a server on another "
                        "computer; its password is asked here and never stored. Default: your Windows login")
    parser.add_argument("--create-reader", action="store_true")
    parser.add_argument("--create-creator", action="store_true", help="let the agent create databases (CREATE ANY DATABASE)")
    parser.add_argument("--creator-user", default="hai_sql_creator")
    parser.add_argument("--database", default="", help="database the read-only account may read")
    parser.add_argument("--user", default="hai_sql_reader")
    parser.add_argument("--profile", default="local_sqlserver")
    parser.add_argument("--env", default=str(ROOT / ".env"))
    parser.add_argument("--profiles", default="", help="profiles JSON (default: HAI_DATABASE_PROFILES in .env)")
    args = parser.parse_args(argv)

    if service_state is None:
        def service_state(name: str) -> str:  # noqa: E306
            import subprocess
            res = subprocess.run(["sc", "query", name], capture_output=True, text=True, timeout=15, check=False)
            match = re.search(r"STATE\s*:\s*\d+\s+(\w+)", res.stdout)
            return match.group(1).lower() if match else "unknown"

    out("SQL Server instances on this computer:")
    instances = instances_reader()
    if not instances:
        out("  none found. Install SQL Server (Express is free), or pass --server <host,port> for another computer.")
    for item in instances:
        state = service_state(service_name(item["name"]))
        tcp = (f"TCP/IP on, port {item['tcp_port'] or 'dynamic'}" if item["tcp_enabled"]
               else "TCP/IP off (reachable from this computer through shared memory; other computers cannot connect)")
        modes = {1: "Windows login only", 2: "Windows or SQL login"}.get(item["login_mode"], "unknown")
        out(f"  - {item['name']}: service {state}; {tcp}; accepts {modes}")

    if pyodbc_module is None:
        try:
            import pyodbc as pyodbc_module  # type: ignore[no-redef]
        except ImportError:
            out("\nThe Python package pyodbc is missing: pip install pyodbc  (or pip install -e .[sqlserver])")
            return 1
    driver = pick_driver(list(pyodbc_module.drivers()))
    if driver is None:
        out("\nNo 'ODBC Driver 17/18 for SQL Server' found. Install Microsoft's ODBC driver for SQL Server.")
        return 1
    out(f"\nODBC driver: {driver}")

    chosen = next((i for i in instances if i["name"].lower() == args.instance.lower()), None) if args.instance \
        else (instances[0] if instances else None)
    if args.server:
        server = args.server
    elif chosen:
        server = server_string(chosen["name"])
    else:
        out("No server to connect to: use --server.")
        return 1
    if args.instance and not chosen and not args.server:
        out(f"Instance '{args.instance}' was not found.")
        return 1
    if chosen and not _INSTANCE.match(chosen["name"]):
        out("Unexpected instance name.")
        return 1

    admin_password = ""
    if args.admin_user:
        admin_password = ask_secret(f"Password of {args.admin_user} (not shown, not stored): ")
        out(f"Connecting to {server} as {args.admin_user}...")
    else:
        out(f"Connecting to {server} with your Windows login...")

    def connect(database: str = "master", user: str = "", password: str = ""):
        return pyodbc_module.connect(connection_string(driver, server, database=database, user=user,
                                                       password=password), timeout=8)

    def connect_admin(database: str = "master"):
        return connect(database, args.admin_user, admin_password) if args.admin_user else connect(database)

    try:
        conn = connect_admin()
    except Exception as exc:  # noqa: BLE001 - the driver's message is the useful part
        out(f"  could not connect: {str(exc)[:300]}")
        return 1
    try:
        info = diagnose(conn, out)
    finally:
        conn.close()

    if args.create_creator:
        profiles_path = Path(args.profiles) if args.profiles else _default_profiles(Path(args.env))
        try:
            return create_creator(
                admin_connect=connect_admin, creator_connect=lambda u, p: connect("master", u, p),
                login=args.creator_user, profile=args.profile, info=info, env_path=Path(args.env),
                profiles_path=profiles_path, ask=ask, out=out)
        except SetupError as exc:
            out(f"\nerror: {exc}")
            return 1
    if not args.create_reader:
        out("\nTo create a read-only account and register it for the agent: "
            "python scripts\\connect_sqlserver.py --create-reader --database <name>\n"
            "To let the agent create databases: python scripts\\connect_sqlserver.py --create-creator")
        return 0
    if not args.database:
        out("\n--create-reader needs --database <name>.")
        return 2
    profiles_path = Path(args.profiles) if args.profiles else _default_profiles(Path(args.env))
    try:
        return create_reader(
            admin_connect=connect_admin, reader_connect=lambda db, u, p: connect(db, u, p),
            database=args.database, login=args.user, profile=args.profile, server=server, info=info,
            env_path=Path(args.env), profiles_path=profiles_path, ask=ask, out=out)
    except SetupError as exc:
        out(f"\nerror: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
