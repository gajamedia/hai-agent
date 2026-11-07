"""Create the read-only MySQL account for a profile, in one step, with a random password.

    python scripts/setup_mysql_readonly_user.py local_mysql

What it does, after showing you the plan and asking for a yes:
  1. connects as an administrator (default: root) using a password you type; the password is
     never shown, never stored and never written anywhere;
  2. creates (or resets) the read-only account named in the profile's `user` field, or
     --reader-user, for 'localhost' and '127.0.0.1';
  3. revokes everything from it and grants SELECT on the profile's database only;
  4. writes the new random password into .env (the variable named by the profile's password_env)
     and points profiles.json at the read-only account;
  5. verifies the result the way the server will (same checks as check_database_profile.py).

The other databases on the server stay out of its reach. It is safe to run again: it resets the
account's password and rewrites .env to match. The administrator password is used only for this
run. Never put an administrator account in a profile: the app refuses it, on purpose.
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

from api.database_engines import EngineError, MySQLProfile, load_profiles  # noqa: E402
from api.database_profile_check import check_profile  # noqa: E402
from api.dotenv_tools import load_env_file, set_env_value  # noqa: E402

HOSTS = ("localhost", "127.0.0.1")
DEFAULT_READER = "hai_reader"
_HOST = re.compile(r"^[A-Za-z0-9_.:%\-]{1,60}$")
_IDENT = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_ACCOUNT = re.compile(r"^[A-Za-z0-9_]{1,32}$")


class SetupError(RuntimeError):
    """Something the user can fix; the message says what."""


def generate_password(length: int = 24) -> str:
    """Letters and digits only: nothing a .env file or a SQL client could mangle."""
    alphabet = string.ascii_letters + string.digits
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.islower() for c in candidate) and any(c.isupper() for c in candidate) \
                and any(c.isdigit() for c in candidate):
            return candidate


def plan_statements(database: str, reader: str, password: str,
                    hosts: tuple[str, ...] = HOSTS) -> list[tuple[str, tuple]]:
    """The exact statements, with values passed as parameters (the driver escapes them)."""
    if not _IDENT.match(database):
        raise SetupError("the database name may only contain letters, digits and underscores")
    if not _ACCOUNT.match(reader):
        raise SetupError("the account name may only contain letters, digits and underscores (max 32)")
    if not hosts or any(not _HOST.match(h) for h in hosts):
        raise SetupError("each host may only contain letters, digits and . : % _ -")
    statements: list[tuple[str, tuple]] = []
    for host in hosts:
        statements += [
            ("CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s", (reader, host, password)),
            ("ALTER USER %s@%s IDENTIFIED BY %s", (reader, host, password)),
            ("REVOKE ALL PRIVILEGES, GRANT OPTION FROM %s@%s", (reader, host)),
            (f"GRANT SELECT ON `{database}`.* TO %s@%s", (reader, host)),
        ]
    statements.append(("FLUSH PRIVILEGES", ()))
    return statements


def _connect(profile: MySQLProfile, admin_user: str, admin_password: str, connect: Callable | None):
    if connect is None:
        import pymysql  # optional dependency
        connect = pymysql.connect
    tls: dict[str, Any] = {}
    if profile.ssl == "require":
        tls = {"ssl": {"check_hostname": False}, "ssl_verify_cert": False}
    elif profile.ssl == "verify":
        tls = {"ssl": {}, "ssl_verify_cert": True, "ssl_verify_identity": True}
    return connect(host=profile.host, port=profile.port, user=admin_user, password=admin_password,
                   connect_timeout=5, read_timeout=15, write_timeout=15, charset="utf8mb4",
                   autocommit=True, **tls)


def run(*, profile_name: str, profiles: dict[str, Any], profiles_path: Path, env_path: Path,
        admin_user: str, admin_password: str, reader_user: str | None,
        confirm: Callable[[str], bool], connect: Callable | None = None,
        out: Callable[[str], None] = print, hosts: tuple[str, ...] = HOSTS) -> int:
    profile = profiles.get(profile_name)
    if profile is None:
        raise SetupError(f"no profile named '{profile_name}'. Profiles in the file: "
                         f"{', '.join(sorted(profiles)) or '(none)'}")
    if profile.type != "mysql":
        raise SetupError("this tool only supports MySQL profiles")
    reader = reader_user or profile.user
    if not reader_user and reader.lower() == admin_user.lower():
        # The profile currently names an administrator (for example root). That account is
        # exactly what the app refuses, so create a dedicated read-only one instead.
        out(f"Note      : the profile's user '{reader}' is the administrator account; "
            f"a separate read-only account '{DEFAULT_READER}' will be created instead")
        reader = DEFAULT_READER
    if reader.lower() == admin_user.lower():
        raise SetupError(f"the read-only account must not be the administrator account ('{admin_user}')")
    if not admin_password:
        raise SetupError("an administrator password is required")

    out(f"Server    : {profile.host}:{profile.port} (encrypted: {profile.ssl})")
    out(f"Database  : {profile.database}")
    out(f"Admin     : {admin_user}   (used only now; its password is not stored)")
    out(f"New access: account '{reader}' for {' and '.join(hosts)}, SELECT on `{profile.database}` only")
    out(f"Will write: the new password into {env_path.name} ({profile.password_env}) and set the "
        f"profile user to '{reader}' in {profiles_path.name}")
    if not confirm("Continue?"):
        out("Nothing was changed.")
        return 1

    password = generate_password()
    statements = plan_statements(profile.database, reader, password, hosts)
    try:
        connection = _connect(profile, admin_user, admin_password, connect)
    except ImportError:
        raise SetupError('the MySQL driver is not installed. Run: pip install "PyMySQL>=1.1,<2"')
    except Exception as exc:
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
        if code == 1045:
            raise SetupError("MySQL rejected the administrator login: wrong administrator "
                             "user or password") from exc
        raise SetupError(f"could not connect to MySQL: {type(exc).__name__}: {exc}") from exc
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s",
                       (profile.database,))
        if cursor.fetchone() is None:
            raise SetupError(f"the database '{profile.database}' does not exist on this server "
                             "(check the name in profiles.json)")
        for sql, args in statements:
            cursor.execute(sql, args)
        cursor.close()
    finally:
        try:
            connection.close()
        except Exception:
            pass
    out("[ok]   account created or reset, privileges limited to SELECT")

    set_env_value(env_path, profile.password_env, password)
    data = json.loads(profiles_path.read_text(encoding="utf-8-sig"))
    data[profile_name]["user"] = reader
    profiles_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out(f"[ok]   wrote {profile.password_env} to {env_path.name} ({len(password)} characters) "
        f"and set the profile user to '{reader}'")

    # Verify the way the server will, with the profile as it is now on disk.
    os.environ[profile.password_env] = password
    verified = load_profiles(profiles_path)
    ok, lines = check_profile(profile_name, verified)
    for line in lines:
        out(line)
    del password
    if not ok:
        raise SetupError("the account was created but the final check failed; see the lines above")
    out("\nDone. Restart the backend (Ctrl+C, then start it again) so it reads the new .env, "
        "then reload the page and press Hubungkan.")
    return 0


def main(argv: list[str] | None = None, *, connect: Callable | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", help="profile name in the profiles file (e.g. local_mysql)")
    parser.add_argument("--admin-user", default="root", help="MySQL administrator (default: root)")
    parser.add_argument("--reader-user", default=None,
                        help="read-only account name (default: the profile's user)")
    parser.add_argument("--profiles", default="", help="profiles JSON (default: HAI_DATABASE_PROFILES)")
    parser.add_argument("--env-file", default=str(ROOT / ".env"), help="settings file (default: .env)")
    parser.add_argument("--hosts", default=",".join(HOSTS),
                        help="hosts the account may connect from (default: localhost,127.0.0.1). If the "
                             "check later says Access denied for 'user'@'<host>', add that host here")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args(argv)

    env_path = Path(args.env_file)
    load_env_file(env_path)
    profiles_path = Path(args.profiles or os.getenv("HAI_DATABASE_PROFILES", ""))
    if not str(profiles_path) or str(profiles_path) == ".":
        print("error: no profiles file: set HAI_DATABASE_PROFILES or pass --profiles <path>", file=sys.stderr)
        return 1
    try:
        profiles = load_profiles(profiles_path)
        admin_password = os.getenv("HAI_MYSQL_ADMIN_PASSWORD") or getpass.getpass(
            f"Password of MySQL account '{args.admin_user}' (hidden): ")
        confirm = (lambda _q: True) if args.yes else (
            lambda question: input(f"{question} [y/N] ").strip().lower() in {"y", "yes"})
        return run(profile_name=args.profile, profiles=profiles, profiles_path=profiles_path,
                   env_path=env_path, admin_user=args.admin_user, admin_password=admin_password,
                   reader_user=args.reader_user, confirm=confirm, connect=connect,
                   hosts=tuple(h.strip() for h in args.hosts.split(",") if h.strip()))
    except (SetupError, EngineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
