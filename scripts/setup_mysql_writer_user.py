"""Create the MySQL account that may change data (INSERT, UPDATE, DELETE), in one step.

    python scripts/setup_mysql_writer_user.py local_mysql

This is separate from the read-only account on purpose: reports never use it, and only the
"Ubah data" page does, after a preview, a fresh login and an exact confirmation phrase.

What it does, after showing the plan and asking for a yes:
  1. connects as an administrator (default: root) with a password you type (never stored);
  2. creates (or resets) the writer account for 'localhost' and '127.0.0.1';
  3. revokes everything from it and grants SELECT, INSERT, UPDATE and DELETE on the profile's
     database only (or only on --tables). No DROP, ALTER, CREATE or GRANT OPTION, ever;
  4. writes the new random password into .env and adds a "writer" block to the profile in
     profiles.json (the file holds only the account and variable names);
  5. verifies the account the way the server will: it refuses a writer that can do more.

Run scripts/setup_mysql_readonly_user.py first. It is safe to run again (it resets the password).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from api.database_engines import EngineError, MySQLEngine, load_profiles  # noqa: E402
from api.dotenv_tools import load_env_file, set_env_value  # noqa: E402
from setup_mysql_readonly_user import (  # noqa: E402
    HOSTS, SetupError, _ACCOUNT, _HOST, _IDENT, _connect, generate_password,
)

DEFAULT_WRITER = "hai_writer"
PRIVILEGES = "SELECT, INSERT, UPDATE, DELETE"


def plan_statements(database: str, writer: str, password: str, hosts: tuple[str, ...] = HOSTS,
                    tables: tuple[str, ...] = ()) -> list[tuple[str, tuple]]:
    if not _IDENT.match(database):
        raise SetupError("the database name may only contain letters, digits and underscores")
    if not _ACCOUNT.match(writer):
        raise SetupError("the account name may only contain letters, digits and underscores (max 32)")
    if not hosts or any(not _HOST.match(h) for h in hosts):
        raise SetupError("each host may only contain letters, digits and . : % _ -")
    if any(not _IDENT.match(t) for t in tables):
        raise SetupError("table names may only contain letters, digits and underscores")
    scopes = [f"`{database}`.`{t}`" for t in tables] or [f"`{database}`.*"]
    statements: list[tuple[str, tuple]] = []
    for host in hosts:
        statements += [
            ("CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s", (writer, host, password)),
            ("ALTER USER %s@%s IDENTIFIED BY %s", (writer, host, password)),
            ("REVOKE ALL PRIVILEGES, GRANT OPTION FROM %s@%s", (writer, host)),
            *[(f"GRANT {PRIVILEGES} ON {scope} TO %s@%s", (writer, host)) for scope in scopes],
        ]
    statements.append(("FLUSH PRIVILEGES", ()))
    return statements


def run(*, profile_name: str, profiles: dict[str, Any], profiles_path: Path, env_path: Path,
        admin_user: str, admin_password: str, writer_user: str, tables: tuple[str, ...],
        confirm: Callable[[str], bool], connect: Callable | None = None,
        out: Callable[[str], None] = print, hosts: tuple[str, ...] = HOSTS) -> int:
    profile = profiles.get(profile_name)
    if profile is None:
        raise SetupError(f"no profile named '{profile_name}'. Profiles in the file: "
                         f"{', '.join(sorted(profiles)) or '(none)'}")
    if profile.type != "mysql":
        raise SetupError("this tool only supports MySQL profiles")
    if writer_user.lower() in {admin_user.lower(), profile.user.lower()}:
        raise SetupError(f"the writer must be its own account: not '{admin_user}' (administrator) and "
                         f"not '{profile.user}' (the read-only account)")
    if not admin_password:
        raise SetupError("an administrator password is required")
    password_env = (profile.writer_password_env
                    or f"HAI_{re.sub(r'[^A-Za-z0-9]', '_', profile_name).upper()}_WRITER_PASSWORD")

    scope = ", ".join(f"`{profile.database}`.`{t}`" for t in tables) or f"`{profile.database}`.*"
    out(f"Server    : {profile.host}:{profile.port} (encrypted: {profile.ssl})")
    out(f"Admin     : {admin_user}   (used only now; its password is not stored)")
    out(f"New access: account '{writer_user}' for {' and '.join(hosts)}: {PRIVILEGES} on {scope}")
    out("            nothing else: no DROP, ALTER, CREATE or GRANT OPTION")
    out(f"Will write: the new password into {env_path.name} ({password_env}) and a writer block into "
        f"{profiles_path.name}")
    if not confirm("Continue?"):
        out("Nothing was changed.")
        return 1

    password = generate_password()
    statements = plan_statements(profile.database, writer_user, password, hosts, tables)
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
            raise SetupError(f"the database '{profile.database}' does not exist on this server")
        for sql, args in statements:
            cursor.execute(sql, args)
        cursor.close()
    finally:
        try:
            connection.close()
        except Exception:
            pass
    out(f"[ok]   account created or reset, privileges limited to {PRIVILEGES}")

    set_env_value(env_path, password_env, password)
    data = json.loads(profiles_path.read_text(encoding="utf-8-sig"))
    data[profile_name]["writer"] = {"user": writer_user, "password_env": password_env}
    profiles_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out(f"[ok]   wrote {password_env} to {env_path.name} and the writer block to {profiles_path.name}")

    os.environ[password_env] = password
    verified = load_profiles(profiles_path)
    try:
        MySQLEngine(verified, connect=connect).open_writer(profile_name)
    except Exception as exc:
        raise SetupError(f"the account was created but the final check failed: {exc}") from exc
    del password
    out("[ok]   the server accepts this writer account (only data-changing rights on this database)")
    out("\nDone. Restart the backend, give your Keycloak user the roles data-writer (and a second person "
        "data-change-approver), then in Data Reports press 'Ubah persetujuan' and tick the tables and "
        "columns that may be changed.")
    return 0


def main(argv: list[str] | None = None, *, connect: Callable | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", help="profile name in the profiles file (e.g. local_mysql)")
    parser.add_argument("--admin-user", default="root")
    parser.add_argument("--writer-user", default=DEFAULT_WRITER)
    parser.add_argument("--tables", default="", help="comma separated tables the writer may change "
                        "(default: every table of the database)")
    parser.add_argument("--profiles", default="", help="profiles JSON (default: HAI_DATABASE_PROFILES)")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--hosts", default=",".join(HOSTS))
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
                   writer_user=args.writer_user,
                   tables=tuple(t.strip() for t in args.tables.split(",") if t.strip()),
                   confirm=confirm, connect=connect,
                   hosts=tuple(h.strip() for h in args.hosts.split(",") if h.strip()))
    except (SetupError, EngineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
