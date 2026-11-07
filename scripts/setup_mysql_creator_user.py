"""Let the agent create NEW databases whose name starts with a prefix (default hai_), and nothing else.

    python scripts/setup_mysql_creator_user.py local_mysql

What it does, after showing the plan and asking for a yes:
  1. connects as an administrator (default: root) with a password you type (never stored);
  2. creates (or resets) a creator account that holds only CREATE on the pattern hai_% : it can make
     databases named hai_..., and cannot drop, alter or even read anything; existing databases such as
     dbsales are out of its reach;
  3. gives the profile's read-only account SELECT, and its writer account (if any) SELECT, INSERT,
     UPDATE and DELETE, on the same pattern, so a newly created database is usable at once;
  4. writes the creator password into .env and a "creator" block into profiles.json.

Safe to run again (it resets the creator's password).
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
    HOSTS, SetupError, _ACCOUNT, _HOST, _connect, generate_password,
)

DEFAULT_CREATOR = "hai_creator"
_PREFIX = re.compile(r"^[a-z][a-z0-9]{0,15}_$")


def plan_statements(prefix: str, creator: str, password: str, reader: str, writer: str,
                    hosts: tuple[str, ...] = HOSTS) -> list[tuple[str, tuple]]:
    if not _PREFIX.match(prefix):
        raise SetupError("the prefix must be lowercase letters/digits ending with an underscore, e.g. hai_")
    for account in (creator, reader, *([writer] if writer else [])):
        if not _ACCOUNT.match(account):
            raise SetupError("account names may only contain letters, digits and underscores (max 32)")
    if not hosts or any(not _HOST.match(h) for h in hosts):
        raise SetupError("each host may only contain letters, digits and . : % _ -")
    # e.g. `hai\_%`.*  The underscore is escaped, and the % is doubled because the driver treats a
    # single % in a statement that has parameters as a placeholder.
    pattern = "`" + prefix[:-1] + "\\_%%`.*"
    statements: list[tuple[str, tuple]] = []
    for host in hosts:
        statements += [
            ("CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s", (creator, host, password)),
            ("ALTER USER %s@%s IDENTIFIED BY %s", (creator, host, password)),
            ("REVOKE ALL PRIVILEGES, GRANT OPTION FROM %s@%s", (creator, host)),
            (f"GRANT CREATE ON {pattern} TO %s@%s", (creator, host)),
            (f"GRANT SELECT ON {pattern} TO %s@%s", (reader, host)),
        ]
        if writer:
            statements.append((f"GRANT SELECT, INSERT, UPDATE, DELETE ON {pattern} TO %s@%s", (writer, host)))
    statements.append(("FLUSH PRIVILEGES", ()))
    return statements


def run(*, profile_name: str, profiles: dict[str, Any], profiles_path: Path, env_path: Path,
        admin_user: str, admin_password: str, creator_user: str, prefix: str,
        confirm: Callable[[str], bool], connect: Callable | None = None,
        out: Callable[[str], None] = print, hosts: tuple[str, ...] = HOSTS) -> int:
    profile = profiles.get(profile_name)
    if profile is None:
        raise SetupError(f"no profile named '{profile_name}'. Profiles in the file: "
                         f"{', '.join(sorted(profiles)) or '(none)'}")
    if profile.type != "mysql":
        raise SetupError("this tool only supports MySQL profiles")
    if creator_user.lower() in {admin_user.lower(), profile.user.lower(), profile.writer_user.lower()}:
        raise SetupError("the creator must be its own account (not the administrator, reader or writer)")
    if profile.user.lower() == admin_user.lower():
        raise SetupError("run setup_mysql_readonly_user.py first: the profile still names the administrator")
    if not admin_password:
        raise SetupError("an administrator password is required")
    password_env = profile.creator_password_env or (
        "HAI_" + re.sub(r"[^A-Za-z0-9]", "_", profile_name).upper() + "_CREATOR_PASSWORD")
    out(f"Server    : {profile.host}:{profile.port} (encrypted: {profile.ssl})")
    out(f"Creator   : '{creator_user}' may only CREATE databases named {prefix}...  (no DROP, no ALTER)")
    out(f"Also grant: reader '{profile.user}' SELECT" + (
        f", writer '{profile.writer_user}' SELECT/INSERT/UPDATE/DELETE" if profile.writer_user else "")
        + f" on {prefix}% databases")
    out(f"Will write: the creator password into {env_path.name} ({password_env}) and a creator block "
        f"into {profiles_path.name}")
    if not confirm("Continue?"):
        out("Nothing was changed.")
        return 1
    password = generate_password()
    statements = plan_statements(prefix, creator_user, password, profile.user, profile.writer_user, hosts)
    try:
        connection = _connect(profile, admin_user, admin_password, connect)
    except ImportError:
        raise SetupError('the MySQL driver is not installed. Run: pip install "PyMySQL>=1.1,<2"')
    except Exception as exc:
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
        if code == 1045:
            raise SetupError("MySQL rejected the administrator login: wrong administrator user or password") from exc
        raise SetupError(f"could not connect to MySQL: {type(exc).__name__}: {exc}") from exc
    try:
        cursor = connection.cursor()
        for sql, args in statements:
            cursor.execute(sql, args)
        cursor.close()
    finally:
        try:
            connection.close()
        except Exception:
            pass
    out("[ok]   creator created or reset; reader/writer can use databases named " + prefix + "...")
    set_env_value(env_path, password_env, password)
    data = json.loads(profiles_path.read_text(encoding="utf-8-sig"))
    data[profile_name]["creator"] = {"user": creator_user, "password_env": password_env, "prefix": prefix}
    profiles_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out(f"[ok]   wrote {password_env} to {env_path.name} and the creator block to {profiles_path.name}")
    os.environ[password_env] = password
    try:
        MySQLEngine(load_profiles(profiles_path), connect=connect).open_creator(profile_name)
    except Exception as exc:
        raise SetupError(f"the account was created but the final check failed: {exc}") from exc
    del password
    out("[ok]   the server accepts this creator account (CREATE only, prefix-limited)")
    out("\nDone. Restart the backend and give your user the role data-schema-creator.")
    return 0


def main(argv: list[str] | None = None, *, connect: Callable | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile")
    parser.add_argument("--admin-user", default="root")
    parser.add_argument("--creator-user", default=DEFAULT_CREATOR)
    parser.add_argument("--prefix", default="hai_", help="databases the agent may create start with this")
    parser.add_argument("--profiles", default="")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--hosts", default=",".join(HOSTS))
    parser.add_argument("--yes", action="store_true")
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
                   creator_user=args.creator_user, prefix=args.prefix, confirm=confirm, connect=connect,
                   hosts=tuple(h.strip() for h in args.hosts.split(",") if h.strip()))
    except (SetupError, EngineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
