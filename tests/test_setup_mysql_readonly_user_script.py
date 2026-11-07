from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

from api.database_engines import parse_profiles

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_mysql_readonly_user.py"
spec = importlib.util.spec_from_file_location("setup_mysql_readonly_user", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

PROFILE = {"type": "mysql", "host": "127.0.0.1", "port": 3306, "database": "dbsales",
           "user": "hai_reader", "password_env": "HAI_MYSQL_LOCAL_PASSWORD", "ssl": "require"}
ADMIN_PW = "AdminPw-ShouldNeverBeWritten9"


class FakeServer:
    """Records what an administrator session would run; no network involved."""

    def __init__(self, databases=("dbsales",), login_ok=True):
        self.databases, self.login_ok = set(databases), login_ok
        self.executed: list[tuple[str, tuple]] = []
        self.closed = False
        self.connect_kwargs: dict = {}

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs
        if not self.login_ok:
            raise Exception(1045, "Access denied for user 'root'@'localhost' (using password: YES)")
        server = self

        class Cursor:
            _row = None

            def execute(self, sql, args=()):
                server.executed.append((sql, tuple(args)))
                if "information_schema.SCHEMATA" in sql:
                    self._row = (args[0],) if args[0] in server.databases else None

            def fetchone(self):
                return self._row

            def close(self):
                pass

        class Connection:
            def cursor(self):
                return Cursor()

            def close(self):
                server.closed = True
        return Connection()


@pytest.fixture
def world(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"local_mysql": PROFILE, "other": {**PROFILE, "database": "dbother"}},
                                   indent=2), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_bytes(b"HAI_AUTH_ENABLED=true\r\nHAI_MYSQL_LOCAL_PASSWORD=\r\nHAI_LLM_PROVIDER=auto\r\n")
    monkeypatch.setattr(tool, "check_profile", lambda name, loaded: (True, [f"[ok]   verified {name}"]))
    saved = dict(os.environ)
    yield tmp_path, profiles, env
    os.environ.clear()
    os.environ.update(saved)


def _run(world, server=None, *, answer=True, **kw):
    tmp, profiles, env = world
    server = server or FakeServer()
    lines: list[str] = []
    args = dict(profile_name="local_mysql", profiles=tool.load_profiles(profiles), profiles_path=profiles,
                env_path=env, admin_user="root", admin_password=ADMIN_PW, reader_user=None,
                confirm=lambda _q: answer, connect=server.connect, out=lines.append)
    args.update(kw)
    return tool.run(**args), server, lines


# ------------------------------------------------------------------ pure pieces
def test_the_generated_password_is_strong_and_safe_for_env_and_sql():
    seen = {tool.generate_password() for _ in range(50)}
    assert len(seen) == 50
    for pw in seen:
        assert len(pw) == 24 and pw.isalnum()
        assert any(c.islower() for c in pw) and any(c.isupper() for c in pw) and any(c.isdigit() for c in pw)


def test_statements_use_parameters_and_never_inline_the_password():
    statements = tool.plan_statements("dbsales", "hai_reader", "PW123")
    assert all("PW123" not in sql for sql, _ in statements)
    sql = [s for s, _ in statements]
    assert sql.count("FLUSH PRIVILEGES") == 1 and sql[-1] == "FLUSH PRIVILEGES"
    for host in ("localhost", "127.0.0.1"):
        assert ("CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s", ("hai_reader", host, "PW123")) in statements
        assert ("REVOKE ALL PRIVILEGES, GRANT OPTION FROM %s@%s", ("hai_reader", host)) in statements
    grants = [s for s in sql if s.startswith("GRANT")]
    assert grants == ["GRANT SELECT ON `dbsales`.* TO %s@%s"] * 2     # SELECT only, one database


@pytest.mark.parametrize("database", ["db`; DROP TABLE x;--", "a b", "", "x" * 65, "db.name", "db*", "d'b"])
def test_a_hostile_database_name_is_refused_before_any_sql(database):
    with pytest.raises(tool.SetupError):
        tool.plan_statements(database, "hai_reader", "pw")


@pytest.mark.parametrize("reader", ["a'b", "a b", "", "x" * 33, "a;b", "ro@ot"])
def test_a_hostile_account_name_is_refused(reader):
    with pytest.raises(tool.SetupError):
        tool.plan_statements("dbsales", reader, "pw")


# ------------------------------------------------------------------ the whole run
def test_a_confirmed_run_creates_the_account_writes_the_password_and_fixes_the_profile(world):
    tmp, profiles, env = world
    code, server, lines = _run(world)
    assert code == 0
    sqls = [s for s, _ in server.executed]
    assert any("information_schema.SCHEMATA" in s for s in sqls)
    assert sum(s.startswith("GRANT SELECT ON `dbsales`.*") for s in sqls) == 2
    assert server.closed
    text = env.read_bytes().decode()
    match = re.search(r"HAI_MYSQL_LOCAL_PASSWORD=(\S+)\r\n", text)
    assert match and len(match.group(1)) == 24
    assert text.startswith("HAI_AUTH_ENABLED=true\r\n") and text.endswith("HAI_LLM_PROVIDER=auto\r\n")
    data = json.loads(profiles.read_text(encoding="utf-8"))
    assert data["local_mysql"]["user"] == "hai_reader" and data["other"]["database"] == "dbother"
    assert "Restart the backend" in "\n".join(lines)
    # the password written to .env is the one the account was given
    created = [args for s, args in server.executed if s.startswith("CREATE USER")]
    assert all(args[2] == match.group(1) for args in created)


def test_the_administrator_password_is_never_written_or_printed(world):
    tmp, profiles, env = world
    code, server, lines = _run(world)
    everything = "\n".join(lines) + env.read_text() + profiles.read_text()
    for path in tmp.rglob("*"):
        if path.is_file():
            everything += path.read_bytes().decode("utf-8", "ignore")
    assert ADMIN_PW not in everything
    assert all(ADMIN_PW not in str(args) for _, args in server.executed)   # it only authenticates the connection


def test_the_new_password_is_never_printed(world):
    tmp, profiles, env = world
    _, _, lines = _run(world)
    password = re.search(r"HAI_MYSQL_LOCAL_PASSWORD=(\S+)", env.read_text()).group(1)
    assert password not in "\n".join(lines)


def test_declining_the_confirmation_changes_nothing_and_never_connects(world):
    tmp, profiles, env = world
    before = (env.read_bytes(), profiles.read_bytes())
    code, server, lines = _run(world, answer=False)
    assert code == 1 and server.executed == [] and server.connect_kwargs == {}
    assert (env.read_bytes(), profiles.read_bytes()) == before


def test_a_missing_database_stops_before_creating_anything(world):
    tmp, profiles, env = world
    before = (env.read_bytes(), profiles.read_bytes())
    with pytest.raises(tool.SetupError, match="does not exist"):
        _run(world, FakeServer(databases=("another",)))
    assert (env.read_bytes(), profiles.read_bytes()) == before


def test_no_statement_after_the_database_check_runs_when_the_database_is_missing(world):
    server = FakeServer(databases=())
    with pytest.raises(tool.SetupError):
        _run(world, server)
    assert [s for s, _ in server.executed if s.startswith(("CREATE", "GRANT", "ALTER", "REVOKE"))] == []
    assert server.closed


def test_a_wrong_administrator_password_is_reported_plainly(world):
    tmp, profiles, env = world
    before = env.read_bytes()
    with pytest.raises(tool.SetupError) as info:
        _run(world, FakeServer(login_ok=False))
    assert "administrator" in str(info.value) and ADMIN_PW not in str(info.value)
    assert env.read_bytes() == before


def test_the_administrator_cannot_also_be_the_read_only_account(world):
    with pytest.raises(tool.SetupError, match="must not be the administrator"):
        _run(world, reader_user="root")
    with pytest.raises(tool.SetupError, match="must not be the administrator"):
        _run(world, admin_user="hai_reader")


def test_unknown_non_mysql_and_empty_inputs_are_refused(world):
    with pytest.raises(tool.SetupError, match="no profile named"):
        _run(world, profile_name="nope")
    with pytest.raises(tool.SetupError, match="administrator password is required"):
        _run(world, admin_password="")
    tmp, profiles, env = world
    pg = parse_profiles({"pg": {"type": "postgresql", "host": "h", "database": "d", "user": "u",
                                "password_env": "P"}})
    with pytest.raises(tool.SetupError, match="only supports MySQL"):
        _run(world, profile_name="pg", profiles=pg)


def test_a_failing_final_check_is_reported_after_the_account_exists(world, monkeypatch):
    monkeypatch.setattr(tool, "check_profile", lambda n, p: (False, ["[FAIL] could not connect"]))
    with pytest.raises(tool.SetupError, match="final check failed"):
        _run(world)


def test_the_tls_settings_follow_the_profile(world):
    _, server, _ = _run(world)
    assert server.connect_kwargs["ssl"] == {"check_hostname": False}
    assert server.connect_kwargs["ssl_verify_cert"] is False
    assert server.connect_kwargs["user"] == "root" and server.connect_kwargs["port"] == 3306


def test_running_twice_resets_the_password_and_keeps_the_files_consistent(world):
    tmp, profiles, env = world
    _run(world)
    first = re.search(r"HAI_MYSQL_LOCAL_PASSWORD=(\S+)", env.read_text()).group(1)
    server = FakeServer()
    _run(world, server)
    second = re.search(r"HAI_MYSQL_LOCAL_PASSWORD=(\S+)", env.read_text()).group(1)
    assert first != second and env.read_text().count("HAI_MYSQL_LOCAL_PASSWORD=") == 1
    assert [a[2] for s, a in server.executed if s.startswith("ALTER USER")] == [second, second]


# ------------------------------------------------------------------------- main()
def test_main_reads_the_admin_password_from_the_environment_and_supports_yes(world, monkeypatch, capsys):
    tmp, profiles, env = world
    monkeypatch.setenv("HAI_MYSQL_ADMIN_PASSWORD", ADMIN_PW)
    monkeypatch.delenv("HAI_DATABASE_PROFILES", raising=False)
    server = FakeServer()
    code = tool.main(["local_mysql", "--profiles", str(profiles), "--env-file", str(env), "--yes"],
                     connect=server.connect)
    out = capsys.readouterr().out
    assert code == 0 and "Done." in out and ADMIN_PW not in out


def test_main_reports_problems_as_a_short_message_not_a_traceback(world, monkeypatch, capsys):
    tmp, profiles, env = world
    monkeypatch.setenv("HAI_MYSQL_ADMIN_PASSWORD", ADMIN_PW)
    code = tool.main(["nope", "--profiles", str(profiles), "--env-file", str(env), "--yes"],
                     connect=FakeServer().connect)
    err = capsys.readouterr().err
    assert code == 1 and err.startswith("error: no profile named") and "Traceback" not in err
    monkeypatch.delenv("HAI_DATABASE_PROFILES", raising=False)
    assert tool.main(["x", "--env-file", str(tmp / "absent.env")]) == 1
    assert "no profiles file" in capsys.readouterr().err


# ------------------------------------------------------------------------- hosts
def test_the_hosts_the_account_may_use_can_be_chosen():
    statements = tool.plan_statements("dbsales", "hai_reader", "pw", ("localhost", "::1", "10.0.0.5"))
    created = [args[1] for sql, args in statements if sql.startswith("CREATE USER")]
    assert created == ["localhost", "::1", "10.0.0.5"]


@pytest.mark.parametrize("hosts", [(), ("",), ("a b",), ("x'; DROP",), ("a" * 61,), ("ok", "bad host")])
def test_hostile_or_empty_hosts_are_refused(hosts):
    with pytest.raises(tool.SetupError):
        tool.plan_statements("dbsales", "hai_reader", "pw", hosts)


def test_the_hosts_option_reaches_the_statements(world, monkeypatch):
    tmp, profiles, env = world
    monkeypatch.setenv("HAI_MYSQL_ADMIN_PASSWORD", ADMIN_PW)
    monkeypatch.delenv("HAI_DATABASE_PROFILES", raising=False)
    server = FakeServer()
    code = tool.main(["local_mysql", "--profiles", str(profiles), "--env-file", str(env), "--yes",
                      "--hosts", "localhost, 172.17.0.1"], connect=server.connect)
    assert code == 0
    created = [args[1] for sql, args in server.executed if sql.startswith("CREATE USER")]
    assert created == ["localhost", "172.17.0.1"]


# ------------------------------------------------------------------ real MySQL (live)
from tests.test_mysql_engine import HOST, PORT, ROOT_PW, live, my  # noqa: E402,F401


def _live_profile(tmp_path, database, user, monkeypatch):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"shop": {
        "type": "mysql", "host": HOST, "port": int(PORT), "database": database, "user": user,
        "password_env": "HAI_TEST_SETUP_PW", "ssl": "require"}}), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_bytes(b"KEEP=me\r\nHAI_TEST_SETUP_PW=\r\n")
    saved = dict(os.environ)
    return profiles, env, saved


def _live_run(profiles, env, **kw):
    lines: list[str] = []
    args = dict(profile_name="shop", profiles=tool.load_profiles(profiles), profiles_path=profiles,
                env_path=env, admin_user="root", admin_password=ROOT_PW, reader_user=None,
                confirm=lambda _q: True, out=lines.append,
                hosts=("%",))     # a Docker server sees the client as the bridge gateway, not localhost
    args.update(kw)
    return tool.run(**args), lines


@live
def test_a_real_server_gets_a_select_only_account_that_the_app_accepts(my, tmp_path, monkeypatch):  # noqa: F811
    import pymysql
    user = "setup_" + my.db[-8:]
    profiles, env, saved = _live_profile(tmp_path, my.db, user, monkeypatch)
    try:
        code, lines = _live_run(profiles, env)
        text = "\n".join(lines)
        assert code == 0, text
        assert "Result" not in text and "[ok]   connected; the session is read-only" in text
        password = re.search(r"HAI_TEST_SETUP_PW=(\S+)", env.read_text()).group(1)
        assert len(password) == 24 and "KEEP=me" in env.read_text() and password not in text

        my.admin.execute("SHOW GRANTS FOR %s@%s", (user, "%"))
        grants = [row[0] for row in my.admin.fetchall()]
        assert any(f"GRANT SELECT ON `{my.db}`.*" in g for g in grants), grants
        assert not any("ALL PRIVILEGES" in g or "INSERT" in g or "GRANT OPTION" in g for g in grants), grants

        conn = pymysql.connect(host=HOST, port=int(PORT), user=user, password=password,
                               ssl={"check_hostname": False}, ssl_verify_cert=False)
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {my.db}.sales")
        assert cur.fetchone()[0] == 30
        with pytest.raises(Exception):
            cur.execute("SELECT COUNT(*) FROM mysql.user")             # no access to anything else
        cur.execute("SHOW DATABASES")
        assert {r[0] for r in cur.fetchall()} <= {my.db, "information_schema", "performance_schema"}
        with pytest.raises(Exception):
            cur.execute(f"CREATE TABLE {my.db}.nope (a INT)")
        conn.close()
    finally:
        os.environ.clear(); os.environ.update(saved)
        my.admin.execute("DROP USER IF EXISTS %s@%s", (user, "%"))


@live
def test_an_account_that_already_has_too_many_rights_is_cut_back_to_select(my, tmp_path, monkeypatch):  # noqa: F811
    user = "fat_" + my.db[-8:]
    profiles, env, saved = _live_profile(tmp_path, my.db, user, monkeypatch)
    try:
        my.admin.execute("CREATE USER %s@%s IDENTIFIED BY 'OldPassword123'", (user, "%"))
        my.admin.execute(f"GRANT ALL PRIVILEGES ON *.* TO %s@%s WITH GRANT OPTION", (user, "%"))
        my.admin.execute("SHOW GRANTS FOR %s@%s", (user, "%"))
        # the precondition: MySQL 8.4 lists the rights one by one instead of saying ALL PRIVILEGES
        assert any("INSERT" in r[0] and "DROP" in r[0] for r in my.admin.fetchall())

        code, lines = _live_run(profiles, env)
        assert code == 0, "\n".join(lines)
        my.admin.execute("SHOW GRANTS FOR %s@%s", (user, "%"))
        grants = [r[0] for r in my.admin.fetchall()]
        assert not any("ALL PRIVILEGES" in g or "GRANT OPTION" in g or "INSERT" in g for g in grants), grants
        assert any(f"GRANT SELECT ON `{my.db}`.*" in g for g in grants), grants
    finally:
        os.environ.clear(); os.environ.update(saved)
        my.admin.execute("DROP USER IF EXISTS %s@%s", (user, "%"))


@live
def test_running_it_twice_keeps_working_with_a_fresh_password(my, tmp_path, monkeypatch):  # noqa: F811
    user = "twice_" + my.db[-8:]
    profiles, env, saved = _live_profile(tmp_path, my.db, user, monkeypatch)
    try:
        assert _live_run(profiles, env)[0] == 0
        first = re.search(r"HAI_TEST_SETUP_PW=(\S+)", env.read_text()).group(1)
        assert _live_run(profiles, env)[0] == 0
        second = re.search(r"HAI_TEST_SETUP_PW=(\S+)", env.read_text()).group(1)
        assert first != second
    finally:
        os.environ.clear(); os.environ.update(saved)
        my.admin.execute("DROP USER IF EXISTS %s@%s", (user, "%"))


@live
def test_real_failures_are_reported_plainly_and_change_nothing(my, tmp_path, monkeypatch):  # noqa: F811
    profiles, env, saved = _live_profile(tmp_path, my.db, "nobody_x", monkeypatch)
    before = env.read_bytes()
    try:
        with pytest.raises(tool.SetupError, match="administrator"):
            _live_run(profiles, env, admin_password="definitely-wrong")
        data = json.loads(profiles.read_text())
        data["shop"]["database"] = "no_such_database"
        profiles.write_text(json.dumps(data))
        with pytest.raises(tool.SetupError, match="does not exist"):
            _live_run(profiles, env)
        assert env.read_bytes() == before
        my.admin.execute("SELECT COUNT(*) FROM mysql.user WHERE user = 'nobody_x'")
        assert my.admin.fetchone()[0] == 0                       # nothing was created
    finally:
        os.environ.clear(); os.environ.update(saved)


def test_a_profile_that_names_the_administrator_gets_a_dedicated_read_only_account(world):
    """The real situation: the user pointed the profile at root, which the app refuses."""
    tmp, profiles, env = world
    data = json.loads(profiles.read_text())
    data["local_mysql"]["user"] = "root"
    profiles.write_text(json.dumps(data), encoding="utf-8")
    code, server, lines = _run(world)
    assert code == 0
    assert "root" in "\n".join(lines) and "hai_reader" in "\n".join(lines)
    created = {args[0] for sql, args in server.executed if sql.startswith("CREATE USER")}
    assert created == {"hai_reader"}                      # never root, never the admin
    assert json.loads(profiles.read_text())["local_mysql"]["user"] == "hai_reader"


def test_an_explicit_reader_name_that_equals_the_administrator_is_still_refused(world):
    with pytest.raises(tool.SetupError, match="must not be the administrator"):
        _run(world, reader_user="root")
