"""SQL Server engine tests.

Offline tests always run. Live tests need a disposable SQL Server you administer with your Windows login
(for example SQL Server Express on this computer). They create a throw-away database and logins and drop them:

    pip install pyodbc
    $env:HAI_SQLSERVER_TEST_SERVER = '(local)\\SQLEXPRESS'
    pytest tests/test_sqlserver_engine.py
"""
from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace

import pytest

from agent.llm.contracts import LLMResponse
from api.database_engines import (
    EngineError, SQLServerEngine, SQLServerProfile, check_sqlserver_creator_permissions,
    check_sqlserver_permissions, map_sqlserver_type, parse_profiles,
)
from api.database_profile_check import check_profile
from api.database_runtime_registry import DatabaseRuntimeRegistry, RegistryError
from api.security.authorization import AuthorizationPolicy
from tests.engine_conformance import ConformanceSuite, Harness

SERVER = os.getenv("HAI_SQLSERVER_TEST_SERVER")
live = pytest.mark.skipif(not SERVER, reason="set HAI_SQLSERVER_TEST_SERVER for live SQL Server tests")
GOOD = {"type": "sqlserver", "server": r"(local)\SQLEXPRESS", "database": "dbsales", "user": "hai_sql_reader",
        "password_env": "P"}


# ------------------------------------------------------------------ offline
def test_a_sqlserver_profile_has_safe_defaults():
    p = parse_profiles({"local_sqlserver": GOOD})["local_sqlserver"]
    assert isinstance(p, SQLServerProfile)
    assert (p.host, p.port, p.schema, p.ssl) == (r"(local)\SQLEXPRESS", 0, "dbo", "require")


def test_host_is_accepted_as_an_alias_of_server_and_port_is_optional():
    p = parse_profiles({"remote_mssql": {**{k: v for k, v in GOOD.items() if k != "server"},
                                         "host": "db.example.com", "port": 1433}})["remote_mssql"]
    assert (p.host, p.port) == ("db.example.com", 1433)


@pytest.mark.parametrize("change", [
    {"ssl": "disable"}, {"password": "p"}, {"port": 70000}, {"port": -1}, {"statement_timeout_ms": 5},
    {"server": "db;Trusted_Connection=yes"}, {"server": "db}{x"}, {"database": "d;x"}, {"database": "my db"},
    {"user": "sa'--"}, {"schema": "dbo;x"}, {"driver": "X};SERVER={evil"}, {"sslmode": "require"},
    {"host": "h"},
])
def test_invalid_sqlserver_profiles_are_rejected(change):
    with pytest.raises(EngineError):
        parse_profiles({"crm": {**GOOD, **change}})


def test_profiles_of_all_engines_can_share_a_file():
    my = {"type": "mysql", "host": "h", "database": "d", "user": "u", "password_env": "P"}
    got = parse_profiles({"a_my": my, "b_ms": GOOD})
    assert (got["a_my"].type, got["b_ms"].type) == ("mysql", "sqlserver")


def test_sqlserver_type_mapping():
    assert [map_sqlserver_type(t) for t in (
        "int", "bigint", "tinyint", "decimal", "money", "float", "varchar", "nvarchar", "bit", "date",
        "datetime", "datetime2", "uniqueidentifier")] == [
        "integer", "integer", "integer", "number", "number", "number", "string", "string", "boolean",
        "date", "datetime", "datetime", "string"]


@pytest.mark.parametrize("db,server,writable", [
    (["CONNECT", "SELECT"], ["CONNECT SQL", "VIEW ANY DATABASE"], []),
    (["CONNECT", "SELECT", "VIEW DEFINITION"], ["CONNECT SQL"], []),
])
def test_a_read_only_account_is_accepted(db, server, writable):
    check_sqlserver_permissions(db, server, writable)


@pytest.mark.parametrize("db,server,writable,needle", [
    (["CONNECT", "SELECT", "INSERT"], ["CONNECT SQL"], [], "INSERT"),
    (["CONNECT", "SELECT", "CONTROL", "ALTER"], ["CONNECT SQL"], [], "ALTER|CONTROL"),
    (["CONNECT", "SELECT"], ["CONNECT SQL", "CONTROL SERVER"], [], "server-wide"),
    (["CONNECT", "SELECT"], ["CONNECT SQL"], ["sales", "orders"], "sales, orders"),
    (["EXECUTE"], [], [], "EXECUTE"),
])
def test_an_account_that_can_do_more_than_read_is_refused(db, server, writable, needle):
    with pytest.raises(EngineError, match=needle):
        check_sqlserver_permissions(db, server, writable)


def test_the_connection_string_cannot_be_extended_through_the_password(monkeypatch):
    engine = SQLServerEngine(parse_profiles({"x1": GOOD}))
    monkeypatch.setenv("P", "pw};Trusted_Connection=yes;PWD={x")
    text = engine._connection_string(engine.profiles["x1"], "ODBC Driver 17 for SQL Server")
    assert "PWD={pw}};Trusted_Connection=yes;PWD={x}" in text and text.count("Trusted_Connection") == 1
    assert "SERVER=(local)\\SQLEXPRESS;" in text and "Encrypt=yes" in text and "TrustServerCertificate=yes" in text
    with_port = SQLServerEngine(parse_profiles({"x2": {**GOOD, "server": "db.example.com", "port": 1433, "ssl": "verify"}}))
    text = with_port._connection_string(with_port.profiles["x2"], "D")
    assert "SERVER=db.example.com,1433;" in text and "TrustServerCertificate=no" in text


def test_the_engine_reads_with_top_brackets_and_question_marks():
    assert (SQLServerEngine.dialect, SQLServerEngine.quote, SQLServerEngine.paramstyle) == ("top", "[", "qmark")


def test_registry_lists_the_profile_without_the_server_and_knows_the_engine(tmp_path):
    (tmp_path / "d").mkdir()
    reg = DatabaseRuntimeRegistry(
        data_root=tmp_path / "d", work_dir=tmp_path / "w", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: None, profiles=parse_profiles({"local_sqlserver": GOOD}))
    assert reg.profiles() == [{"name": "local_sqlserver", "type": "sqlserver"}]     # no creator: no prefix offered
    assert reg.profile_engine("local_sqlserver") == "sqlserver"
    assert reg.available_engines() == ["sqlite", "sqlserver"]
    assert "sqlserver" in reg.supported_engine_names() and "mssql" in reg.supported_engine_names()
    assert "SQLEXPRESS" not in str(reg.profiles())


def test_a_missing_password_variable_is_reported_without_connecting(tmp_path, monkeypatch):
    monkeypatch.delenv("P", raising=False)
    ok, lines = check_profile("x1", parse_profiles({"x1": GOOD}))
    assert not ok and any("environment variable P is not set" in line for line in lines)
    assert lines[0].startswith("[ok]   profile 'x1': sqlserver at (local)\\SQLEXPRESS, database")


# --------------------------------------------------------------------- live
class _Provider:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


@pytest.fixture
def ms(monkeypatch, tmp_path):
    import pyodbc
    tag = uuid.uuid4().hex[:8]
    db, reader, writer, owner = f"hai_rep_{tag}", f"rd_{tag}", f"wr_{tag}", f"ow_{tag}"
    creator, new_db = f"cr_{tag}", f"hai_new_{tag}"
    base = ("DRIVER={ODBC Driver 17 for SQL Server};SERVER=" + SERVER +
            ";TrustServerCertificate=yes;Encrypt=yes;")
    admin = pyodbc.connect(base + "DATABASE=master;Trusted_Connection=yes", autocommit=True)
    cur = admin.cursor()
    password = "Rolepw" + uuid.uuid4().hex[:12]
    try:
        cur.execute(f"CREATE DATABASE [{db}]")
        for user in (reader, writer, owner, creator):
            cur.execute(f"CREATE LOGIN [{user}] WITH PASSWORD = N'{password}', CHECK_POLICY = OFF")
        cur.execute(f"CREATE USER [{creator}] FOR LOGIN [{creator}]")      # in master: the permission lives there
        cur.execute(f"GRANT CREATE ANY DATABASE TO [{creator}]")
        cur.execute(f"USE [{db}]")
        for user in (reader, writer, owner):
            cur.execute(f"CREATE USER [{user}] FOR LOGIN [{user}]")
        cur.execute(f"ALTER ROLE db_datareader ADD MEMBER [{reader}]")
        cur.execute(f"ALTER ROLE db_datareader ADD MEMBER [{writer}]")
        cur.execute(f"ALTER ROLE db_datawriter ADD MEMBER [{writer}]")
        cur.execute(f"ALTER ROLE db_owner ADD MEMBER [{owner}]")
        cur.execute("CREATE TABLE dbo.sales (id INT PRIMARY KEY, region VARCHAR(20), product VARCHAR(20), "
                    "amount DECIMAL(10,2), sold_on DATE)")
        cur.execute("ALTER TABLE dbo.sales ADD [weird name] VARCHAR(5)")
        cur.execute("CREATE TABLE dbo.[order] (id INT PRIMARY KEY, [group] VARCHAR(10), [user] VARCHAR(10), "
                    "amount DECIMAL(8,2))")
        for i in range(1, 9):
            cur.execute("INSERT INTO dbo.[order] VALUES (?,?,?,?)", i, ["a", "b"][i % 2], "u1", f"{i}.50")
        for i in range(1, 31):
            cur.execute("INSERT INTO dbo.sales (id, region, product, amount, sold_on) VALUES (?,?,?,?,?)",
                        i, ["west", "east", "north"][i % 3], "widget" if i % 2 else "gadget",
                        f"{10 + i}.25", f"2026-01-{(i % 28) + 1:02d}")
        monkeypatch.setenv("HAI_TEST_MS_ROLE_PW", password)
        monkeypatch.setenv("HAI_TEST_MS_WRONG_PW", "not-the-password")

        def profile(user, pw_env="HAI_TEST_MS_ROLE_PW"):
            return {"type": "sqlserver", "server": SERVER, "database": db, "user": user, "password_env": pw_env}

        profiles = parse_profiles({
            "ms_reader": profile(reader), "ms_writer": profile(writer), "ms_owner": profile(owner),
            "ms_wrongpw": profile(reader, "HAI_TEST_MS_WRONG_PW"),
            "ms_creator": {**profile(reader), "creator": {"user": creator, "password_env": "HAI_TEST_MS_ROLE_PW",
                                                          "prefix": "hai_"}}})
        made = []

        def registry(*answers):
            r = DatabaseRuntimeRegistry(
                data_root=tmp_path, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
                llm_provider_factory=lambda: _Provider(*answers), profiles=profiles)
            made.append(r)
            return r

        yield SimpleNamespace(registry=registry, db=db, admin=cur, reader=reader, profiles=profiles,
                              password=password, new_db=new_db, creator=creator, work=tmp_path / "work")
    finally:
        for r in locals().get("made", []):
            r.shutdown()
        cur.execute("USE master")
        for user in (reader, writer, owner, creator):                      # pooled sessions keep a login "in use"
            cur.execute("SELECT session_id FROM sys.dm_exec_sessions WHERE login_name = ?", user)
            for (session,) in cur.fetchall():
                cur.execute(f"KILL {int(session)}")
        for name in (db, new_db):
            cur.execute(f"IF DB_ID(N'{name}') IS NOT NULL ALTER DATABASE [{name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE")
            cur.execute(f"IF DB_ID(N'{name}') IS NOT NULL DROP DATABASE [{name}]")
        cur.execute(f"IF USER_ID(N'{creator}') IS NOT NULL DROP USER [{creator}]")
        for user in (reader, writer, owner, creator):
            cur.execute(f"IF SUSER_ID(N'{user}') IS NOT NULL DROP LOGIN [{user}]")
        admin.close()


@live
def test_a_select_only_login_connects_and_hidden_names_are_reported(ms):
    out = ms.registry().connect("mssales", engine="sqlserver", source="ms_reader", actor_id="alice")
    tables = {t["name"]: t for t in out["schema"]["tables"]}
    assert set(tables) == {"sales", "order"}
    assert "weird name" not in {c["name"] for c in tables["sales"]["columns"]}
    cols = {c["name"]: c for c in tables["sales"]["columns"]}
    assert cols["id"]["primary_key"] is True and cols["amount"]["declared_type"] == "decimal"
    assert any("hidden" in w for w in out["warnings"])


@live
def test_the_owner_and_the_writer_login_are_refused_with_the_reason(ms):
    reg = ms.registry()
    with pytest.raises(RegistryError):
        reg.connect("a1", engine="sqlserver", source="ms_owner", actor_id="alice")
    with pytest.raises(RegistryError):
        reg.connect("a2", engine="sqlserver", source="ms_writer", actor_id="alice")
    assert reg.list() == []
    from agent.database import ConnectionManager
    engine = SQLServerEngine(ms.profiles)
    with pytest.raises(PermissionError, match="administrator or database owner"):
        engine.open(ConnectionManager(), "t1", "ms_owner", "diag")
    with pytest.raises(EngineError, match="can modify data|more than SELECT"):
        engine.open(ConnectionManager(), "t2", "ms_writer", "diag")


@live
def test_the_check_script_logic_reports_a_working_profile_and_a_wrong_password(ms):
    ok, lines = check_profile("ms_reader", ms.profiles)
    assert ok and any("2 table(s) visible" in line for line in lines)
    ok, lines = check_profile("ms_wrongpw", ms.profiles)
    assert not ok and any("could not connect" in line for line in lines)
    assert any("hint:" in line and "refused the login" in line for line in lines)
    assert not any(ms.password in line for line in lines)


@live
def test_the_reader_cannot_write_even_by_its_own_hand(ms):
    reg = ms.registry()
    reg.connect("mssales", engine="sqlserver", source="ms_reader", actor_id="alice")
    with reg._connections["mssales"].manager.connection("mssales") as c:
        cur = c.cursor()
        for sql in ("UPDATE dbo.sales SET amount = 0", "DELETE FROM dbo.sales", "CREATE TABLE dbo.x (a INT)",
                    "INSERT INTO dbo.sales (id) VALUES (999)"):
            with pytest.raises(Exception):
                cur.execute(sql)
    ms.admin.execute(f"SELECT COUNT(*) FROM [{ms.db}].dbo.sales")
    assert ms.admin.fetchone()[0] == 30


@live
def test_a_connection_survives_a_restart(ms):
    reg = ms.registry()
    reg.connect("mssales", engine="sqlserver", source="ms_reader", actor_id="alice")
    reg.activate("mssales", {"sales": {"columns": ["id", "region", "amount"], "key_column": "id"}},
                 actor_id="bob")
    reg.shutdown()
    (item,) = ms.registry().list()
    assert item["status"] == "active" and item["database"] == "sqlserver"


@live
class TestSQLServerConformance(ConformanceSuite):
    """The shared engine contract (see tests/engine_conformance.py)."""

    @pytest.fixture
    def harness(self, ms):
        def probe(connection):
            cur = connection.cursor()
            with pytest.raises(Exception):
                cur.execute("CREATE TABLE dbo.should_fail (a INT)")
            with pytest.raises(Exception):
                cur.execute("UPDATE dbo.sales SET amount = 0")

        def sql(query):
            ms.admin.execute(query)
            return [tuple(r) for r in ms.admin.fetchall()]
        return Harness(
            engine="sqlserver", registry=ms.registry, reader="ms_reader", writer="ms_writer",
            privileged="ms_owner", wrong_password="ms_wrongpw", sql=sql,
            q_sales=f"[{ms.db}].dbo.sales", q_order=f"[{ms.db}].dbo.[order]", order_name="order",
            quote=lambda column: f"[{column}]", assert_read_only=probe,
            leak_tokens=[SERVER or "", ms.reader, ms.db, ms.password])


# ------------------------------------------------------------ creating databases
@pytest.mark.parametrize("perms,roles,needle", [
    (["CONNECT SQL", "VIEW ANY DATABASE", "CREATE ANY DATABASE"], [], None),
    (["CONNECT SQL", "VIEW ANY DATABASE", "CREATE ANY DATABASE"], ["sysadmin"], "sysadmin"),
    (["CONNECT SQL", "CREATE ANY DATABASE"], ["dbcreator"], "dbcreator"),
    (["CONNECT SQL", "ALTER ANY DATABASE", "CREATE ANY DATABASE"], [], "ALTER ANY DATABASE"),
    (["CONNECT SQL", "VIEW ANY DATABASE"], [], "cannot create"),
    (["CONNECT SQL", "CREATE ANY DATABASE", "ALTER ANY LOGIN"], [], "ALTER ANY LOGIN"),
])
def test_the_creator_may_hold_create_any_database_and_nothing_else(perms, roles, needle):
    if needle is None:
        check_sqlserver_creator_permissions(perms, roles)
    else:
        with pytest.raises(EngineError, match=needle):
            check_sqlserver_creator_permissions(perms, roles)


def test_a_reader_in_a_server_role_is_refused():
    with pytest.raises(EngineError, match="securityadmin"):
        check_sqlserver_permissions(["CONNECT", "SELECT"], ["CONNECT SQL"], [], ["securityadmin"])


@pytest.mark.parametrize("creator,needle", [
    ({"user": "hai_sql_reader", "password_env": "C"}, "own login"),
    ({"user": "c1"}, "creator must be"),
    ({"user": "c1", "password_env": "C", "prefix": "Bad"}, "creator must be"),
    ({"user": "c1; DROP", "password_env": "C"}, "creator must be"),
    ({"user": "c1", "password_env": "C", "extra": 1}, "creator must be"),
])
def test_invalid_creator_blocks_are_rejected(creator, needle):
    with pytest.raises(EngineError, match=needle):
        parse_profiles({"crm": {**GOOD, "creator": creator}})


def test_a_creator_block_is_optional_and_a_database_can_be_chosen_only_with_the_prefix():
    plain = SQLServerEngine(parse_profiles({"crm": GOOD}))
    assert not plain.has_creator("crm")
    with pytest.raises(EngineError, match="only databases named"):
        plain._resolve("crm/hai_x")
    engine = SQLServerEngine(parse_profiles({"crm": {**GOOD, "creator": {"user": "c1", "password_env": "C"}}}))
    assert engine.has_creator("crm") and engine._resolve("crm/hai_shop").database == "hai_shop"
    for bad in ("crm/dbsales", "crm/hai_", "crm/HAI_X", "crm/hai_x;DROP"):
        with pytest.raises(EngineError):
            engine._resolve(bad)


@live
def test_a_server_with_a_creator_is_listed_with_its_prefix_and_type(ms):
    assert ms.registry().creatable_profiles() == [{"name": "ms_creator", "prefix": "hai_", "type": "sqlserver"}]


def _design(db, table="tb_pegawai"):
    return json.dumps({"supported": True, "database": db, "title": "hr", "tables": [{"name": table, "columns": [
        {"name": "nip", "type": "text", "nullable": False}, {"name": "gaji", "type": "decimal"},
        {"name": "tanggal_masuk", "type": "date"}, {"name": "catatan", "type": "long_text"}]}]})


@live
def test_a_database_and_tables_are_created_then_read_by_the_reader_and_more_tables_added(ms):
    reg = ms.registry(_design(ms.new_db), _design(ms.new_db, "tb_cuti"))
    service = reg.database_creator("ms_creator")
    first = service.designer.plan("buat database hr")
    mode, statements = service.preview(first.design)
    assert mode == "create_database" and statements[0] == f"CREATE DATABASE [{ms.new_db}]"
    assert any("db_datareader" in s for s in statements) and not any("db_owner" in s for s in statements)
    assert service.execute(first.design, actor="alice", plan_id="p1", question="q")["mode"] == "create_database"
    ms.admin.execute(f"SELECT name FROM [{ms.new_db}].sys.tables")
    assert [r[0] for r in ms.admin.fetchall()] == ["tb_pegawai"]
    ms.admin.execute(f"SELECT c.name FROM [{ms.new_db}].sys.identity_columns c "
                     f"WHERE c.object_id = OBJECT_ID('[{ms.new_db}].dbo.tb_pegawai')")
    assert [r[0] for r in ms.admin.fetchall()] == ["id"]                 # the automatic numeric key

    # the same tables again are refused, new tables are added without touching the existing one
    with pytest.raises(Exception, match="already has the table"):
        service.preview(first.design)
    more = service.designer.plan("tambah tabel cuti")
    mode, statements = service.preview(more.design)
    assert mode == "add_tables" and len(statements) == 1
    assert service.execute(more.design, actor="alice", plan_id="p2", question="q")["mode"] == "add_tables"
    ms.admin.execute(f"SELECT name FROM [{ms.new_db}].sys.tables ORDER BY 1")
    assert [r[0] for r in ms.admin.fetchall()] == ["tb_cuti", "tb_pegawai"]

    # the read-only login can read the new database through the same profile, and only read
    out = reg.connect("hrdb", engine="sqlserver", source=f"ms_creator/{ms.new_db}", actor_id="alice")
    assert {t["name"] for t in out["schema"]["tables"]} == {"tb_cuti", "tb_pegawai"}
    with reg._connections["hrdb"].manager.connection("hrdb") as c:
        with pytest.raises(Exception):
            c.cursor().execute("INSERT INTO dbo.tb_cuti (nip) VALUES ('x')")
    audit = (ms.work / "audit" / "creations-ms_creator.jsonl").read_text("utf-8")
    assert "create_database" in audit and "add_tables" in audit


@live
def test_the_creator_cannot_touch_a_database_it_did_not_create_or_drop_anything(ms):
    from agent.workflow.database_creation import CreationRefused
    reg = ms.registry(_design(ms.db))
    service = reg.database_creator("ms_creator")
    design = service.designer.plan("tambah ke database lama").design
    with pytest.raises(CreationRefused, match="cannot open the existing database"):
        service.preview(design)                                           # exists, but not the creator's
    with pytest.raises(CreationRefused):
        service.execute(design, actor="a", plan_id="p", question="q")
    ms.admin.execute(f"SELECT COUNT(*) FROM [{ms.db}].sys.tables")
    assert ms.admin.fetchone()[0] == 2                                    # untouched: sales and order
    connection = service._connect()
    try:
        for sql in (f"DROP DATABASE [{ms.db}]", f"CREATE TABLE [{ms.db}].dbo.x (a INT)", "CREATE LOGIN zz WITH PASSWORD = 'Aa1x234567890'"):
            with pytest.raises(Exception):
                connection.cursor().execute(sql)
    finally:
        connection.close()


@live
def test_a_name_outside_the_prefix_is_refused_before_anything_runs(ms):
    reg = ms.registry(json.dumps({"supported": True, "database": "dbsales_x", "tables": []}),
                      json.dumps({"supported": True, "database": "dbsales_x", "tables": []}))
    result = reg.database_creator("ms_creator").designer.plan("buat dbsales_x")
    assert result.design is None and "must start with" in result.reason


@live
def test_a_login_with_more_than_create_any_database_is_refused_as_creator(ms):
    ms.admin.execute(f"ALTER SERVER ROLE dbcreator ADD MEMBER [{ms.creator}]")
    try:
        with pytest.raises(RegistryError, match="dbcreator|cannot use the creator"):
            ms.registry().database_creator("ms_creator")
    finally:
        ms.admin.execute(f"ALTER SERVER ROLE dbcreator DROP MEMBER [{ms.creator}]")


@live
def test_the_reader_plan_of_connect_sqlserver_works_on_a_real_server_and_is_read_only(ms):
    """The statements the setup script runs, executed for real (a fake connection hid a missing role once)."""
    import importlib.util
    import sys as _sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "connect_sqlserver.py"
    spec = importlib.util.spec_from_file_location("connect_sqlserver_live", path)
    tool = importlib.util.module_from_spec(spec)
    _sys.modules["connect_sqlserver_live"] = tool
    spec.loader.exec_module(tool)
    login = f"sr_{uuid.uuid4().hex[:8]}"
    password = tool.generate_password()
    plan = tool.plan_statements(login, ms.db, password)
    try:
        for database, statements in plan.items():
            ms.admin.execute(f"USE [{database}]")
            for sql in statements:
                ms.admin.execute(sql)
        for database, statements in plan.items():          # running it again (a reset) must work as well
            ms.admin.execute(f"USE [{database}]")
            for sql in statements:
                ms.admin.execute(sql)
        ms.admin.execute("USE master")
        from agent.database import ConnectionManager
        import os as _os
        _os.environ["HAI_TEST_MS_SCRIPT_PW"] = password
        profile = parse_profiles({"sr": {"type": "sqlserver", "server": SERVER, "database": ms.db, "user": login,
                                         "password_env": "HAI_TEST_MS_SCRIPT_PW"}})
        schema, _ = SQLServerEngine(profile).open(ConnectionManager(), "sr", "sr", "diag")   # passes the safety check
        assert {t.name for t in schema.tables} == {"sales", "order"}
    finally:
        _os.environ.pop("HAI_TEST_MS_SCRIPT_PW", None)
        ms.admin.execute("USE master")
        ms.admin.execute("SELECT session_id FROM sys.dm_exec_sessions WHERE login_name = ?", login)
        for (session,) in ms.admin.fetchall():
            ms.admin.execute(f"KILL {int(session)}")
        ms.admin.execute(f"USE [{ms.db}]")
        ms.admin.execute(f"IF USER_ID(N'{login}') IS NOT NULL DROP USER [{login}]")
        ms.admin.execute("USE master")
        ms.admin.execute(f"IF SUSER_ID(N'{login}') IS NOT NULL DROP LOGIN [{login}]")


@live
def test_the_creator_plan_of_connect_sqlserver_works_on_a_real_server_and_passes_the_safety_check(ms):
    import importlib.util
    import sys as _sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "connect_sqlserver.py"
    spec = importlib.util.spec_from_file_location("connect_sqlserver_live2", path)
    tool = importlib.util.module_from_spec(spec)
    _sys.modules["connect_sqlserver_live2"] = tool
    spec.loader.exec_module(tool)
    login = f"sc_{uuid.uuid4().hex[:8]}"
    password = tool.generate_password()
    try:
        ms.admin.execute("USE master")
        for _ in range(2):                                   # twice: a reset after a half-finished run must work
            for sql in tool.plan_creator_statements(login, password):
                ms.admin.execute(sql)
        import os as _os
        _os.environ["HAI_TEST_MS_SCRIPT_PW"] = password
        profile = parse_profiles({"sc": {**GOOD, "server": SERVER, "database": ms.db, "creator": {
            "user": login, "password_env": "HAI_TEST_MS_SCRIPT_PW"}}})
        factory = SQLServerEngine(profile).open_creator("sc")          # runs the creator safety check
        connection = factory()
        connection.close()
    finally:
        _os.environ.pop("HAI_TEST_MS_SCRIPT_PW", None)
        ms.admin.execute("USE master")
        ms.admin.execute("SELECT session_id FROM sys.dm_exec_sessions WHERE login_name = ?", login)
        for (session,) in ms.admin.fetchall():
            ms.admin.execute(f"KILL {int(session)}")
        ms.admin.execute(f"IF USER_ID(N'{login}') IS NOT NULL DROP USER [{login}]")
        ms.admin.execute(f"IF SUSER_ID(N'{login}') IS NOT NULL DROP LOGIN [{login}]")


def test_a_profile_with_a_creator_tells_the_page_its_prefix_and_never_anything_else(tmp_path):
    (tmp_path / "d").mkdir()
    profiles = parse_profiles({
        "with_creator": {**GOOD, "creator": {"user": "c1", "password_env": "C", "prefix": "hai_"}},
        "plain": GOOD})
    reg = DatabaseRuntimeRegistry(
        data_root=tmp_path / "d", work_dir=tmp_path / "w", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: None, profiles=profiles)
    assert reg.profiles() == [{"name": "plain", "type": "sqlserver"},
                              {"name": "with_creator", "type": "sqlserver", "prefix": "hai_"}]
