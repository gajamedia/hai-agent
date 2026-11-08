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
    EngineError, SQLServerEngine, SQLServerProfile, check_sqlserver_permissions, map_sqlserver_type,
    parse_profiles,
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
    assert reg.profiles() == [{"name": "local_sqlserver", "type": "sqlserver"}]
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
    base = ("DRIVER={ODBC Driver 17 for SQL Server};SERVER=" + SERVER +
            ";TrustServerCertificate=yes;Encrypt=yes;")
    admin = pyodbc.connect(base + "DATABASE=master;Trusted_Connection=yes", autocommit=True)
    cur = admin.cursor()
    password = "Rolepw" + uuid.uuid4().hex[:12]
    try:
        cur.execute(f"CREATE DATABASE [{db}]")
        for user in (reader, writer, owner):
            cur.execute(f"CREATE LOGIN [{user}] WITH PASSWORD = N'{password}', CHECK_POLICY = OFF")
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
            "ms_wrongpw": profile(reader, "HAI_TEST_MS_WRONG_PW")})
        made = []

        def registry(*answers):
            r = DatabaseRuntimeRegistry(
                data_root=tmp_path, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
                llm_provider_factory=lambda: _Provider(*answers), profiles=profiles)
            made.append(r)
            return r

        yield SimpleNamespace(registry=registry, db=db, admin=cur, reader=reader, profiles=profiles,
                              password=password)
    finally:
        for r in locals().get("made", []):
            r.shutdown()
        cur.execute("USE master")
        cur.execute(f"IF DB_ID(N'{db}') IS NOT NULL ALTER DATABASE [{db}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE")
        cur.execute(f"IF DB_ID(N'{db}') IS NOT NULL DROP DATABASE [{db}]")
        for user in (reader, writer, owner):
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
