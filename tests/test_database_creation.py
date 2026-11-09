"""Creating databases: validated designs, safe DDL, prefix-limited accounts, fresh login, exact phrase.

Offline tests always run; live tests need the disposable MySQL described in tests/test_mysql_engine.py.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.llm.contracts import LLMResponse
from agent.llm.service import LLMService
from agent.workflow.database_creation import (
    COLUMN_TYPES, CreationRefused, DatabaseCreationService, DatabaseDesign, DatabaseDesigner, TableDef, ColumnDef,
    build_statements,
)
from api.database_engines import (
    EngineError, MySQLEngine, check_mysql_creator_grants, check_mysql_writer_grants, parse_profiles,
)
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from tests.test_data_changes_api import USERS, _AsUser

PORT = os.getenv("HAI_MYSQL_TEST_PORT")
HOST = os.getenv("HAI_MYSQL_TEST_HOST", "127.0.0.1")
ROOT_PW = os.getenv("HAI_MYSQL_TEST_ROOT_PASSWORD", "")
live = pytest.mark.skipif(not PORT, reason="set HAI_MYSQL_TEST_PORT for live MySQL tests")
CREATE = "database:create:manage"
BASE = {"type": "mysql", "host": "127.0.0.1", "database": "dbsales", "user": "hai_reader", "password_env": "R"}


class Scripted:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _designer(*answers, prefix="hai_"):
    return DatabaseDesigner(LLMService(Scripted(*answers)), prefix=prefix)


GOOD = {"supported": True, "database": "hai_toko", "title": "Toko", "tables": [
    {"name": "tb_bahan", "columns": [{"name": "nama", "type": "text", "nullable": False},
                                     {"name": "harga_beli", "type": "decimal"}]}]}


# ------------------------------------------------------------------ design validation
@pytest.mark.parametrize("change,needle", [
    ({"database": "dbsales"}, "must start with"),
    ({"database": "hai_"}, "must start with"),
    ({"database": "hai_x; DROP DATABASE y"}, "must start with"),
    ({"database": "HAI_TOKO"}, "must start with"),
    ({"database": None}, "must start with"),
    ({"tables": "x"}, "list"),
    ({"tables": [{"name": "t", "columns": []}]}, "no columns"),
    ({"tables": [{"name": "t", "columns": [{"name": f"c{i}", "type": "long_text"} for i in range(501)]}]}, "limit is 500"),
    ({"tables": [{"name": "t", "columns": [{"name": f"c{i}", "type": "text"} for i in range(70)]}]}, "too wide"),
    ({"tables": [{"name": "t; x", "columns": [{"name": "a", "type": "text"}]}]}, "table name"),
    ({"tables": [{"name": "t", "columns": [{"name": "id", "type": "integer"}]}]}, "repeats"),
    ({"tables": [{"name": "t", "columns": [{"name": "a", "type": "blob"}]}]}, "not allowed"),
    ({"tables": [{"name": "t", "columns": [{"name": "a", "type": "text", "default": "x"}]}]}, "Each column"),
    ({"tables": [{"name": "t", "columns": [{"name": "a b", "type": "text"}]}]}, "column name"),
    ({"tables": [{"name": "t", "columns": [{"name": "a", "type": "text", "nullable": "no"}]}]}, "nullable"),
    ({"tables": [{"name": "t", "columns": [{"name": "a", "type": "text"}]}] * 2}, "repeated"),
    ({"tables": [{"name": f"t{i}", "columns": [{"name": "a", "type": "text"}]} for i in range(21)]}, "at most"),
])
def test_bad_designs_are_refused(change, needle):
    with pytest.raises(ValueError, match=needle):
        _designer().validate({**GOOD, **change})


def test_a_wide_table_of_hundreds_of_columns_is_allowed_when_it_fits_a_mysql_row():
    cols = [{"name": f"c{i}", "type": "integer" if i % 2 else "long_text"} for i in range(500)]
    design = _designer().validate({**GOOD, "tables": [{"name": "t", "columns": cols}]})
    assert len(design.tables[0].columns) == 500


def test_sqlserver_statements_use_brackets_its_types_an_identity_key_and_give_the_reader_read_only_access():
    from agent.workflow.database_creation import SQLSERVER
    design = _designer().validate(GOOD)
    statements = build_statements(design, dialect=SQLSERVER, grant_reader="hai_sql_reader")
    assert statements[0] == "CREATE DATABASE [hai_toko]"
    assert "CREATE USER [hai_sql_reader] FOR LOGIN [hai_sql_reader]" in statements[1]
    assert "db_datareader" in statements[2] and "db_owner" not in " ".join(statements) and "db_datawriter" not in " ".join(statements)
    table = statements[3]
    assert table.startswith("CREATE TABLE [hai_toko].[dbo].[tb_bahan] ([id] INT IDENTITY(1,1) NOT NULL, [nama] NVARCHAR(255) NOT NULL")
    assert table.endswith("PRIMARY KEY ([id]))") and "`" not in table
    # adding tables to an existing database touches nothing else
    assert build_statements(design, database_exists=True, dialect=SQLSERVER, grant_reader="x") == [table]
    from agent.workflow.database_creation import CreationRefused
    with pytest.raises(CreationRefused):
        build_statements(design, dialect=SQLSERVER, grant_reader="x]; DROP")


def test_sqlserver_gives_the_writer_data_rights_in_a_new_database_only_when_there_is_a_writer():
    from agent.workflow.database_creation import SQLSERVER
    design = _designer().validate(GOOD)
    both = " ".join(build_statements(design, dialect=SQLSERVER, grant_reader="hai_sql_reader", grant_writer="hai_sql_writer"))
    assert "CREATE USER [hai_sql_writer]" in both and "db_datawriter] ADD MEMBER [hai_sql_writer]" in both
    assert "db_datawriter] ADD MEMBER [hai_sql_reader]" not in both and "db_owner" not in both
    assert "hai_sql_writer" not in " ".join(build_statements(design, dialect=SQLSERVER, grant_reader="hai_sql_reader"))
    assert "hai_sql_writer" not in " ".join(build_statements(design, database_exists=True, dialect=SQLSERVER,
                                                             grant_reader="x", grant_writer="hai_sql_writer"))
    with pytest.raises(CreationRefused):
        build_statements(design, dialect=SQLSERVER, grant_writer="w]; DROP")


def test_sqlserver_has_no_mysql_row_size_cap_but_keeps_the_500_column_limit():
    from agent.workflow.database_creation import SQLSERVER
    designer = DatabaseDesigner(None, prefix="hai_", dialect=SQLSERVER)
    wide = [{"name": f"c{i}", "type": "text"} for i in range(200)]
    assert len(designer.validate({**GOOD, "tables": [{"name": "t", "columns": wide}]}).tables[0].columns) == 200
    with pytest.raises(ValueError, match="too wide"):
        _designer().validate({**GOOD, "tables": [{"name": "t", "columns": wide}]})        # MySQL still refuses
    with pytest.raises(ValueError, match="limit is 500"):
        designer.validate({**GOOD, "tables": [{"name": "t", "columns": wide * 3}]})


def test_a_design_with_only_a_database_is_allowed():
    design = _designer().validate({"database": "hai_kosong"})
    assert design.tables == () and build_statements(design) == ["CREATE DATABASE `hai_kosong` CHARACTER SET utf8mb4"]


def test_the_designer_asks_the_llm_again_after_an_invalid_answer():
    result = _designer(json.dumps({**GOOD, "database": "dbsales"}), json.dumps(GOOD)).plan("buat toko")
    assert result.design.database == "hai_toko" and result.attempts == 2


def test_an_unsupported_request_is_a_reason_not_an_error():
    result = _designer(json.dumps({"supported": False, "reason": "hapus database tidak didukung"})).plan("drop dbsales")
    assert result.design is None and "tidak didukung" in result.reason


# ------------------------------------------------------------------ the statements
def test_statements_use_only_the_closed_type_set_and_add_the_key():
    sql = build_statements(_designer().validate(GOOD))
    assert sql[0] == "CREATE DATABASE `hai_toko` CHARACTER SET utf8mb4"
    assert "`id` INT NOT NULL AUTO_INCREMENT" in sql[1] and "PRIMARY KEY (`id`)" in sql[1]
    assert "`nama` VARCHAR(255) NOT NULL" in sql[1] and "`harga_beli` DECIMAL(14,2) NULL" in sql[1]
    joined = " ".join(sql)
    for forbidden in ("DROP", "ALTER", "GRANT", "TRUNCATE", "DELETE"):
        assert forbidden not in joined
    assert set(COLUMN_TYPES.values()) >= {"INT", "TEXT", "DATE"}


# ------------------------------------------------------------------ the service (fake connection)
class FakeConnection:
    def __init__(self, existing=(), fail_on=None, tables=()):
        self.existing, self.fail_on, self.executed, self.closed = set(existing), fail_on, [], False
        self.tables = list(tables)

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, connection):
        self.c, self.row, self.rows = connection, None, []

    def execute(self, sql, params=()):
        if sql.startswith("SELECT SCHEMA_NAME"):
            self.rows = [(params[0],)] if params[0] in self.c.existing else []
            return
        if sql.startswith("SELECT TABLE_NAME"):
            self.rows = [(t,) for t in self.c.tables]
            return
        if self.c.fail_on and self.c.fail_on in sql:
            raise RuntimeError("boom")
        self.c.executed.append(sql)

    def fetchall(self):
        return self.rows

    def close(self):
        pass


def _service(tmp_path, **kw):
    conn = FakeConnection(**kw)
    return DatabaseCreationService(designer=_designer(), connect=lambda: conn, profile="loc",
                                   audit_path=tmp_path / "a.jsonl"), conn


def test_an_existing_database_with_no_new_tables_asked_is_refused(tmp_path):
    service, conn = _service(tmp_path, existing={"hai_toko"})
    design = service.designer.validate({"database": "hai_toko"})
    with pytest.raises(CreationRefused, match="already exists"):
        service.preview(design)
    with pytest.raises(CreationRefused):
        service.execute(design, actor="a", plan_id="p", question="q")
    assert conn.executed == []


def test_tables_are_added_to_an_existing_database_without_touching_it(tmp_path):
    service, conn = _service(tmp_path, existing={"hai_toko"}, tables=["tb_lama"])
    design = service.designer.validate(GOOD)
    mode, statements = service.preview(design)
    assert mode == "add_tables" and len(statements) == 1 and statements[0].startswith("CREATE TABLE `hai_toko`.`tb_bahan`")
    assert service.confirm_phrase(design, mode) == "TAMBAH TABEL hai_toko"
    assert service.execute(design, actor="alice", plan_id="p", question="q")["mode"] == "add_tables"
    assert [s.split()[0:2] for s in conn.executed] == [["CREATE", "TABLE"]]
    audit = json.loads((tmp_path / "a.jsonl").read_text("utf-8").strip())
    assert audit["mode"] == "add_tables" and audit["statements_done"] == 1


def test_a_table_name_that_already_exists_is_refused_and_nothing_is_changed(tmp_path):
    service, conn = _service(tmp_path, existing={"hai_toko"}, tables=["TB_BAHAN"])   # case-insensitive
    design = service.designer.validate(GOOD)
    with pytest.raises(CreationRefused, match="already has the table"):
        service.preview(design)
    with pytest.raises(CreationRefused):
        service.execute(design, actor="a", plan_id="p", question="q")
    assert conn.executed == []


def test_a_new_database_is_created_and_the_phrase_says_so(tmp_path):
    service, _ = _service(tmp_path)
    design = service.designer.validate(GOOD)
    mode, statements = service.preview(design)
    assert mode == "create_database" and len(statements) == 2
    assert service.confirm_phrase(design, mode) == "BUAT DATABASE hai_toko"


def test_execute_runs_the_statements_in_order_and_audits_them(tmp_path):
    service, conn = _service(tmp_path)
    design = service.designer.validate(GOOD)
    assert service.execute(design, actor="alice", plan_id="p1", question="buat toko")["database"] == "hai_toko"
    assert [s.split()[0:2] for s in conn.executed] == [["CREATE", "DATABASE"], ["CREATE", "TABLE"]]
    audit = json.loads((tmp_path / "a.jsonl").read_text("utf-8").strip())
    assert audit["status"] == "created" and audit["actor"] == "alice" and audit["statements_done"] == 2


def test_a_failure_part_way_is_audited_and_never_drops_anything(tmp_path):
    service, conn = _service(tmp_path, fail_on="CREATE TABLE")
    design = service.designer.validate(GOOD)
    with pytest.raises(RuntimeError):
        service.execute(design, actor="alice", plan_id="p1", question="q")
    audit = json.loads((tmp_path / "a.jsonl").read_text("utf-8").strip())
    assert audit["status"] == "failed" and audit["statements_done"] == 1
    assert not any("DROP" in s for s in conn.executed)


# ------------------------------------------------------------------ profiles and grants
def test_a_creator_block_is_optional_and_needs_its_own_account():
    plain = parse_profiles({"my": BASE})["my"]
    assert plain.creator_user == "" and plain.creator_prefix == "hai_"
    ok = parse_profiles({"my": {**BASE, "creator": {"user": "hai_creator", "password_env": "C", "prefix": "ai_"}}})["my"]
    assert (ok.creator_user, ok.creator_prefix) == ("hai_creator", "ai_")
    for bad in ({"user": "hai_reader", "password_env": "C"}, {"user": "c"}, {"user": "c", "password_env": "C", "prefix": "X"},
                {"user": "c", "password_env": "C", "prefix": "hai"}, {"user": "c", "password_env": "C", "x": 1}):
        with pytest.raises(EngineError):
            parse_profiles({"my": {**BASE, "creator": bad}})


@pytest.mark.parametrize("grants", [
    ["GRANT USAGE ON *.* TO `c`@`%`", "GRANT CREATE ON `hai\\_%`.* TO `c`@`%`"]])
def test_a_creator_limited_to_create_on_the_prefix_is_accepted(grants):
    check_mysql_creator_grants(grants, "hai_")


@pytest.mark.parametrize("grant,needle", [
    ("GRANT CREATE ON *.* TO `c`@`%`", "limited to databases"),
    ("GRANT CREATE ON `dbsales`.* TO `c`@`%`", "limited to databases"),
    ("GRANT CREATE, DROP ON `hai\\_%`.* TO `c`@`%`", "only CREATE"),
    ("GRANT CREATE, SELECT ON `hai\\_%`.* TO `c`@`%`", "only CREATE"),
    ("GRANT ALL PRIVILEGES ON `hai\\_%`.* TO `c`@`%`", "only CREATE"),
    ("GRANT CREATE ON `hai\\_%`.* TO `c`@`%` WITH GRANT OPTION", "grant privileges"),
    ("GRANT CREATE ON `other\\_%`.* TO `c`@`%`", "limited to databases"),
    ("GRANT `r`@`%` TO `c`@`%`", "role or proxy"),
])
def test_a_creator_with_more_power_is_refused(grant, needle):
    with pytest.raises(EngineError, match=needle):
        check_mysql_creator_grants([grant], "hai_")


def test_a_writer_may_hold_the_prefix_pattern_only_on_a_profile_that_has_a_creator():
    pattern = ["GRANT SELECT, INSERT, UPDATE, DELETE ON `hai\\_%`.* TO `w`@`%`"]
    check_mysql_writer_grants(pattern, "hai_toko", "hai_")
    with pytest.raises(EngineError, match="outside"):
        check_mysql_writer_grants(pattern, "hai_toko")           # no creator: the pattern is out of scope
    with pytest.raises(EngineError, match="outside"):
        check_mysql_writer_grants(["GRANT SELECT ON `dbsales`.* TO `w`@`%`"], "hai_toko", "hai_")


def test_only_prefixed_databases_can_be_chosen_on_a_profile_with_a_creator():
    engine = MySQLEngine(parse_profiles({
        "my": {**BASE, "creator": {"user": "c", "password_env": "C"}}, "plain": BASE}))
    assert engine._resolve("my/hai_toko").database == "hai_toko"
    assert engine._resolve("my").database == "dbsales"
    for bad in ("my/dbsales", "my/hai_", "my/HAI_X", "my/hai_x y", "plain/hai_toko"):
        with pytest.raises(EngineError):
            engine._resolve(bad)


def test_any_database_name_can_be_read_only_when_the_switch_is_on(monkeypatch):
    engine = MySQLEngine(parse_profiles({
        "plain": {**BASE, "writer": {"user": "w", "password_env": "W"}}}))
    monkeypatch.delenv("HAI_ALLOW_ANY_DATABASE_NAME", raising=False)
    with pytest.raises(EngineError):
        engine._resolve("plain/dbsales2")                        # off by default
    monkeypatch.setenv("HAI_ALLOW_ANY_DATABASE_NAME", "true")
    assert engine._resolve("plain/dbsales2").database == "dbsales2"
    assert engine._resolve("plain/Sales_DB").database == "Sales_DB"
    for bad in ("plain/x;DROP", "plain/a b", "plain/a`b", "plain/a.b", "plain/"):
        if bad.endswith("/"):
            assert engine._resolve(bad).database == "dbsales"    # empty name = the profile's own
            continue
        with pytest.raises(EngineError):
            engine._resolve(bad)
    # such a database can also be changed, through the profile's writer account
    assert engine.has_writer("plain") and engine.has_writer("plain/dbsales2")
    # ...but only with the switch on, and only when the profile has a writer account
    monkeypatch.delenv("HAI_ALLOW_ANY_DATABASE_NAME")
    assert not engine.has_writer("plain/dbsales2")
    no_writer = MySQLEngine(parse_profiles({"plain": BASE}))
    monkeypatch.setenv("HAI_ALLOW_ANY_DATABASE_NAME", "true")
    assert not no_writer.has_writer("plain/dbsales2")


# ------------------------------------------------------------------ live: the whole path on MySQL
@pytest.fixture
def my_create(monkeypatch, tmp_path):
    import pymysql
    tag = uuid.uuid4().hex[:6]
    base_db, reader, writer, creator = f"base_{tag}", f"rd_{tag}", f"wr_{tag}", f"cr_{tag}"
    admin = pymysql.connect(host=HOST, port=int(PORT), user="root", password=ROOT_PW, autocommit=True)
    cur = admin.cursor()
    cur.execute(f"CREATE DATABASE {base_db}")
    pattern = "`hai\\_%`.*"
    for user in (reader, writer, creator):
        cur.execute(f"CREATE USER '{user}'@'%' IDENTIFIED BY 'rolepw'")
    cur.execute(f"GRANT SELECT ON {base_db}.* TO '{reader}'@'%'")
    cur.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {base_db}.* TO '{writer}'@'%'")
    cur.execute(f"GRANT SELECT ON {pattern} TO '{reader}'@'%'")
    cur.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {pattern} TO '{writer}'@'%'")
    cur.execute(f"GRANT CREATE ON {pattern} TO '{creator}'@'%'")
    monkeypatch.setenv("HAI_TEST_CR_PW", "rolepw")
    made_dbs: list[str] = []
    profiles = parse_profiles({"my_all": {
        "type": "mysql", "host": HOST, "port": int(PORT), "database": base_db, "user": reader,
        "password_env": "HAI_TEST_CR_PW", "ssl": "require",
        "writer": {"user": writer, "password_env": "HAI_TEST_CR_PW"},
        "creator": {"user": creator, "password_env": "HAI_TEST_CR_PW"}}})

    def registry(*answers):
        provider = Scripted(*answers)
        return DatabaseRuntimeRegistry(
            data_root=tmp_path, work_dir=tmp_path / "w", authorization_policy=AuthorizationPolicy(),
            llm_provider_factory=lambda: provider, profiles=profiles)

    regs = []
    try:
        yield SimpleNamespace(registry=lambda *a: regs.append(registry(*a)) or regs[-1], admin=cur,
                              made=made_dbs, tmp=tmp_path, creator=creator, tag=tag)
    finally:
        for r in regs:
            r.shutdown()
        for db in made_dbs:
            cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
        cur.execute(f"DROP DATABASE {base_db}")
        for user in (reader, writer, creator):
            cur.execute(f"DROP USER '{user}'@'%'")
        admin.close()


def _exists(env, db):
    env.admin.execute("SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s", (db,))
    return env.admin.fetchone()[0] == 1


@live
def test_a_new_database_is_created_then_connected_and_written_to(my_create):
    env = my_create
    db = f"hai_t{env.tag}"
    env.made.append(db)
    design = {**GOOD, "database": db}
    reg = env.registry(json.dumps(design))
    service = reg.database_creator()
    result = service.designer.plan("buat toko")
    assert result.design.database == db
    assert len(service.preview(result.design)[1]) == 2 and not _exists(env, db)    # preview creates nothing
    out = service.execute(result.design, actor="alice", plan_id="p", question="q")
    assert out["tables"] == ["tb_bahan"] and _exists(env, db)
    with pytest.raises(CreationRefused, match="already has the table"):            # never overwrites a table
        service.preview(result.design)
    # later: a new table in the SAME, existing database (the database is not created again)
    more = service.designer.validate({"database": db, "tables": [
        {"name": "tb_supplier", "columns": [{"name": "nama", "type": "text"}]}]})
    mode, statements = service.preview(more)
    assert mode == "add_tables" and len(statements) == 1
    assert service.execute(more, actor="alice", plan_id="p3", question="q")["mode"] == "add_tables"
    env.admin.execute(f"SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = '{db}' ORDER BY 1")
    assert [r[0] for r in env.admin.fetchall()] == ["tb_bahan", "tb_supplier"]

    # connect to it through the profile (prefix database only), approve a table for writing, insert a row
    info = reg.connect("toko", engine="mysql", source=f"my_all/{db}", actor_id="alice")
    assert {t["name"] for t in info["schema"]["tables"]} == {"tb_bahan", "tb_supplier"}
    reg.activate("toko", {"tb_bahan": {
        "columns": ["id", "nama", "harga_beli"], "filterable_columns": ["id", "nama"], "key_column": "id",
        "writable": {"operations": ["insert", "update"], "columns": ["nama", "harga_beli"]}}}, actor_id="bob")
    changes = reg.data_changes("toko")
    request = changes.planner.validate({"operation": "insert", "table": "tb_bahan",
                                        "rows": [{"nama": "gula", "harga_beli": 15000}]}, max_rows=5)
    assert changes.execute(request, expected_rows=changes.preview(request).expected_rows, actor="alice",
                           approver=None, plan_id="p2", question="q").rows_affected == 1
    env.admin.execute(f"SELECT nama, harga_beli FROM `{db}`.tb_bahan")
    assert [(n, float(h)) for n, h in env.admin.fetchall()] == [("gula", 15000.0)]


@live
def test_the_creator_cannot_touch_existing_databases_or_drop_anything(my_create):
    import pymysql
    env = my_create
    conn = pymysql.connect(host=HOST, port=int(PORT), user=env.creator, password="rolepw", autocommit=True)
    cur = conn.cursor()
    with pytest.raises(Exception):
        cur.execute("CREATE DATABASE not_prefixed_db")
    cur.execute(f"CREATE DATABASE hai_c{env.tag}")
    env.made.append(f"hai_c{env.tag}")
    with pytest.raises(Exception):
        cur.execute(f"DROP DATABASE hai_c{env.tag}")
    with pytest.raises(Exception):
        cur.execute(f"SELECT * FROM hai_c{env.tag}.nothing")
    conn.close()


@live
def test_connecting_to_a_database_outside_the_prefix_is_refused(my_create):
    reg = my_create.registry()
    from api.database_runtime_registry import RegistryError
    with pytest.raises(RegistryError, match="can be chosen"):
        reg.connect("x1", engine="mysql", source="my_all/mysql", actor_id="alice")


# ------------------------------------------------------------------ live: over HTTP
USERS["creator"] = AuthenticatedPrincipal(actor_id="carol", permissions=(CREATE,), auth_time=int(time.time()))
USERS["creator-stale"] = AuthenticatedPrincipal(actor_id="carol", permissions=(CREATE,),
                                                auth_time=int(time.time()) - 3600)
USERS["nocreate"] = AuthenticatedPrincipal(actor_id="dave", permissions=("data:change:plan",),
                                           auth_time=int(time.time()))


@live
def test_http_flow_needs_the_permission_a_fresh_login_and_the_exact_phrase(my_create):
    env = my_create
    db = f"hai_h{env.tag}"
    env.made.append(db)
    extra = {"database": db, "tables": [{"name": "tb_supplier", "columns": [{"name": "nama", "type": "text"}]}]}
    reg = env.registry(json.dumps({**GOOD, "database": db}), json.dumps({**GOOD, "database": db}),
                       json.dumps(extra))
    app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                     database_runtime_registry_factory=lambda _a: reg)
    app.add_middleware(_AsUser)
    base = "/api/v1/database-creation"

    def call(method, path, user="creator", **kw):
        return client.request(method, f"{base}{path}", headers={"X-Test-User": user}, **kw)

    with TestClient(app) as client:
        assert call("GET", "/config").json()["profiles"] == [{"name": "my_all", "prefix": "hai_", "type": "mysql"}]
        assert call("POST", "/plans", "nocreate", json={"question": "q"}).status_code == 403
        body = call("POST", "/plans", json={"question": "buat toko"}).json()
        assert body["status"] == "planned" and body["confirm_phrase"] == f"BUAT DATABASE {db}"
        assert len(body["statements"]) == 2 and not _exists(env, db)
        run = f"/plans/{body['plan_id']}/run"
        stale = call("POST", run, "creator-stale", json={"confirm": body["confirm_phrase"]})
        assert stale.status_code == 403 and stale.json()["detail"]["error"] == "step_up_required"
        wrong = call("POST", run, json={"confirm": "ya"})
        assert wrong.status_code == 422 and not _exists(env, db)
        done = call("POST", run, json={"confirm": body["confirm_phrase"]})
        assert done.status_code == 200 and done.json()["database"] == db and _exists(env, db)
        # the same tables again are refused before anything runs
        again = call("POST", "/plans", json={"question": "buat toko lagi"}).json()
        assert again["status"] == "refused" and "already has the table" in again["reason"]
        # new tables in the existing database are planned as an addition, with their own phrase
        more = call("POST", "/plans", json={"question": "tambah tabel supplier"}).json()
        assert more["status"] == "planned" and more["mode"] == "add_tables"
        assert more["confirm_phrase"] == f"TAMBAH TABEL {db}" and len(more["statements"]) == 1
