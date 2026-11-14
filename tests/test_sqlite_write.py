"""SQLite as a database the agent can create and change data in, next to MySQL and SQL Server.

Nothing here needs a server: a SQLite database is a file in the data folder. Writing is opt-in
(HAI_SQLITE_WRITE=true); without it SQLite stays read-only, as it always was.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.llm.service import LLMService
from agent.workflow.database_creation import (
    SQLITE, CreationRefused, DatabaseDesigner, SQLiteCreationService, build_statements,
)
from api.database_engines import EngineError, SQLiteEngine
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from tests.test_data_changes_api import USERS, Provider, _AsUser
from tests.test_database_creation import GOOD, Scripted

CREATE, ENTRY = "database:create:manage", "data:entry:record"
USERS["maker"] = AuthenticatedPrincipal(actor_id="maker", permissions=(CREATE,), auth_time=int(time.time()))
USERS["sclerk"] = AuthenticatedPrincipal(actor_id="sclerk", permissions=(ENTRY,), auth_time=int(time.time()))

DESIGN = {"supported": True, "database": "hai_toko", "title": "Toko", "tables": [
    {"name": "tb_penjualan", "columns": [{"name": "produk", "type": "text", "nullable": False},
                                         {"name": "qty", "type": "integer"},
                                         {"name": "harga", "type": "decimal"},
                                         {"name": "tanggal", "type": "date"}]}]}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("HAI_SQLITE_WRITE", "true")
    monkeypatch.delenv("HAI_ALLOW_ANY_DATABASE_NAME", raising=False)
    monkeypatch.delenv("HAI_DATA_CHANGE_APPROVAL", raising=False)


def _designer(*answers, prefix="hai_"):
    return DatabaseDesigner(LLMService(Scripted(*answers)), prefix=prefix, dialect=SQLITE)


def _service(tmp_path, *answers):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return SQLiteCreationService(designer=_designer(*answers), data_root=data, profile="sqlite",
                                 audit_path=tmp_path / "audit.jsonl"), data


def _tables(path: Path) -> list[str]:
    with sqlite3.connect(path) as c:
        return sorted(r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                              "AND name NOT LIKE 'sqlite_%'"))


# ------------------------------------------------------------------ statements and the service
def test_the_statements_make_an_integer_key_and_types_that_read_back_as_the_same_kinds():
    design = _designer().validate(DESIGN)
    (sql,) = build_statements(design, dialect=SQLITE)
    assert sql.startswith('CREATE TABLE "tb_penjualan" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "produk" TEXT NOT NULL')
    assert '"qty" INTEGER NULL' in sql and '"harga" NUMERIC NULL' in sql and '"tanggal" DATE NULL'
    assert "DATABASE" not in sql                                   # the file is the database


def test_creating_makes_the_file_with_its_tables_and_verifies_it(tmp_path):
    service, data = _service(tmp_path)
    design = service.designer.validate(DESIGN)
    mode, statements = service.preview(design)
    assert mode == "create_database" and len(statements) == 1 and not (data / "hai_toko.sqlite3").exists()
    done = service.execute(design, actor="maker", plan_id="p1", question="buat toko")
    assert done == {"database": "hai_toko", "tables": ["tb_penjualan"], "profile": "sqlite", "mode": "create_database"}
    assert _tables(data / "hai_toko.sqlite3") == ["tb_penjualan"]
    assert service.verify(design)["confirmed"] is True
    audit = json.loads((tmp_path / "audit.jsonl").read_text("utf-8").splitlines()[0])
    assert audit["status"] == "created" and audit["actor"] == "maker"


def test_an_existing_database_is_never_overwritten_and_only_gets_new_tables(tmp_path):
    service, data = _service(tmp_path)
    design = service.designer.validate(DESIGN)
    service.execute(design, actor="maker", plan_id="p1", question="q")
    before = (data / "hai_toko.sqlite3").read_bytes()
    with pytest.raises(CreationRefused, match="already has the table"):
        service.preview(design)                                     # the same table again
    with pytest.raises(CreationRefused, match="already exists"):
        service.preview(service.designer.validate({**DESIGN, "tables": []}))   # no tables: nothing to add
    assert (data / "hai_toko.sqlite3").read_bytes() == before
    more = service.designer.validate({**DESIGN, "tables": [
        {"name": "tb_pelanggan", "columns": [{"name": "nama", "type": "text"}]}]})
    assert service.preview(more)[0] == "add_tables"
    service.execute(more, actor="maker", plan_id="p2", question="q")
    assert _tables(data / "hai_toko.sqlite3") == ["tb_pelanggan", "tb_penjualan"]


def test_a_failed_creation_leaves_no_file_behind(tmp_path):
    service, data = _service(tmp_path)
    from agent.workflow.database_creation import ColumnDef, DatabaseDesign, TableDef
    one = (ColumnDef("a", "text", True),)
    # Validation now puts tb_ in front of every name, so a name SQLite refuses is built by hand here.
    bad = DatabaseDesign("hai_toko", (TableDef("tb_ok", one), TableDef("sqlite_reserved", one)), "t")
    with pytest.raises(sqlite3.Error):
        service.execute(bad, actor="maker", plan_id="p1", question="q")
    assert not (data / "hai_toko.sqlite3").exists()
    assert json.loads((tmp_path / "audit.jsonl").read_text("utf-8").splitlines()[-1])["status"] == "failed"


@pytest.mark.parametrize("name", ["dbsales", "hai_", "hai_../x", "../hai_x", "hai_x/y", "HAI_TOKO", "hai_x.sqlite3"])
def test_a_name_that_is_not_a_plain_prefixed_name_never_reaches_the_file_system(tmp_path, name):
    service, data = _service(tmp_path)
    from agent.workflow.database_creation import DatabaseDesign
    with pytest.raises(CreationRefused):
        service.preview(DatabaseDesign(name))
    assert list(data.iterdir()) == []


# ------------------------------------------------------------------ the engine
def test_writing_is_off_unless_switched_on(tmp_path, monkeypatch):
    engine = SQLiteEngine(tmp_path)
    assert engine.profiles and engine.has_creator("sqlite")
    monkeypatch.delenv("HAI_SQLITE_WRITE")
    assert engine.profiles == {} and not engine.has_creator("sqlite")
    (tmp_path / "hai_a.sqlite3").write_bytes(b"")
    assert engine.has_writer("hai_a.sqlite3") is False
    with pytest.raises(EngineError):
        engine.open_writer("hai_a.sqlite3")


def test_the_writer_only_opens_what_this_application_made_unless_any_name_is_allowed(tmp_path, monkeypatch):
    sqlite3.connect(tmp_path / "hai_a.sqlite3").close()
    sqlite3.connect(tmp_path / "other.sqlite3").close()
    engine = SQLiteEngine(tmp_path)
    assert engine.has_writer("hai_a.sqlite3") and engine.has_writer("sqlite/hai_a")
    assert not engine.has_writer("other.sqlite3") and not engine.has_writer("sqlite/other")
    assert not engine.has_writer("../hai_a.sqlite3") and not engine.has_writer("sqlite/hai_a/../x")
    monkeypatch.setenv("HAI_ALLOW_ANY_DATABASE_NAME", "true")
    assert engine.has_writer("other.sqlite3") and engine.has_writer("sqlite/other")


def test_the_reader_stays_read_only_even_when_writing_is_on(tmp_path):
    from agent.database import ConnectionManager
    path = tmp_path / "hai_a.sqlite3"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT)")
    manager = ConnectionManager()
    SQLiteEngine(tmp_path).open(manager, "a", "sqlite/hai_a", "ref")
    with manager.connection("a") as connection:
        with pytest.raises(Exception):
            connection.execute("INSERT INTO t (a) VALUES ('x')")


# ------------------------------------------------------------------ the whole flow over HTTP
@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    provider = Provider()
    app = create_app(
        authorization_policy_factory=lambda _a: AuthorizationPolicy(),
        database_runtime_registry_factory=lambda a: DatabaseRuntimeRegistry(
            data_root=data, work_dir=tmp_path / "work", authorization_policy=a.state.authorization_policy,
            llm_provider_factory=lambda: provider))
    app.add_middleware(_AsUser)
    with TestClient(app) as client:
        yield client, provider, data


def _call(client, method, path, user, **kw):
    return client.request(method, f"/api/v1{path}", headers={"X-Test-User": user}, **kw)


def test_create_connect_approve_record_and_change_on_sqlite_end_to_end(world):
    client, provider, data = world
    # 1. the Buat database page offers SQLite
    assert _call(client, "GET", "/database-creation/config", "maker").json()["profiles"] == [
        {"name": "sqlite", "prefix": "hai_", "type": "sqlite"}]
    # 2. create the database with its exact phrase
    provider.answers.append(json.dumps(DESIGN))
    plan = _call(client, "POST", "/database-creation/plans", "maker", json={"question": "buat toko"}).json()
    assert plan["status"] == "planned" and plan["confirm_phrase"] == "BUAT DATABASE hai_toko"
    assert not (data / "hai_toko.sqlite3").exists()
    done = _call(client, "POST", f"/database-creation/plans/{plan['plan_id']}/run", "maker",
                 json={"confirm": plan["confirm_phrase"]})
    assert done.status_code == 200, done.text
    assert _tables(data / "hai_toko.sqlite3") == ["tb_penjualan"]
    # 3. connect it by profile and database, then approve reading and writing
    assert _call(client, "POST", "/database-connections", "alice",
                 json={"name": "toko", "profile": "sqlite", "database": "hai_toko"}).status_code == 200
    selection = {"tables": {"tb_penjualan": {
        "columns": ["id", "produk", "qty", "harga", "tanggal"], "filterable_columns": ["id", "produk"],
        "key_column": "id", "writable": {"operations": ["insert", "update", "delete"],
                                         "columns": ["produk", "qty", "harga", "tanggal"]}}}}
    approved = _call(client, "POST", "/database-connections/toko/approve", "alice", json=selection)
    assert approved.status_code == 200, approved.text
    # 4. a clerk records two sales: look first, then Simpan
    provider.answers.append(json.dumps({"supported": True, "operation": "insert", "table": "tb_penjualan", "rows": [
        {"produk": "kopi", "qty": 10, "harga": 25000, "tanggal": "2026-11-10"},
        {"produk": "teh", "qty": 5, "harga": 15000, "tanggal": "2026-11-10"}]}))
    entry = _call(client, "POST", "/data-entries/plans", "sclerk", json={"question": "catat penjualan"}).json()
    assert entry["status"] == "planned" and entry["expected_rows"] == 2 and entry["database"] == "toko"
    saved = _call(client, "POST", f"/data-entries/plans/{entry['plan_id']}/run", "sclerk")
    assert saved.status_code == 200, saved.text
    assert saved.json()["rows_affected"] == 2 and saved.json()["verification"] == "verified"
    with sqlite3.connect(data / "hai_toko.sqlite3") as c:
        assert c.execute("SELECT produk, qty FROM tb_penjualan ORDER BY id").fetchall() == [("kopi", 10), ("teh", 5)]
    # 5. a writer corrects one of them (exact phrase and a fresh login, as for every change)
    provider.answers.append(json.dumps({"supported": True, "operation": "update", "table": "tb_penjualan",
                                        "set": {"qty": 12},
                                        "conditions": [{"column": "produk", "operator": "eq", "value": "kopi"}]}))
    change = _call(client, "POST", "/data-changes/plans", "alice", json={"question": "kopi jadi 12", "max_rows": 10}).json()
    assert change["status"] == "planned" and change["expected_rows"] == 1
    ran = _call(client, "POST", f"/data-changes/plans/{change['plan_id']}/run", "alice",
                json={"confirm": change["confirm_phrase"]})
    assert ran.status_code == 200, ran.text
    assert ran.json()["verification"] == "verified"
    with sqlite3.connect(data / "hai_toko.sqlite3") as c:
        assert c.execute("SELECT qty FROM tb_penjualan WHERE produk = 'kopi'").fetchone() == (12,)


def test_without_the_switch_sqlite_cannot_create_and_is_not_offered(world, monkeypatch):
    client, _, data = world
    monkeypatch.delenv("HAI_SQLITE_WRITE")
    assert _call(client, "GET", "/database-creation/config", "maker").json()["profiles"] == []
    sqlite3.connect(data / "hai_old.sqlite3").close()
    r = _call(client, "POST", "/database-connections", "alice", json={"name": "old", "filename": "hai_old.sqlite3"})
    assert r.status_code == 200                                    # reading is unchanged
