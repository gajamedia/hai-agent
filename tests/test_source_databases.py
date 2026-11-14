"""The Koneksi page lists what there is to choose from: SQLite files, and the databases of a server profile."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from api.database_engines import EngineError, MySQLEngine, PostgresEngine, SQLServerEngine, parse_profiles
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authorization import AuthorizationPolicy
from tests.test_data_changes_api import Provider, _AsUser

BASE = {"host": "127.0.0.1", "database": "dbsales", "user": "hai_reader", "password_env": "HAI_TEST_PW"}


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=()):
        self.sql = sql

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, rows, seen):
        self.rows, self.seen, self.closed = rows, seen, False

    def cursor(self):
        cursor = FakeCursor(self.rows)
        self.seen.append(cursor)
        return cursor

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("HAI_TEST_PW", "x")
    monkeypatch.delenv("HAI_ALLOW_ANY_DATABASE_NAME", raising=False)
    monkeypatch.delenv("HAI_SQLITE_WRITE", raising=False)


def _mysql(rows, **extra):
    seen, made = [], []

    def connect(**kwargs):
        made.append(kwargs)
        return FakeConnection(rows, seen)

    profiles = parse_profiles({"my": {"type": "mysql", **BASE, **extra}})
    return MySQLEngine(profiles, connect=connect), seen, made


# ------------------------------------------------------------------ MySQL, SQL Server, PostgreSQL
def test_mysql_lists_the_databases_the_reader_sees_without_the_system_ones_and_marks_what_may_be_picked():
    rows = [("information_schema",), ("mysql",), ("dbsales",), ("hai_toko",), ("kantor",), ("sys",), ("performance_schema",)]
    engine, seen, made = _mysql(rows, creator={"user": "c", "password_env": "HAI_TEST_PW", "prefix": "hai_"})
    got = engine.list_databases("my")
    assert [d["name"] for d in got] == ["dbsales", "hai_toko", "kantor"]
    assert {d["name"]: d["allowed"] for d in got} == {"dbsales": True, "hai_toko": True, "kantor": False}
    assert got[0]["default"] is True and got[1]["default"] is False
    assert seen[0].sql == "SHOW DATABASES" and made[0]["user"] == "hai_reader"      # the read-only account


def test_any_database_may_be_picked_only_when_the_server_allows_it(monkeypatch):
    engine, _, _ = _mysql([("dbsales",), ("kantor",)])
    assert {d["name"]: d["allowed"] for d in engine.list_databases("my")} == {"dbsales": True, "kantor": False}
    monkeypatch.setenv("HAI_ALLOW_ANY_DATABASE_NAME", "true")
    assert {d["name"]: d["allowed"] for d in engine.list_databases("my")} == {"dbsales": True, "kantor": True}


def test_an_unknown_profile_or_a_missing_password_is_a_plain_engine_error(monkeypatch):
    engine, _, _ = _mysql([])
    with pytest.raises(EngineError, match="unknown"):
        engine.list_databases("nope")
    monkeypatch.delenv("HAI_TEST_PW")
    with pytest.raises(EngineError, match="not set"):
        engine.list_databases("my")


def test_sqlserver_asks_only_for_user_databases_the_login_can_open():
    seen = []

    def connect(text, timeout=0, **kw):
        return FakeConnection([("dbsales",), ("hai_hr",)], seen)

    profiles = parse_profiles({"ms": {"type": "sqlserver", **BASE, "driver": "ODBC Driver 17 for SQL Server"}})
    got = SQLServerEngine(profiles, connect=connect).list_databases("ms")
    assert [d["name"] for d in got] == ["dbsales", "hai_hr"]
    assert "database_id > 4" in seen[0].sql and "HAS_DBACCESS" in seen[0].sql


def test_postgresql_is_one_database_per_profile():
    profiles = parse_profiles({"pg": {"type": "postgresql", **BASE}})
    assert PostgresEngine(profiles).list_databases("pg") == [{"name": "dbsales", "allowed": True, "default": True}]


# ------------------------------------------------------------------ over HTTP
@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data"
    (data / "sub").mkdir(parents=True)
    for name in ("sales.sqlite3", "hai_toko.sqlite3", "old.db", "sub/deep.sqlite"):
        sqlite3.connect(data / name).close()
    (data / "notes.txt").write_text("x")
    engine, _, _ = _mysql([("dbsales",), ("kantor",)])
    provider = Provider()
    app = create_app(
        authorization_policy_factory=lambda _a: AuthorizationPolicy(),
        database_runtime_registry_factory=lambda a: DatabaseRuntimeRegistry(
            data_root=data, work_dir=tmp_path / "work", authorization_policy=a.state.authorization_policy,
            llm_provider_factory=lambda: provider, engines={"mysql": engine}))
    app.add_middleware(_AsUser)
    with TestClient(app) as client:
        yield client


def _get(client, path, user="alice"):
    return client.get(f"/api/v1/database-connections{path}", headers={"X-Test-User": user})


def test_the_files_of_the_data_folder_are_listed_and_marked_when_connected(world):
    assert [f["name"] for f in _get(world, "/files").json()["files"]] == [
        "hai_toko.sqlite3", "old.db", "sales.sqlite3", "sub/deep.sqlite"]
    assert all(f["connected"] is False for f in _get(world, "/files").json()["files"])
    assert world.post("/api/v1/database-connections", json={"name": "sales", "filename": "sales.sqlite3"},
                      headers={"X-Test-User": "alice"}).status_code == 200
    marked = {f["name"]: f["connected"] for f in _get(world, "/files").json()["files"]}
    assert marked["sales.sqlite3"] is True and marked["old.db"] is False


def test_a_servers_databases_are_listed_for_a_connection_manager_only(world):
    body = _get(world, "/profiles/my/databases").json()
    assert body["profile"] == "my" and [d["name"] for d in body["databases"]] == ["dbsales", "kantor"]
    assert body["databases"][0]["allowed"] is True and body["databases"][1]["allowed"] is False
    assert all("connected" in d for d in body["databases"])
    assert _get(world, "/profiles/my/databases", "reader").status_code == 403
    assert _get(world, "/files", "reader").status_code == 403
    assert _get(world, "/profiles/nope/databases").status_code == 404


def test_a_driver_failure_never_reaches_the_browser(tmp_path):
    def boom(**kwargs):
        raise RuntimeError("Access denied for user 'hai_reader'@'10.0.0.5' (password: YES)")

    engine = MySQLEngine(parse_profiles({"my": {"type": "mysql", **BASE}}), connect=boom)
    data = tmp_path / "d"
    data.mkdir()
    registry = DatabaseRuntimeRegistry(
        data_root=data, work_dir=tmp_path / "w", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: None, engines={"mysql": engine})
    from api.database_runtime_registry import RegistryError
    with pytest.raises(RegistryError) as info:
        registry.list_source_databases("my")
    assert "10.0.0.5" not in str(info.value) and "hai_reader" not in str(info.value)


def test_the_sqlite_profile_lists_the_databases_this_application_made(tmp_path, monkeypatch):
    from api.database_engines import SQLiteEngine
    for name in ("hai_toko.sqlite3", "hai_dua.sqlite3", "sales.sqlite3"):
        sqlite3.connect(tmp_path / name).close()
    engine = SQLiteEngine(tmp_path)
    with pytest.raises(EngineError):
        engine.list_databases("sqlite")                              # writing is off: no such profile
    monkeypatch.setenv("HAI_SQLITE_WRITE", "true")
    assert [d["name"] for d in engine.list_databases("sqlite")] == ["hai_dua", "hai_toko"]
