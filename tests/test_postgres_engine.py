"""PostgreSQL engine tests.

Offline tests always run. Live tests need a disposable PostgreSQL server:

    docker run -d --name hai_test_pg -e POSTGRES_PASSWORD=testpw -e POSTGRES_DB=salesdb \
        -p 127.0.0.1:55432:5432 postgres:16-alpine
    HAI_PG_TEST_PORT=55432 HAI_PG_TEST_ADMIN_PASSWORD=testpw pytest tests/test_postgres_engine.py
"""
from __future__ import annotations

import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from agent.llm.contracts import LLMResponse
from api.database_engines import EngineError, map_postgres_type, parse_profiles
from api.database_runtime_registry import DatabaseRuntimeRegistry, RegistryError
from api.main import create_app
from api.security.authorization import AuthorizationPolicy
from tests.engine_conformance import ConformanceSuite, Harness
from tests.test_table_report_job_e2e import _context  # noqa: F401
from tests.test_table_reports_api import PERMISSION, READ, _Attach, _principal

CONNECT = "database:connection:manage"
PORT = os.getenv("HAI_PG_TEST_PORT")
ADMIN_PW = os.getenv("HAI_PG_TEST_ADMIN_PASSWORD", "")
live = pytest.mark.skipif(not PORT, reason="set HAI_PG_TEST_PORT for live PostgreSQL tests")

GOOD = {"type": "postgresql", "host": "h", "database": "d", "user": "u", "password_env": "P"}


# ------------------------------------------------------------------ offline
def test_profile_parsing_accepts_a_valid_profile_and_applies_safe_defaults():
    p = parse_profiles({"crm": GOOD})["crm"]
    assert (p.port, p.schema, p.sslmode) == (5432, "public", "require")


@pytest.mark.parametrize("change", [
    {"password": "secret"}, {"type": "oracle"}, {"sslmode": "none"}, {"schema": "Public;drop"},
    {"port": 0}, {"port": "x"}, {"statement_timeout_ms": 1}, {"extra": 1}, {"host": " "},
    {"password_env": ""}])
def test_invalid_profiles_are_rejected(change):
    with pytest.raises(EngineError):
        parse_profiles({"crm": {**GOOD, **change}})


def test_profile_names_are_validated():
    with pytest.raises(EngineError):
        parse_profiles({"Bad Name": GOOD})


def test_a_password_in_the_file_is_refused_with_guidance():
    with pytest.raises(EngineError, match="password_env"):
        parse_profiles({"crm": {**GOOD, "password": "x"}})


def test_postgres_type_mapping():
    assert [map_postgres_type(t) for t in (
        "integer", "bigint", "numeric", "double precision", "text", "character varying",
        "boolean", "date", "timestamp without time zone", "timestamp with time zone", "uuid")] == [
        "integer", "integer", "number", "number", "string", "string", "boolean", "date",
        "datetime", "datetime", "string"]


def test_unconfigured_registry_has_no_postgres_and_unknown_profile_is_refused(tmp_path):
    (tmp_path / "d").mkdir()
    reg = DatabaseRuntimeRegistry(data_root=tmp_path / "d", work_dir=tmp_path / "w",
                                  authorization_policy=AuthorizationPolicy(),
                                  llm_provider_factory=lambda: None)
    assert reg.available_engines() == ["sqlite"] and reg.profiles() == []
    with pytest.raises(RegistryError, match="not available"):
        reg.connect("crm", engine="postgresql", source="crm", actor_id="a")


def test_missing_password_variable_is_reported_by_name_only(tmp_path, monkeypatch):
    monkeypatch.delenv("HAI_TEST_MISSING_PW", raising=False)
    (tmp_path / "d").mkdir()
    reg = DatabaseRuntimeRegistry(
        data_root=tmp_path / "d", work_dir=tmp_path / "w", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: None,
        profiles=parse_profiles({"crm": {**GOOD, "password_env": "HAI_TEST_MISSING_PW"}}))
    with pytest.raises(RegistryError, match="HAI_TEST_MISSING_PW is not set"):
        reg.connect("crm", engine="postgresql", source="crm", actor_id="a")


# --------------------------------------------------------------------- live
class _Provider:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _plan(**extra):
    return json.dumps({"supported": True, "table": "sales", "group_by": ["region"],
                       "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"},
                                   {"output": "orders", "operation": "count"}], **extra})


@pytest.fixture
def pg(monkeypatch, tmp_path):
    import psycopg
    tag = uuid.uuid4().hex[:8]
    schema, reader, writer = f"rep_{tag}", f"rd_{tag}", f"wr_{tag}"
    admin = psycopg.connect(host="127.0.0.1", port=int(PORT), dbname="salesdb",
                            user="postgres", password=ADMIN_PW, autocommit=True)
    cur = admin.cursor()
    cur.execute(f"CREATE SCHEMA {schema}")
    cur.execute(f"CREATE TABLE {schema}.sales (id integer PRIMARY KEY, region text, "
                f"product text, amount numeric(10,2), sold_on date)")
    cur.execute(f'ALTER TABLE {schema}.sales ADD COLUMN "weird name" text')
    cur.execute(f'CREATE TABLE {schema}."Order" (id integer PRIMARY KEY, "group" text, '
                f'"user" text, amount numeric(8,2))')
    cur.executemany(f'INSERT INTO {schema}."Order" VALUES (%s,%s,%s,%s)',
                    [(i, ["a", "b"][i % 2], "u1", f"{i}.50") for i in range(1, 9)])
    rows = [(i, ["west", "east", "north"][i % 3], "widget" if i % 2 else "gadget",
             f"{10 + i}.25", f"2026-01-{(i % 28) + 1:02d}") for i in range(1, 31)]
    cur.executemany(f"INSERT INTO {schema}.sales VALUES (%s,%s,%s,%s,%s)", rows)
    for role in (reader, writer):
        cur.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'rolepw'")
        cur.execute(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
        cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {role}")
    cur.execute(f"GRANT INSERT ON {schema}.sales TO {writer}")
    monkeypatch.setenv("HAI_TEST_PG_ROLE_PW", "rolepw")
    monkeypatch.setenv("HAI_TEST_PG_ADMIN_PW", ADMIN_PW)

    def profile(user, password_env="HAI_TEST_PG_ROLE_PW", **extra):
        return {"type": "postgresql", "host": "127.0.0.1", "port": int(PORT), "database": "salesdb",
                "user": user, "password_env": password_env, "schema": schema,
                "sslmode": "disable", **extra}

    profiles = parse_profiles({
        "pg_reader": profile(reader), "pg_writer": profile(writer),
        "pg_admin": profile("postgres", "HAI_TEST_PG_ADMIN_PW"),
        "pg_wrongpw": profile(reader, "HAI_TEST_PG_ADMIN_PW")})
    made = []

    def registry(*answers):
        r = DatabaseRuntimeRegistry(
            data_root=tmp_path, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
            llm_provider_factory=lambda: _Provider(*answers), profiles=profiles)
        made.append(r)
        return r
    try:
        yield SimpleNamespace_(registry=registry, schema=schema, admin=cur, profiles=profiles,
                               reader=reader)
    finally:
        for r in made:
            r.shutdown()
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
        for role in (reader, writer):
            cur.execute(f"DROP OWNED BY {role}")
            cur.execute(f"DROP ROLE {role}")
        admin.close()


class SimpleNamespace_:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@live
def test_select_only_role_connects_and_schema_is_discovered_with_mapped_types(pg):
    reg = pg.registry()
    out = reg.connect("pgsales", engine="postgresql", source="pg_reader", actor_id="alice")
    tables = {t["name"]: t for t in out["schema"]["tables"]}
    assert set(tables) == {"sales", "Order"}  # mixed case is kept: identifiers are quoted
    assert "weird name" not in {c["name"] for c in tables["sales"]["columns"]}
    cols = {c["name"]: c for c in tables["sales"]["columns"]}
    assert cols["id"]["primary_key"] is True and cols["amount"]["declared_type"] == "numeric"
    assert any("hidden" in w for w in out["warnings"])  # the column with a space


@live
def test_superuser_and_write_capable_roles_are_refused(pg):
    reg = pg.registry()
    with pytest.raises(RegistryError, match="superuser"):
        reg.connect("a1", engine="postgresql", source="pg_admin", actor_id="alice")
    with pytest.raises(RegistryError, match="can modify data"):
        reg.connect("a2", engine="postgresql", source="pg_writer", actor_id="alice")
    assert reg.list() == []


@live
def test_api_exposes_profile_names_only_and_connects_by_profile(pg, tmp_path):
    registry = pg.registry()
    app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                     database_runtime_registry_factory=lambda _a: registry)
    app.add_middleware(_Attach, principal=_principal((CONNECT, PERMISSION, READ)))
    with TestClient(app) as client:
        listing = client.get("/api/v1/database-connections/profiles")
        assert listing.status_code == 200
        assert {p["name"] for p in listing.json()["profiles"]} >= {"pg_reader"}
        assert "127.0.0.1" not in listing.text and "salesdb" not in listing.text
        ok = client.post("/api/v1/database-connections", json={"name": "pgsales", "profile": "pg_reader"})
        assert ok.status_code == 200 and ok.json()["database"] == "postgresql"
        for body in ({"name": "x1", "profile": "pg_reader", "filename": "a.db"},
                     {"name": "x1"}, {"name": "x1", "profile": "pg_reader", "password": "p"},
                     {"name": "x1", "profile": "pg_reader", "type": "sqlite"}):
            assert client.post("/api/v1/database-connections", json=body).status_code == 422
        assert client.post("/api/v1/database-connections",
                           json={"name": "x2", "profile": "nope"}).status_code == 422


def _harness(pg, registry=None):
    def probe(connection):
        cur = connection.cursor()
        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        with pytest.raises(Exception):
            cur.execute(f"CREATE TABLE {pg.schema}.x (a int)")

    def sql(query):
        pg.admin.execute(query)
        return pg.admin.fetchall()
    return Harness(
        engine="postgresql", registry=registry or pg.registry, reader="pg_reader",
        writer="pg_writer", privileged="pg_admin", wrong_password="pg_wrongpw", sql=sql,
        q_sales=f"{pg.schema}.sales", q_order=f'{pg.schema}."Order"', order_name="Order",
        quote=lambda column: f'"{column}"', assert_read_only=probe,
        leak_tokens=["127.0.0.1", "salesdb", ADMIN_PW, pg.reader])


@live
class TestPostgresConformance(ConformanceSuite):
    """The shared engine contract (see tests/engine_conformance.py)."""

    @pytest.fixture
    def harness(self, pg):
        return _harness(pg)


@live
def test_the_conformance_suite_rejects_an_engine_that_skips_the_role_check(pg, tmp_path):
    """The suite must fail a faulty engine, otherwise it proves nothing."""
    from api.database_engines import PostgresEngine

    class SkipsRoleCheck(PostgresEngine):
        def _check_role(self, connection, profile):
            return None  # superusers and writers would slip through

    def faulty_registry(*answers):
        r = DatabaseRuntimeRegistry(
            data_root=tmp_path, work_dir=tmp_path / "faulty", authorization_policy=AuthorizationPolicy(),
            llm_provider_factory=lambda: _Provider(*answers), profiles=pg.profiles,
            engines={"postgresql": SkipsRoleCheck(pg.profiles)})
        faulty_registry.made.append(r)
        return r
    faulty_registry.made = []
    try:
        with pytest.raises(pytest.fail.Exception):
            ConformanceSuite().test_4_privileged_and_write_capable_accounts_are_refused(
                _harness(pg, faulty_registry))
    finally:
        for r in faulty_registry.made:
            r.shutdown()
