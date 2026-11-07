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
from tests.test_table_report_job_e2e import _context
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
    {"password": "secret"}, {"type": "mysql"}, {"sslmode": "none"}, {"schema": "Public;drop"},
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
    cur.execute(f'CREATE TABLE {schema}."MixedCase" (id integer)')
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
        yield SimpleNamespace_(registry=registry, schema=schema, admin=cur, profiles=profiles)
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
    assert set(tables) == {"sales"}  # mixed-case table is skipped: the adapter cannot quote it
    cols = {c["name"]: c for c in tables["sales"]["columns"]}
    assert cols["id"]["primary_key"] is True and cols["amount"]["declared_type"] == "numeric"
    assert out["warnings"] == []


@live
def test_session_is_read_only(pg):
    reg = pg.registry()
    reg.connect("pgsales", engine="postgresql", source="pg_reader", actor_id="alice")
    conn = reg._connections["pgsales"]
    with conn.manager.connection("pgsales") as c:
        cur = c.cursor()
        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        with pytest.raises(Exception):
            cur.execute(f"CREATE TABLE {pg.schema}.x (a int)")


@live
def test_question_to_report_matches_sql_including_numeric_and_filters(pg):
    reg = pg.registry(_plan(filters={"product": "gadget"}))
    reg.connect("pgsales", engine="postgresql", source="pg_reader", actor_id="alice")
    reg.activate("pgsales", {"sales": {
        "columns": ["id", "region", "product", "amount"], "filterable_columns": ["region", "product"],
        "key_column": "id", "required_non_null_columns": ["amount"]}}, actor_id="bob")
    outcome = reg.service("pgsales").ask("Penjualan gadget per wilayah", execution_context=_context())
    assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
    pg.admin.execute(f"SELECT region, SUM(amount), COUNT(id) FROM {pg.schema}.sales "
                     f"WHERE product='gadget' GROUP BY region")
    expected = {r: (float(s), n) for r, s, n in pg.admin.fetchall()}
    got = {r["region"]: (float(r["revenue"]), r["orders"]) for r in outcome.rows}
    assert got.keys() == expected.keys()
    for region, (rev, n) in expected.items():
        assert got[region][0] == pytest.approx(rev) and got[region][1] == n


@live
def test_superuser_and_write_capable_roles_are_refused(pg):
    reg = pg.registry()
    with pytest.raises(RegistryError, match="superuser"):
        reg.connect("a1", engine="postgresql", source="pg_admin", actor_id="alice")
    with pytest.raises(RegistryError, match="can modify data"):
        reg.connect("a2", engine="postgresql", source="pg_writer", actor_id="alice")
    assert reg.list() == []


@live
def test_connection_errors_do_not_leak_host_user_or_password(pg):
    reg = pg.registry()
    with pytest.raises(RegistryError) as info:
        reg.connect("a3", engine="postgresql", source="pg_wrongpw", actor_id="alice")
    text = str(info.value)
    assert "127.0.0.1" not in text and "rd_" not in text and ADMIN_PW not in text
    assert "salesdb" not in text


@live
def test_postgres_connection_survives_restart(pg):
    reg = pg.registry()
    reg.connect("pgsales", engine="postgresql", source="pg_reader", actor_id="alice")
    reg.activate("pgsales", {"sales": {"columns": ["id", "region", "amount"], "key_column": "id"}},
                 actor_id="bob")
    reg.shutdown()
    again = pg.registry()
    (item,) = again.list()
    assert item["status"] == "active" and item["database"] == "postgresql"
    assert item["file"] == "pg_reader"


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
