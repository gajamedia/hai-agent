from __future__ import annotations

import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from agent.llm.contracts import LLMResponse
from api.database_runtime_registry import DatabaseRuntimeRegistry, map_sqlite_type
from api.main import create_app
from api.security.authorization import AuthorizationPolicy
from tests.test_table_report_job_e2e import _make_db, _sample_rows
from tests.test_table_reports_api import PERMISSION, READ, _Attach, _principal

CONNECT = "database:connection:manage"
DB, REPORTS = "/api/v1/database-connections", "/api/v1/table-reports"
APPROVE = {"tables": {"sales": {
    "columns": ["id", "region", "product", "amount"], "filterable_columns": ["region", "product"],
    "key_column": "id", "required_non_null_columns": ["amount"]}}}


class _Provider:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)

    def complete(self, request):
        self.last = request
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _plan(**extra):
    return json.dumps({"supported": True, "table": "sales", "group_by": ["region"],
                       "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"}],
                       "title": "Revenue", **extra})


@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_db(data / "sales.sqlite3", _sample_rows(40))
    (tmp_path / "secret.sqlite3").write_bytes((data / "sales.sqlite3").read_bytes())
    holder = {}

    def build(perms=(CONNECT, PERMISSION, READ), *answers):
        provider = _Provider(*answers)
        app = create_app(
            authorization_policy_factory=lambda _a: AuthorizationPolicy(),
            database_runtime_registry_factory=lambda a: DatabaseRuntimeRegistry(
                data_root=data, work_dir=tmp_path / "work",
                authorization_policy=a.state.authorization_policy,
                llm_provider_factory=lambda: provider))
        app.add_middleware(_Attach, principal=_principal(perms))
        client = TestClient(app)
        client.__enter__()
        holder["client"] = client
        return client, provider
    yield build, tmp_path
    if "client" in holder:
        holder["client"].__exit__(None, None, None)


def _connect(client, name="sales", filename="sales.sqlite3"):
    return client.post(DB, json={"name": name, "filename": filename})


def test_full_flow_connect_approve_ask_matches_sql(world):
    build, tmp = world
    client, _ = build((CONNECT, PERMISSION, READ), _plan(filters={"product": "gadget"}))
    r = _connect(client)
    assert r.status_code == 200 and r.json()["status"] == "connected_pending_approval"
    assert {t["name"] for t in r.json()["schema"]["tables"]} == {"sales"}
    assert client.post(f"{DB}/sales/approve", json=APPROVE).json()["status"] == "active"
    body = client.post(REPORTS, json={"question": "Penjualan gadget per wilayah"}).json()
    assert body["status"] == "succeeded", body
    report = json.loads((tmp / "work" / "artifacts" / "sales" / body["artifact"]["relative_path"]
                         ).read_text("utf-8"))
    with sqlite3.connect(tmp / "data" / "sales.sqlite3") as db:
        expected = dict(db.execute(
            "SELECT region, SUM(amount) FROM sales WHERE product='gadget' GROUP BY region"))
    assert {x["region"]: x["revenue"] for x in report["aggregation"]["rows"]} == pytest.approx(expected)


@pytest.mark.parametrize("filename", [
    "../secret.sqlite3", "..\\secret.sqlite3", "sub/../../secret.sqlite3",
    "C:\\Windows\\win.ini", "/etc/passwd", "sales.txt", "missing.sqlite3", "", None, 5,
    "sales.sqlite3\x00.txt"])
def test_filename_cannot_escape_the_data_folder_or_use_other_types(world, filename):
    client, _ = world[0]()
    r = client.post(DB, json={"name": "x1", "filename": filename})
    assert r.status_code in (404, 422), r.text
    assert client.get(DB).json()["connections"] == []


def test_symlink_escape_is_refused(world):
    build, tmp = world
    link = tmp / "data" / "link.sqlite3"
    try:
        os.symlink(tmp / "secret.sqlite3", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")
    client, _ = build()
    assert _connect(client, filename="link.sqlite3").status_code == 422


def test_only_sqlite_and_known_fields_are_accepted(world):
    client, _ = world[0]()
    for body in ({"name": "x1", "filename": "sales.sqlite3", "type": "postgresql"},
                 {"name": "x1", "filename": "sales.sqlite3", "password": "p"},
                 {"name": "x1", "filename": "sales.sqlite3", "host": "evil"}):
        assert client.post(DB, json=body).status_code == 422


@pytest.mark.parametrize("name", ["", "A", "1abc", "has space", "x" * 40, "a;drop"])
def test_bad_connection_names_are_refused(world, name):
    client, _ = world[0]()
    assert _connect(client, name=name).status_code == 422


def test_connect_permission_is_required_and_separate(world):
    client, _ = world[0]((PERMISSION, READ))
    assert _connect(client).status_code == 403
    assert client.get(DB).status_code == 403


def test_report_before_approval_is_refused(world):
    client, _ = world[0]()
    _connect(client)
    r = client.post(REPORTS, json={"question": "x", "database": "sales"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "database_not_ready"
    assert client.post(REPORTS, json={"question": "x", "database": "nope"}).status_code == 404


@pytest.mark.parametrize("tables", [
    {"ghost": {"columns": ["id"]}}, {"sales": {"columns": ["nope"]}},
    {"sales": {"columns": []}}, {"sales": {"columns": ["id"], "filterable_columns": ["amount"]}},
    {"sales": {"columns": ["id"], "key_column": "zzz"}},
    {"sales": {"columns": ["id"], "sql": "DROP"}}, {}, "sales"])
def test_invalid_approvals_are_rejected(world, tables):
    client, _ = world[0]()
    _connect(client)
    assert client.post(f"{DB}/sales/approve", json={"tables": tables}).status_code == 422
    assert client.get(DB).json()["connections"][0]["status"] == "connected_pending_approval"


def test_llm_only_sees_approved_columns(world):
    client, provider = world[0]((CONNECT, PERMISSION, READ), _plan())
    _connect(client)
    narrow = {"tables": {"sales": {"columns": ["id", "region", "amount"], "key_column": "id"}}}
    client.post(f"{DB}/sales/approve", json=narrow)
    client.post(REPORTS, json={"question": "q"})
    system = provider.last.messages[0].content
    assert '"region"' in system and '"product"' not in system


def test_two_databases_are_isolated_and_disconnect_removes_access(world):
    build, tmp = world
    _make_db(tmp / "data" / "other.sqlite3", _sample_rows(6))
    client, _ = build((CONNECT, PERMISSION, READ), _plan(), _plan())
    for name, f in (("sales", "sales.sqlite3"), ("other", "other.sqlite3")):
        _connect(client, name, f)
        client.post(f"{DB}/{name}/approve", json=APPROVE)
    assert client.post(REPORTS, json={"question": "q"}).status_code == 422  # ambiguous
    a = client.post(REPORTS, json={"question": "q", "database": "other"}).json()
    assert a["status"] == "succeeded"
    assert client.delete(f"{DB}/other").json()["status"] == "disconnected"
    assert client.post(REPORTS, json={"question": "q", "database": "other"}).status_code == 404
    assert client.delete(f"{DB}/other").status_code == 404


def test_duplicate_name_is_refused_and_registry_is_unconfigured_by_default(world):
    client, _ = world[0]()
    assert _connect(client).status_code == 200 and _connect(client).status_code == 422
    app = create_app()
    app.add_middleware(_Attach, principal=_principal((CONNECT,)))
    with TestClient(app) as c:
        assert c.get(DB).status_code == 503


def test_type_mapping():
    assert [map_sqlite_type(t) for t in ("INTEGER", "REAL", "NUMERIC(10,2)", "TEXT", "BOOLEAN",
                                         "DATE", "DATETIME", "")] == [
        "integer", "number", "number", "string", "boolean", "date", "datetime", "string"]


def test_an_active_connection_shows_what_is_approved_and_can_be_changed(world):
    client, _ = world[0]()
    _connect(client)
    assert client.get(f"{DB}/sales/schema").json()["approved"] == {}
    assert client.post(f"{DB}/sales/approve", json=APPROVE).json()["status"] == "active"
    approved = client.get(f"{DB}/sales/schema").json()["approved"]
    assert approved == {"sales": {"columns": ["id", "region", "product", "amount"],
                                  "filterable_columns": ["region", "product"]}}
    # Changing it replaces the selection: now amount may be filtered too, and the connection stays active.
    changed = {"tables": {"sales": {**APPROVE["tables"]["sales"],
                                    "filterable_columns": ["region", "product", "amount"]}}}
    assert client.post(f"{DB}/sales/approve", json=changed).json()["status"] == "active"
    assert client.get(f"{DB}/sales/schema").json()["approved"]["sales"]["filterable_columns"] == [
        "region", "product", "amount"]
