from __future__ import annotations

import io
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from api.database_runtime_registry import LIST_READ_LIMIT, DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authorization import AuthorizationPolicy
from tests.test_table_report_job_e2e import _make_db
from tests.test_table_report_plans_api import (  # noqa: F401
    BASE, SELECTION, _Provider, _who, world,
)
from tests.test_table_reports_api import PERMISSION, READ, _Attach


def _listing(**fields):
    body = {"supported": True, "kind": "list", "table": "sales", "columns": ["id", "region", "amount"]}
    body.update(fields)
    return json.dumps(body)


def _expected(tmp, query):
    with sqlite3.connect(tmp / "data" / "sales.sqlite3") as db:
        return [tuple(r) for r in db.execute(query)]


def test_a_listing_goes_through_plan_and_run_and_matches_sql(world):  # noqa: F811
    build, tmp = world
    client, *_ = build(_listing(order_by=[{"column": "amount", "direction": "desc"}], limit=5, format="csv"))
    c = client()
    plan = c.post(f"{BASE}/plans", json={"question": "tampilkan 5 penjualan terbesar"}).json()
    assert plan["status"] == "planned"
    summary = plan["plan"]
    assert summary["kind"] == "list" and summary["columns"] == ["id", "region", "amount"]
    assert summary["limit"] == 5 and summary["order_by"] == [{"column": "amount", "direction": "desc"}]
    assert summary["group_by"] == [] and summary["metrics"] == []

    body = c.post(f"{BASE}/plans/{plan['plan_id']}/run").json()
    assert body["status"] == "succeeded", body
    assert body["kind"] == "list" and body["row_count"] == 5 and body["limit_reached"] is True
    got = [(r["id"], r["region"], r["amount"]) for r in body["rows"]]
    assert got == _expected(tmp, "SELECT id, region, amount FROM sales ORDER BY amount DESC LIMIT 5")
    assert body["download"]["filename"] == "report.csv"


def test_the_one_call_endpoint_lists_rows_too(world):  # noqa: F811
    build, tmp = world
    client, *_ = build(_listing(filters={"product": "widget"}, order_by=[{"column": "id", "direction": "asc"}],
                                limit=4))
    body = client().post(BASE, json={"question": "tampilkan widget"}).json()
    assert body["status"] == "succeeded" and body["kind"] == "list"
    assert [(r["id"], r["region"], r["amount"]) for r in body["rows"]] == _expected(
        tmp, "SELECT id, region, amount FROM sales WHERE product='widget' ORDER BY id ASC LIMIT 4")


def test_an_aggregate_report_is_unchanged_next_to_listings(world):  # noqa: F811
    build, _ = world
    aggregate = json.dumps({"supported": True, "table": "sales", "group_by": ["region"],
                            "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"}]})
    client, *_ = build(aggregate)
    body = client().post(BASE, json={"question": "total per region"}).json()
    assert body["status"] == "succeeded" and body["kind"] == "aggregate" and body["row_count"] == 3
    assert body["limit_reached"] is False


def test_a_column_the_user_did_not_approve_cannot_be_listed(world):  # noqa: F811
    build, _ = world
    bad = _listing(columns=["id", "salary"])
    client, *_ = build(bad, bad)
    plan = client().post(f"{BASE}/plans", json={"question": "tampilkan gaji"}).json()
    assert plan["status"] == "unsupported" and plan["plan_id"] is None
    assert "Unknown columns" in plan["reason"]


def test_the_list_limit_cannot_exceed_what_the_table_allows(world):  # noqa: F811
    build, _ = world
    too_many = _listing(limit=LIST_READ_LIMIT)           # one more than the 1000 rows a listing may hold
    client, *_ = build(too_many, too_many)
    plan = client().post(f"{BASE}/plans", json={"question": "tampilkan semua"}).json()
    assert plan["status"] == "unsupported" and "between 1 and 1000" in plan["reason"]


def test_no_matching_rows_for_a_listing_is_reported_plainly(world):  # noqa: F811
    build, _ = world
    client, *_ = build(_listing(filters={"region": "atlantis"}))
    body = client().post(BASE, json={"question": "tampilkan atlantis"}).json()
    assert body["status"] == "failed" and body["error_code"] == "no_matching_rows" and body["download"] is None


def test_the_data_read_permission_is_still_required_to_list_rows(world):  # noqa: F811
    build, _ = world
    client, *_ = build(_listing())
    planner_only = client("alice", PERMISSION)
    body = planner_only.post(BASE, json={"question": "tampilkan"}).json()
    assert body["status"] == "failed" and body["error_code"] == "capability_authorization_denied"
    assert body["rows"] == []


def test_after_the_approval_is_narrowed_an_unapproved_column_cannot_be_used_to_sort(world):  # noqa: F811
    bad = _listing(columns=["id"], order_by=[{"column": "amount", "direction": "asc"}])
    build, _ = world
    client, registry, _ = build(bad, bad)
    registry.activate("sales", {"sales": {"columns": ["id", "region"], "key_column": "id"}}, actor_id="alice")
    plan = client().post(f"{BASE}/plans", json={"question": "tampilkan urut amount"}).json()
    assert plan["status"] == "unsupported" and "Unknown order_by column 'amount'" in plan["reason"]


def test_a_large_listing_shows_two_hundred_rows_but_the_file_holds_a_thousand(tmp_path):
    """The screen is capped, the report is not, and the cut is announced."""
    openpyxl = pytest.importorskip("openpyxl")
    data = tmp_path / "data"
    data.mkdir()
    _make_db(data / "sales.sqlite3", [(i, ["west", "east"][i % 2], "widget", float(i)) for i in range(1, 1501)])
    provider = _Provider(_listing(columns=["id", "amount"], order_by=[{"column": "id", "direction": "asc"}],
                                  limit=1000, format="xlsx", title="Rows of sales"))
    registry = DatabaseRuntimeRegistry(
        data_root=data, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: provider)
    registry.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
    registry.activate("sales", SELECTION, actor_id="alice")
    app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                     database_runtime_registry_factory=lambda _a: registry)
    app.add_middleware(_Attach, principal=_who("alice", PERMISSION, READ))
    try:
        with TestClient(app) as c:
            body = c.post(BASE, json={"question": "tampilkan semua penjualan"}).json()
            assert body["status"] == "succeeded", body
            assert len(body["rows"]) == 200 and body["row_count"] == 1000 and body["limit_reached"] is True
            assert body["rows"][0]["id"] == 1 and body["rows"][-1]["id"] == 200
            file = c.get(body["download"]["url"].replace("/api/v1/table-reports", BASE))
            assert file.status_code == 200
    finally:
        registry.shutdown()
    ws = openpyxl.load_workbook(io.BytesIO(file.content)).active
    assert ws.max_row == 1001                                     # header + 1000 rows
    assert [cell.value for cell in ws[1]] == ["id", "amount"] and ws.cell(row=1001, column=1).value == 1000
    assert ws.title.startswith("Rows of sales")                   # the title (with the cut note) names the sheet
