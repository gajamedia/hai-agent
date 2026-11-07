from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.llm.contracts import LLMResponse
from agent.llm.service import LLMService
from agent.workflow.table_report_planner import TableCatalogEntry, TableReportPlanner
from agent.workflow.table_report_service import TableReportService
from api.main import create_app
from api.routes.table_reports import router
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from tests.test_table_report_job_e2e import (  # noqa: F401
    COLUMNS, SALES_SPEC, _sample_rows, env,
)

URL = "/api/v1/table-reports"
PERMISSION = "report:table:run"
READ = "business:database:read"


class _Attach:
    def __init__(self, app, principal):
        self.app, self.principal = app, principal

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.principal is not None:
            scope.setdefault("state", {})["authenticated_principal"] = self.principal
        await self.app(scope, receive, send)


class _Provider:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _principal(permissions):
    return AuthenticatedPrincipal(
        actor_id="analyst-1", roles=("analyst",), permissions=tuple(permissions),
        issuer="test", subject="analyst-1", is_authenticated=True)


def _app(service, permissions=(PERMISSION, READ), authenticated=True):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.authorization_policy = AuthorizationPolicy()
    app.state.table_report_service = service
    app.add_middleware(_Attach, principal=_principal(permissions) if authenticated else None)
    return app


def _plan(**extra):
    return json.dumps({
        "supported": True, "table": "sales", "group_by": ["region"],
        "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"}],
        "title": "Revenue", **extra})


def _service(e, *answers):
    entry = TableCatalogEntry.from_table_spec(
        SALES_SPEC, COLUMNS, key_column="id", required_non_null_columns=("amount",))
    planner = TableReportPlanner(LLMService(_Provider(*answers)), [entry])
    return TableReportService(planner=planner, runner=e.runner)


def test_create_app_registers_endpoint_and_is_503_when_unconfigured():
    assert "post" in create_app().openapi()["paths"][URL]
    r = TestClient(_app(None)).post(URL, json={"question": "x"})
    assert r.status_code == 503 and r.json()["detail"]["error"] == "table_report_unavailable"


def test_unauthenticated_and_unauthorized_requests_are_rejected():
    assert TestClient(_app(None, authenticated=False)).post(URL, json={"question": "x"}).status_code == 401
    assert TestClient(_app(None, permissions=())).post(URL, json={"question": "x"}).status_code == 403


@pytest.mark.parametrize("body", [{}, {"question": 5}, {"question": "x", "table": "users"}])
def test_bad_payloads_are_422(body):
    assert TestClient(_app(None)).post(URL, json=body).status_code == 422


def test_blank_question_is_422(env):  # noqa: F811
    r = TestClient(_app(_service(env(_sample_rows(5))))).post(URL, json={"question": "   "})
    assert r.status_code == 422


def test_question_returns_report_matching_independent_sql(env):  # noqa: F811
    e = env(_sample_rows(40))
    client = TestClient(_app(_service(e, _plan(filters={"product": "widget"}))))
    r = client.post(URL, json={"question": "Penjualan widget per wilayah"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "succeeded" and body["plan"]["filters"] == {"product": "widget"}
    report = json.loads((e.artifacts_root / body["artifact"]["relative_path"]).read_text("utf-8"))
    with sqlite3.connect(e.db_path) as db:
        expected = dict(db.execute(
            "SELECT region, SUM(amount) FROM sales WHERE product='widget' GROUP BY region"))
    got = {x["region"]: x["revenue"] for x in report["aggregation"]["rows"]}
    assert got == pytest.approx(expected)
    assert {x["region"]: x["revenue"] for x in body["rows"]} == pytest.approx(expected)


def test_unsupported_question_is_200_with_reason_and_no_workflow(env):  # noqa: F811
    e = env(_sample_rows(5))
    client = TestClient(_app(_service(e, json.dumps({"supported": False, "reason": "no date filters"}))))
    body = client.post(URL, json={"question": "bulan lalu"}).json()
    assert body["status"] == "unsupported" and "date" in body["reason"]
    assert body["workflow_id"] is None


def test_no_matching_rows_is_reported_as_failed(env):  # noqa: F811
    e = env(_sample_rows(5))
    body = TestClient(_app(_service(e, _plan(filters={"region": "atlantis"})))).post(
        URL, json={"question": "x"}).json()
    assert body["status"] == "failed" and body["error_code"] == "no_matching_rows"


def test_same_question_twice_creates_two_independent_runs(env):  # noqa: F811
    e = env(_sample_rows(10))
    client = TestClient(_app(_service(e, _plan(), _plan())))
    a = client.post(URL, json={"question": "q"}).json()
    b = client.post(URL, json={"question": "q"}).json()
    assert a["status"] == b["status"] == "succeeded" and a["workflow_id"] != b["workflow_id"]


def test_llm_outage_is_503_without_leaking_details(env):  # noqa: F811
    class Down:
        name = "down"

        def complete(self, request):
            raise OSError("secret-host:1234 refused")
    e = env(_sample_rows(5))
    entry = TableCatalogEntry.from_table_spec(SALES_SPEC, COLUMNS, key_column="id")
    svc = TableReportService(planner=TableReportPlanner(LLMService(Down()), [entry]), runner=e.runner)
    r = TestClient(_app(svc)).post(URL, json={"question": "x"})
    assert r.status_code == 503 and "secret-host" not in r.text


def test_report_permission_alone_cannot_read_data(env):  # noqa: F811
    e = env(_sample_rows(10))
    client = TestClient(_app(_service(e, _plan()), permissions=(PERMISSION,)))
    body = client.post(URL, json={"question": "q"}).json()
    assert body["status"] == "failed" and body["failed_task"] == "read"
    assert body["error_code"] == "capability_authorization_denied" and body["artifact"] is None
