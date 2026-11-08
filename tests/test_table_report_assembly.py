from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from agent.llm.contracts import LLMResponse
from api.main import create_app
from api.security.authorization import AuthorizationPolicy
from api.table_report_assembly import (
    TableReportConfigError, build_create_app_kwargs, load_config, parse_config,
)
from tests.test_table_report_job_e2e import _make_db, _sample_rows
from tests.test_table_reports_api import PERMISSION, READ, URL, _Attach, _principal

TABLE = {"table": "sales", "description": "Sales orders", "key_column": "id",
         "columns": {"id": "integer", "region": "string", "product": "string", "amount": "number"},
         "filterable_columns": ["region", "product"], "required_non_null_columns": ["amount"]}


class _Provider:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _config_file(tmp_path, **override):
    db = tmp_path / "business.sqlite3"
    _make_db(db, _sample_rows(40))
    data = {"database_path": "business.sqlite3", "profile_name": "sales",
            "authorization_ref": "approval:test", "workflow_db": "out/workflow.sqlite3",
            "artifacts_dir": "out/artifacts", "trace_path": "out/trace.jsonl", "tables": [TABLE]}
    data.update(override)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, db


def test_relative_paths_resolve_against_config_directory(tmp_path):
    path, db = _config_file(tmp_path)
    cfg = load_config(path)
    assert cfg.database_path == db and cfg.artifacts_dir == tmp_path / "out" / "artifacts"
    assert cfg.entries[0].filterable_columns == ("region", "product")
    assert cfg.specs["sales"].filterable_columns == ("region", "product")


@pytest.mark.parametrize("change", [
    {"database_path": "missing.sqlite3"}, {"tables": []}, {"profile_name": " "},
    {"tables": [{**TABLE, "filterable_columns": ["nope"]}]},
    {"tables": [{**TABLE, "columns": {"id": "weird"}}]},
    {"tables": [TABLE, TABLE]},
])
def test_invalid_configuration_is_rejected(tmp_path, change):
    path, _ = _config_file(tmp_path)
    data = {**json.loads(path.read_text("utf-8")), **change}
    with pytest.raises(TableReportConfigError):
        parse_config(data, base_dir=tmp_path)


def test_unreadable_config_file_is_rejected(tmp_path):
    with pytest.raises(TableReportConfigError):
        load_config(tmp_path / "absent.json")


def test_assembled_app_answers_a_question_from_the_real_database(tmp_path):
    path, db = _config_file(tmp_path)
    plan = json.dumps({
        "supported": True, "table": "sales", "group_by": ["region"],
        "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"}],
        "filters": {"product": "gadget"}, "title": "Revenue"})
    kwargs = build_create_app_kwargs(load_config(path),
                                     llm_provider_factory=lambda: _Provider(plan))
    app = create_app(authorization_policy_factory=lambda _app: AuthorizationPolicy(), **kwargs)
    app.add_middleware(_Attach, principal=_principal((PERMISSION, READ)))
    with TestClient(app) as client:
        body = client.post(URL, json={"question": "Penjualan gadget per wilayah"}).json()
    assert body["status"] == "succeeded", body
    report = json.loads((tmp_path / "out" / "artifacts" / body["artifact"]["relative_path"])
                        .read_text("utf-8"))
    with sqlite3.connect(db) as conn:
        expected = dict(conn.execute(
            "SELECT region, SUM(amount) FROM sales WHERE product='gadget' GROUP BY region"))
    got = {r["region"]: r["revenue"] for r in report["aggregation"]["rows"]}
    assert got == pytest.approx(expected)


def test_default_app_has_reporting_disabled_without_config(monkeypatch):
    monkeypatch.delenv("HAI_TABLE_REPORT_CONFIG", raising=False)
    from api.main import create_app_from_environment
    app = create_app_from_environment()
    app.add_middleware(_Attach, principal=_principal((PERMISSION, READ)))
    with TestClient(app) as client:
        assert client.post(URL, json={"question": "x"}).status_code == 503


def test_config_without_auth_enabled_refuses_to_start(tmp_path, monkeypatch):
    path, _ = _config_file(tmp_path)
    monkeypatch.setenv("HAI_TABLE_REPORT_CONFIG", str(path))
    monkeypatch.delenv("HAI_AUTH_ENABLED", raising=False)
    from api.main import create_app_from_environment
    with pytest.raises(RuntimeError, match="HAI_AUTH_ENABLED"):
        create_app_from_environment()
