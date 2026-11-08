"""End-to-end: approved SQLite table -> validated analytics -> persisted report.

Uses real components only (SQLite file, read-only gateway, authorization policy,
execution security policy, workflow store, artifact store). The expected
numbers come from an independent ``GROUP BY`` run directly against SQLite.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from types import SimpleNamespace

import pytest

from agent.llm.business_database_adapter import ReadOnlyTableSpec
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.trusted_capability_invocation_port import (
    CapabilityInvocationSpec,
    TrustedCapabilityInvocationPort,
)
from agent.workflow import SQLiteWorkflowStore
from agent.workflow.analytics import WorkflowAnalyticalTaskExecutor
from agent.workflow.invocation import WorkflowCapabilityTaskExecutor
from agent.workflow.reports import LocalArtifactStore, WorkflowReportTaskExecutor
from agent.workflow.service import WorkflowApplicationService
from agent.workflow.table_report_job import (
    READ_ACTION,
    TableReportJobRunner,
    TableReportRequest,
)
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from api.security.trusted_execution_context import TrustedExecutionContext
from api.sqlite_database_gateway_factory import create_sqlite_readonly_gateway_factory

COLUMNS = {"id": "integer", "region": "string", "product": "string", "amount": "number"}
SALES_SPEC = ReadOnlyTableSpec(
    table="sales", columns=tuple(COLUMNS),
    filterable_columns=("region", "product"), max_limit=500,
)


class _App:
    state = type("State", (), {"authorization_policy": AuthorizationPolicy()})()


def _context(permissions=("business:database:read",)):
    return TrustedExecutionContext(AuthenticatedPrincipal(
        actor_id="analyst-1", roles=("analyst",), permissions=permissions,
        issuer="test", subject="analyst-1", is_authenticated=True,
    ))


def _make_db(path, rows):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, "
                   "product TEXT, amount REAL)")
        db.executemany("INSERT INTO sales VALUES (?,?,?,?)", rows)


def _sample_rows(n=40):
    regions = ["west", "east", "north"]
    return [(i, regions[i % 3], "widget" if i % 2 else "gadget", 10.0 + i) for i in range(1, n + 1)]


@pytest.fixture
def env(tmp_path):
    def build(rows):
        db_path = tmp_path / "sales.sqlite3"
        _make_db(db_path, rows)
        bundle = create_sqlite_readonly_gateway_factory(
            database_path=db_path, profile_name="sales", authorization_ref="approval:test-001",
            table_specs={"sales": SALES_SPEC},
            trace_path=tmp_path / "trace.jsonl", fsync_trace=False,
        )(_App())
        port = TrustedCapabilityInvocationPort(
            gateway=bundle.gateway,
            specs={READ_ACTION: CapabilityInvocationSpec(
                action_name=READ_ACTION, target="business_database",
                capability_id="business_database.select")},
        )
        agent = SimpleNamespace(
            actions={}, unified_executor=object(),
            execution_security_policy=ExecutionSecurityPolicy(action_risks={READ_ACTION: 0.1}),
        )
        store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        runner = TableReportJobRunner(
            service=WorkflowApplicationService(store=store), store=store,
            capability_executor=WorkflowCapabilityTaskExecutor(
                store=store, agent=agent, trusted_capability_invocation_port=port),
            analytical_executor=WorkflowAnalyticalTaskExecutor(store=store),
            report_executor=WorkflowReportTaskExecutor(store=store, artifact_store=artifacts),
        )
        return SimpleNamespace(runner=runner, bundle=bundle, db_path=db_path,
                               artifacts_root=tmp_path / "artifacts")
    created = []

    def factory(rows):
        e = build(rows)
        created.append(e)
        return e
    yield factory
    for e in created:
        for resource in e.bundle.owned_resources:
            resource.shutdown()


def _request(**overrides):
    base = dict(
        table="sales", columns=COLUMNS, group_by=("region",),
        metrics=(("revenue", "sum", "amount"), ("orders", "count", "id")),
        title="Sales by region", required_non_null_columns=("id", "amount"),
        unique_key_columns=("id",), numeric_columns=("amount",),
        categorical_columns=("region",),
    )
    base.update(overrides)
    return TableReportRequest(**base)


def _run(e, request, key, context=None):
    return e.runner.run(goal="Report sales by region", idempotency_key=key,
                        request=request, execution_context=context or _context())


def _expected(db_path):
    with sqlite3.connect(db_path) as db:
        return {r: (rev, n) for r, rev, n in db.execute(
            "SELECT region, SUM(amount), COUNT(id) FROM sales GROUP BY region")}


def test_json_report_matches_independent_group_by(env):
    e = env(_sample_rows(40))
    outcome = _run(e, _request(), "job-json")
    assert outcome.succeeded, (outcome.failed_task_key, outcome.error_code, outcome.message)
    assert set(outcome.task_statuses.values()) == {"succeeded"}

    report = json.loads((e.artifacts_root / outcome.artifact["relative_path"]).read_text("utf-8"))
    got = {r["region"]: (r["revenue"], r["orders"]) for r in report["aggregation"]["rows"]}
    expected = _expected(e.db_path)
    assert got.keys() == expected.keys()
    for region, (revenue, orders) in expected.items():
        assert got[region][0] == pytest.approx(revenue)
        assert got[region][1] == orders
    assert report["source"]["row_count"] == 40
    assert report["data_quality"]["valid"] is True


def test_csv_report_is_produced(env):
    e = env(_sample_rows(12))
    outcome = _run(e, _request(report_format="csv"), "job-csv")
    assert outcome.succeeded, (outcome.failed_task_key, outcome.error_code)
    text = (e.artifacts_root / outcome.artifact["relative_path"]).read_text("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    assert {r["region"] for r in rows} == {"west", "east", "north"}


def test_table_reaching_row_cap_fails_instead_of_reporting_partial_totals(env):
    e = env(_sample_rows(40))
    outcome = _run(e, _request(row_limit=10), "job-truncated")
    assert not outcome.succeeded
    assert outcome.failed_task_key == "analyze"
    assert outcome.artifact is None
    assert outcome.error_code == "dataset_validation_failed"
    assert outcome.task_statuses["report"] != "succeeded"


def test_null_in_required_column_fails_validation_and_produces_no_report(env):
    e = env(_sample_rows(10))
    with sqlite3.connect(e.db_path) as db:
        db.execute("UPDATE sales SET amount = NULL WHERE id = 3")
    outcome = _run(e, _request(), "job-null")
    assert not outcome.succeeded
    assert outcome.failed_task_key == "analyze"
    assert outcome.error_code == "dataset_validation_failed"
    assert outcome.artifact is None


def test_principal_without_permission_cannot_read_and_no_report_is_written(env):
    e = env(_sample_rows(10))
    outcome = _run(e, _request(), "job-denied", context=_context(permissions=()))
    assert not outcome.succeeded
    assert outcome.failed_task_key == "read"
    assert outcome.artifact is None
    assert outcome.error_code == "capability_authorization_denied"
    assert not e.artifacts_root.exists() or not list(e.artifacts_root.rglob("*report.*"))


def test_request_rejects_undeclared_columns():
    with pytest.raises(ValueError):
        _request(group_by=("missing_column",))


# ----------------------------------------------------------------- filters
def test_filtered_report_matches_independent_where_query_and_shows_scope(env):
    e = env(_sample_rows(40))
    outcome = _run(e, _request(filters={"product": "widget"}), "job-filter")
    assert outcome.succeeded, (outcome.failed_task_key, outcome.error_code, outcome.message)
    report = json.loads((e.artifacts_root / outcome.artifact["relative_path"]).read_text("utf-8"))
    assert "[filter: product=widget]" in report["title"]
    with sqlite3.connect(e.db_path) as db:
        expected = {r: (rev, n) for r, rev, n in db.execute(
            "SELECT region, SUM(amount), COUNT(id) FROM sales "
            "WHERE product = 'widget' GROUP BY region")}
    got = {r["region"]: (r["revenue"], r["orders"]) for r in report["aggregation"]["rows"]}
    assert got.keys() == expected.keys()
    for region, (revenue, orders) in expected.items():
        assert got[region][0] == pytest.approx(revenue) and got[region][1] == orders
    assert report["source"]["row_count"] == 20  # half the rows, not the whole table


def test_two_filters_combine_with_and(env):
    e = env(_sample_rows(40))
    outcome = _run(e, _request(filters={"product": "widget", "region": "west"}), "job-2f")
    assert outcome.succeeded, (outcome.error_code, outcome.message)
    report = json.loads((e.artifacts_root / outcome.artifact["relative_path"]).read_text("utf-8"))
    assert [r["region"] for r in report["aggregation"]["rows"]] == ["west"]
    with sqlite3.connect(e.db_path) as db:
        (n,) = db.execute("SELECT COUNT(*) FROM sales WHERE product='widget' AND region='west'").fetchone()
    assert report["source"]["row_count"] == n


def test_filter_matching_nothing_is_reported_clearly_without_a_report(env):
    e = env(_sample_rows(10))
    outcome = _run(e, _request(filters={"region": "atlantis"}), "job-empty")
    assert not outcome.succeeded and outcome.artifact is None
    assert outcome.failed_task_key == "analyze"
    assert outcome.error_code == "no_matching_rows"


def test_sql_injection_text_in_filter_value_is_just_a_value(env):
    e = env(_sample_rows(10))
    outcome = _run(e, _request(filters={"region": "west' OR '1'='1"}), "job-inject")
    assert not outcome.succeeded and outcome.error_code == "no_matching_rows"
    with sqlite3.connect(e.db_path) as db:  # table untouched and no row leaked
        assert db.execute("SELECT COUNT(*) FROM sales").fetchone() == (10,)


def test_filter_on_non_filterable_column_is_refused_by_the_adapter(env):
    e = env(_sample_rows(10))
    outcome = _run(e, _request(filters={"amount": 12.0}), "job-badcol")
    assert not outcome.succeeded and outcome.failed_task_key == "read"
    assert outcome.artifact is None


def test_row_cap_guard_still_applies_to_filtered_reads(env):
    e = env(_sample_rows(40))
    outcome = _run(e, _request(filters={"product": "widget"}, row_limit=10), "job-fcap")
    assert not outcome.succeeded and outcome.failed_task_key == "analyze"
    assert outcome.error_code == "dataset_validation_failed"


def test_request_rejects_bad_filter_shapes():
    with pytest.raises(ValueError):
        _request(filters={"a": [1, 2]})
    with pytest.raises(ValueError):
        _request(filters={"a": None})
    with pytest.raises(ValueError):
        _request(filters={"a": 1, "b": 2, "c": 3, "d": 4})
    with pytest.raises(ValueError):
        _request(filters={"a": "x" * 201})
