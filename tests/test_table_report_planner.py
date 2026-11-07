from __future__ import annotations

import json
import os
import sqlite3

import pytest

from agent.llm.contracts import LLMResponse
from agent.llm.service import LLMService
from agent.workflow.table_report_planner import TableCatalogEntry, TableReportPlanner
# Reuse the real-component fixture and helpers from the end-to-end job tests.
from tests.test_table_report_job_e2e import (  # noqa: F401
    COLUMNS, SALES_SPEC, _context, _expected, _sample_rows, env,
)

CATALOG = [TableCatalogEntry(
    table="sales", columns=COLUMNS, description="Sales orders", key_column="id",
    required_non_null_columns=("amount",), row_limit=500,
    filterable_columns=("region", "product"))]


class ScriptedProvider:
    """Returns canned answers in order; records the prompts it received."""
    name = "scripted"

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _planner(*answers, **kw):
    provider = ScriptedProvider(*answers)
    return TableReportPlanner(LLMService(provider), CATALOG, **kw), provider


GOOD = json.dumps({
    "supported": True, "table": "sales", "group_by": ["region"],
    "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"},
                {"output": "orders", "operation": "count"}],
    "title": "Revenue by region", "format": "json"})


def test_valid_answer_becomes_request_with_catalog_owned_rules():
    planner, _ = _planner(GOOD)
    result = planner.plan("Total penjualan per wilayah?")
    assert result.ok and result.attempts == 1
    req = result.request
    assert req.table == "sales" and req.group_by == ("region",)
    assert req.metrics == (("revenue", "sum", "amount"), ("orders", "count", "id"))
    assert req.unique_key_columns == ("id",)            # from catalog, not the LLM
    assert set(req.required_non_null_columns) == {"amount", "id"}
    assert req.row_limit == 500                          # from catalog
    assert req.columns["amount"] == "number"            # type from catalog


def test_markdown_fenced_json_is_accepted():
    planner, _ = _planner(f"Here you go:\n```json\n{GOOD}\n```")
    assert planner.plan("x").ok


def test_invalid_answer_is_retried_once_with_the_exact_error():
    bad = json.dumps({"supported": True, "table": "sales", "group_by": ["nope"],
                      "metrics": [{"output": "n", "operation": "count"}]})
    planner, provider = _planner(bad, GOOD)
    result = planner.plan("x")
    assert result.ok and result.attempts == 2 and len(result.errors) == 1
    retry_prompt = provider.requests[1].messages[-1].content
    assert "Unknown group_by columns ['nope']" in retry_prompt


@pytest.mark.parametrize("answer,needle", [
    ({"table": "users; DROP TABLE sales", "group_by": ["region"],
      "metrics": [{"output": "n", "operation": "count"}]}, "Unknown table"),
    ({"table": "sales", "group_by": ["region"],
      "metrics": [{"output": "t", "operation": "sum", "column": "product"}]}, "numeric"),
    ({"table": "sales", "group_by": ["region"],
      "metrics": [{"output": "t", "operation": "median", "column": "amount"}]}, "not allowed"),
    ({"table": "sales", "group_by": ["region"],
      "metrics": [{"output": "Bad Name", "operation": "count"}]}, "snake_case"),
    ({"table": "sales", "group_by": ["region"],
      "metrics": [{"output": "region", "operation": "count"}]}, "Duplicate output"),
    ({"table": "sales", "group_by": ["region"], "format": "pdf",
      "metrics": [{"output": "n", "operation": "count"}]}, "format"),
])
def test_hallucinated_or_unsafe_plans_are_rejected(answer, needle):
    answer = {"supported": True, **answer}
    planner, _ = _planner(json.dumps(answer), json.dumps(answer))
    result = planner.plan("x")
    assert not result.ok and result.request is None
    assert needle in result.errors[-1]


def test_unsupported_question_is_reported_not_forced():
    planner, provider = _planner(json.dumps(
        {"supported": False, "reason": "Perlu filter tanggal, belum didukung."}))
    result = planner.plan("Penjualan bulan lalu per wilayah")
    assert not result.ok and "filter tanggal" in result.reason
    assert len(provider.requests) == 1  # no pointless retry


def test_garbage_output_fails_cleanly_after_max_attempts():
    planner, _ = _planner("I think you want sales.", "still prose")
    result = planner.plan("x")
    assert not result.ok and result.attempts == 2 and "valid plan" in result.reason


def test_prompt_contains_catalog_but_no_secrets_or_rules():
    planner, provider = _planner(GOOD)
    planner.plan("x")
    system = provider.requests[0].messages[0].content
    assert '"sales"' in system and '"region"' in system and "NOT supported" in system


def test_catalog_validation():
    with pytest.raises(ValueError):
        TableCatalogEntry(table="t", columns={"a": "weird"})
    with pytest.raises(ValueError):
        TableCatalogEntry(table="t", columns={"a": "string"}, key_column="zzz")
    with pytest.raises(ValueError):
        TableReportPlanner(LLMService(ScriptedProvider()), [])


def test_question_to_report_end_to_end(env):  # noqa: F811
    """Natural-language question -> planner -> real job -> report == independent GROUP BY."""
    e = env(_sample_rows(40))
    planner, _ = _planner(GOOD)
    plan = planner.plan("Berapa total penjualan dan jumlah order per wilayah?")
    assert plan.ok
    outcome = e.runner.run(goal=plan.question, idempotency_key="nl-job-1",
                           request=plan.request, execution_context=_context())
    assert outcome.succeeded, (outcome.failed_task_key, outcome.error_code, outcome.message)
    report = json.loads((e.artifacts_root / outcome.artifact["relative_path"]).read_text("utf-8"))
    got = {r["region"]: (r["revenue"], r["orders"]) for r in report["aggregation"]["rows"]}
    for region, (revenue, orders) in _expected(e.db_path).items():
        assert got[region][0] == pytest.approx(revenue) and got[region][1] == orders


@pytest.mark.skipif(os.getenv("HAI_RUN_LIVE_LLM_TESTS") != "1",
                    reason="live Claude call; set HAI_RUN_LIVE_LLM_TESTS=1 to run")
def test_live_claude_plans_and_report_matches_database(env):  # noqa: F811
    from agent.llm.claude_code_provider import ClaudeCodeProvider
    e = env(_sample_rows(40))
    planner = TableReportPlanner(LLMService(ClaudeCodeProvider(timeout=180)), CATALOG)
    plan = planner.plan("Berapa total penjualan (amount) per wilayah? Tolong sertakan jumlah order.")
    assert plan.ok, (plan.reason, plan.errors)
    assert plan.provider == "claude-code"
    outcome = e.runner.run(goal=plan.question, idempotency_key="live-1",
                           request=plan.request, execution_context=_context())
    assert outcome.succeeded, (outcome.failed_task_key, outcome.error_code)
    report = json.loads((e.artifacts_root / outcome.artifact["relative_path"]).read_text("utf-8"))
    revenue_col = plan.request.metrics[0][0]
    got = {r["region"]: r[revenue_col] for r in report["aggregation"]["rows"]}
    for region, (revenue, _n) in _expected(e.db_path).items():
        assert got[region] == pytest.approx(revenue)


# ----------------------------------------------------------------- filters
def _filtered(filters, **extra):
    return json.dumps({
        "supported": True, "table": "sales", "group_by": ["region"],
        "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"}],
        "filters": filters, **extra})


def test_valid_equality_filter_is_carried_into_the_request():
    planner, _ = _planner(_filtered({"product": "widget"}))
    result = planner.plan("Total penjualan widget per wilayah")
    assert result.ok and dict(result.request.filters) == {"product": "widget"}


def test_no_filters_means_empty_filters():
    planner, _ = _planner(GOOD)
    assert dict(planner.plan("x").request.filters) == {}


@pytest.mark.parametrize("filters,needle", [
    ({"amount": 10}, "Cannot filter"),                       # not filterable
    ({"nonexistent": "x"}, "Cannot filter"),                 # hallucinated column
    ({"region": 5}, "must match its type"),                  # wrong type
    ({"region": True}, "must match its type"),
    ({"region": "  "}, "empty or too long"),
    ({"region": "a", "product": "b", "id": 1, "amount": 2}, "at most"),
    ("region=west", "filters must be an object"),
])
def test_invalid_filters_are_rejected_with_a_specific_reason(filters, needle):
    answer = _filtered(filters)
    planner, _ = _planner(answer, answer)
    result = planner.plan("x")
    assert not result.ok and needle in result.errors[-1]


def test_prompt_lists_filterable_columns_and_the_range_limitation():
    planner, provider = _planner(GOOD)
    planner.plan("x")
    system = provider.requests[0].messages[0].content
    assert '"filterable_columns": ["region", "product"]' in system
    assert "EQUALITY" in system and "date ranges" in system


def test_catalog_derived_from_table_spec_cannot_drift():
    entry = TableCatalogEntry.from_table_spec(SALES_SPEC, COLUMNS, key_column="id")
    assert entry.filterable_columns == SALES_SPEC.filterable_columns
    assert entry.row_limit == SALES_SPEC.max_limit
    assert list(entry.columns) == list(SALES_SPEC.columns)
    with pytest.raises(ValueError):
        TableCatalogEntry.from_table_spec(SALES_SPEC, {"id": "integer"})


def test_filtered_question_to_report_end_to_end(env):  # noqa: F811
    e = env(_sample_rows(40))
    entry = TableCatalogEntry.from_table_spec(
        SALES_SPEC, COLUMNS, key_column="id", required_non_null_columns=("amount",))
    provider = ScriptedProvider(_filtered({"region": "west"}, title="Revenue"))
    plan = TableReportPlanner(LLMService(provider), [entry]).plan("Penjualan wilayah west per wilayah")
    assert plan.ok
    outcome = e.runner.run(goal=plan.question, idempotency_key="nl-filter",
                           request=plan.request, execution_context=_context())
    assert outcome.succeeded, (outcome.failed_task_key, outcome.error_code)
    report = json.loads((e.artifacts_root / outcome.artifact["relative_path"]).read_text("utf-8"))
    with sqlite3.connect(e.db_path) as db:
        (revenue,) = db.execute("SELECT SUM(amount) FROM sales WHERE region='west'").fetchone()
    assert report["aggregation"]["rows"][0]["revenue"] == pytest.approx(revenue)
    assert "[filter: region=west]" in report["title"]
