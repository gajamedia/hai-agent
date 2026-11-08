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
    ({"table": "sales", "group_by": ["region"], "format": "docx",
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
    assert "CONDITIONS" in system and "30 ribu" in system and "Today is 2026-" in system


def _with(**extra):
    base = json.loads(GOOD)
    base.update(extra)
    return json.dumps(base)


def _cond(column, operator, value=None):
    c = {"column": column, "operator": operator}
    if value is not None:
        c["value"] = value
    return _with(conditions=[c])


def test_conditions_become_request_conditions():
    planner, _ = _planner(_with(conditions=[
        {"column": "region", "operator": "in", "value": ["east", "west"]},
        {"column": "region", "operator": "is_not_null"}]))
    req = planner.plan("x").request
    assert req.conditions == (("region", "in", ("east", "west")), ("region", "is_not_null", None))


@pytest.mark.parametrize("answer,needle", [
    (_cond("amount", "lt", 5), "Cannot filter on"),            # not a filterable column
    (_cond("region", "lt", "a"), "cannot be used"),            # text is not ordered
    (_cond("region", "like", "a"), "not allowed"),
    (_cond("region", "contains", 5), "text"),
    (_cond("region", "in", []), "'in' needs"),
    (_cond("region", "is_null", "x"), "takes no value"),
    (_with(conditions="x"), "list"),
    (_with(conditions=[{"column": "region", "operator": "eq", "value": "a", "sql": "1"}]), "Each condition"),
    (_with(conditions=[{"column": "region", "operator": "eq", "value": "a"}] * 6), "at most"),
])
def test_bad_conditions_are_rejected(answer, needle):
    planner, _ = _planner(answer, answer)
    result = planner.plan("x")
    assert not result.ok and needle in result.errors[-1]


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


# ------------------------------------------------------------------ kind "list"
def _list_answer(**fields):
    body = {"supported": True, "kind": "list", "table": "sales", "columns": ["id", "region", "amount"]}
    body.update(fields)
    return json.dumps(body)


def test_a_list_plan_becomes_a_list_request_with_catalog_owned_limits():
    planner, _ = _planner(_list_answer(order_by=[{"column": "amount", "direction": "DESC"}], limit=25,
                                       filters={"region": "west"}, format="pdf", title="Top sales"))
    result = planner.plan("tampilkan 25 penjualan terbesar di west")
    assert result.ok and result.attempts == 1
    req = result.request
    assert req.kind == "list" and req.select_columns == ("id", "region", "amount")
    assert req.order_by == (("amount", "desc"), ) and req.limit == 25      # direction is normalized
    assert dict(req.filters) == {"region": "west"} and req.report_format == "pdf"
    assert req.row_limit == 500 and req.group_by == () and req.metrics == ()
    assert req.columns["amount"] == "number"                               # types come from the catalog


def test_the_limit_defaults_to_one_hundred_and_all_columns_can_be_asked_for():
    planner, _ = _planner(_list_answer(columns="all"))
    req = planner.plan("tampilkan data").request
    assert req.select_columns == tuple(COLUMNS) and req.limit == 100 and req.order_by == ()
    planner, _ = _planner(_list_answer(columns=["*"]))
    assert planner.plan("x").request.select_columns == tuple(COLUMNS)


def test_a_missing_kind_still_means_an_aggregate_report():
    planner, _ = _planner(GOOD)                         # the answer from before lists existed
    req = planner.plan("x").request
    assert req.kind == "aggregate" and req.group_by == ("region",)


@pytest.mark.parametrize("fields,needle", [
    ({"columns": ["nope"]}, "Unknown columns"), ({"columns": []}, "columns must be"),
    ({"columns": ["id", "id"]}, "distinct"), ({"columns": [5]}, "columns must be"),
    ({"limit": 0}, "limit must be"), ({"limit": 500}, "between 1 and 499"),
    ({"limit": True}, "limit must be"), ({"limit": "10"}, "limit must be"), ({"limit": 2.5}, "limit must be"),
    ({"order_by": "amount"}, "list of at most"),
    ({"order_by": [{"column": "nope"}]}, "Unknown order_by"),
    ({"order_by": [{"column": "amount", "direction": "up"}]}, "asc or desc"),
    ({"order_by": ["amount"]}, "Each order_by item"),
    ({"order_by": [{"column": "id"}, {"column": "id", "direction": "desc"}]}, "twice"),
    ({"order_by": [{"column": c} for c in ("id", "region", "product", "amount")]}, "at most"),
    ({"filters": {"amount": 1}}, "Cannot filter"),
    ({"format": "docx"}, "format must be"),
])
def test_invalid_list_plans_are_rejected_with_a_specific_reason(fields, needle):
    answer = _list_answer(**fields)
    planner, _ = _planner(answer, answer)
    result = planner.plan("x")
    assert not result.ok and needle in result.errors[-1]


def test_an_unknown_kind_is_rejected():
    answer = json.dumps({"supported": True, "kind": "pivot", "table": "sales"})
    planner, _ = _planner(answer, answer)
    assert "kind must be" in planner.plan("x").errors[-1]


def test_a_list_plan_cannot_smuggle_sql():
    answer = _list_answer(columns=["id; DROP TABLE sales"], order_by=[{"column": "amount; --"}])
    planner, _ = _planner(answer, answer)
    result = planner.plan("x")
    assert not result.ok and "Unknown columns" in result.errors[-1]


def test_the_invalid_list_answer_is_retried_once_with_the_exact_error():
    planner, provider = _planner(_list_answer(limit=9999), _list_answer(limit=10))
    result = planner.plan("x")
    assert result.ok and result.attempts == 2 and result.request.limit == 10
    assert "between 1 and 499" in provider.requests[1].messages[-1].content


def test_the_prompt_teaches_both_kinds_and_states_the_row_limit_but_never_asks_for_sql():
    planner, provider = _planner(GOOD)
    planner.plan("x")
    system = provider.requests[0].messages[0].content
    assert 'kind "list"' in system and "tampilkan" in system and '"list_limit_max": 499' in system
    assert "order_by" in system and 'kind "aggregate"' in system
    assert "you never write SQL" in system


def test_list_and_aggregate_requests_share_the_catalog_columns_but_not_the_rules():
    planner, _ = _planner(_list_answer(columns=["region"]))
    req = planner.plan("x").request
    assert req.required_non_null_columns == () and req.unique_key_columns == ()    # a listing shows data as it is
