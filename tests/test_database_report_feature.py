import json

import pytest

from agent.platform import Effect, OperationRegistry
from agent.workflow.table_report_job import TableReportRequest
from api.features.database_reports import (
    OutcomeCache, build_report_feature, plan_result_from_task_plan, request_from_json, request_to_json,
)

COLUMNS = {"id": "integer", "region": "string", "amount": "number", "day": "date"}


def aggregate(**over):
    base = dict(table="sales", columns=COLUMNS, group_by=("region",),
                metrics=(("revenue", "sum", "amount"), ("orders", "count", "id")), title="Report: sales",
                report_format="xlsx", row_limit=500, required_non_null_columns=("amount", "id"),
                unique_key_columns=("id",), numeric_columns=("amount",), categorical_columns=("region",),
                filters={"region": "west"},
                conditions=(("amount", "between", (10, 20)), ("region", "in", ("west", "east")),
                            ("amount", "gt", 5), ("day", "is_null", None)))
    base.update(over)
    return TableReportRequest(**base)


def list_request(**over):
    base = dict(table="sales", columns=COLUMNS, group_by=(), metrics=(), kind="list",
                select_columns=("id", "region"), order_by=(("amount", "desc"), ("id", "asc")), limit=25)
    base.update(over)
    return TableReportRequest(**base)


def join_request():
    return TableReportRequest(
        table="sales", columns={"sales.region": "string", "stores.name": "string", "sales.amount": "number"},
        group_by=("stores.name",), metrics=(("revenue", "sum", "sales.amount"),),
        joins=(("stores", "left", "sales.store_id", "stores.id"),))


@pytest.mark.parametrize("request_", [aggregate(), list_request(), join_request(),
                                      aggregate(filters={}, conditions=())])
def test_a_plan_survives_json_unchanged(request_):
    as_json = request_to_json(request_)
    text = json.dumps(as_json)                       # must be plain JSON
    assert request_from_json(json.loads(text)) == request_


def test_between_and_in_values_come_back_as_tuples_like_the_planner_makes_them():
    back = request_from_json(json.loads(json.dumps(request_to_json(aggregate()))))
    values = {c: v for c, op, v in back.conditions if op in {"between", "in"}}
    assert values == {"amount": (10, 20), "region": ("west", "east")}


def test_a_stored_plan_is_revalidated_when_it_is_read_back():
    data = request_to_json(aggregate())
    data["row_limit"] = 1                            # the request's own rules still apply
    with pytest.raises(ValueError):
        request_from_json(data)
    data = request_to_json(aggregate())
    data["kind"] = "delete"
    with pytest.raises(ValueError):
        request_from_json(data)


def test_plan_result_is_rebuilt_with_its_proposed_relations():
    plan = {"question": "q", "request": request_to_json(aggregate()), "attempts": 2, "provider": "p",
            "proposed_relations": [{"table": "sales", "column": "store_id", "ref_table": "stores",
                                    "ref_column": "id", "source": "ai"}]}
    result = plan_result_from_task_plan(plan)
    assert result.ok and result.attempts == 2 and result.provider == "p"
    assert result.proposed_relations[0].source == "ai"


def test_the_report_operation_is_a_read_with_no_approval_and_a_verifier():
    registry = OperationRegistry()
    registry.register(build_report_feature(lambda database: (None, None, None), OutcomeCache()))
    op = registry.get("database.report")
    assert op.effect is Effect.READ and not op.approval.required
    assert op.permission == "report:table:run" and op.verify is not None


def test_the_cache_is_bounded():
    cache = OutcomeCache(max_items=2)
    for i in range(5):
        cache.put_body(str(i), {"i": i})
    assert cache.body("0") is None and cache.body("4") == {"i": 4}
