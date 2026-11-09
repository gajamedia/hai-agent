"""The Database Agent's report operation (``database.report``) on the shared platform.

What moved here from the report routes: the plan is now a JSON document stored in the platform
task store (so it survives a restart), the run goes through ``PlatformPipeline`` (authorize, execute
once, verify, audit), and the plan remembers a *token* of the approvals it was made under.

What did not move: the planner, the job runner and the read-only gateway are exactly the ones the
routes always used, and report **rows are never written to the task store**. A task keeps only
metadata (status, workflow id, file info, counts); the rows live in the report file and, for an
idempotent replay in the same process, in a small in-memory cache.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import fields
from typing import Any, Callable, Mapping, Optional

from fastapi.encoders import jsonable_encoder

from agent.platform import (
    ApprovalPolicy, Effect, Feature, Operation, PlanUnsupported, Verdict, Verification,
)
from agent.workflow.table_report_job import TableReportRequest
from agent.workflow.table_report_planner import TablePlanResult, TableRelation
from agent.workflow.table_report_service import AskOutcome, TableReportService
from api.security.trusted_execution_context import TrustedExecutionContext

FEATURE = "database"
REPORT_OPERATION = "database.report"
REPORT_PERMISSION = "report:table:run"

# fields of TableReportRequest that hold tuples (so JSON lists must be turned back into tuples)
_TUPLE_FIELDS = ("group_by", "required_non_null_columns", "unique_key_columns", "numeric_columns",
                 "categorical_columns", "select_columns")
_TUPLE_OF_TUPLE_FIELDS = ("metrics", "order_by", "joins")


class ConnectionChanged(ValueError):
    """The approved tables of the connection changed after the plan was made."""


# --------------------------------------------------------------------------- plan <-> JSON

def request_to_json(request: TableReportRequest) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(request):
        value = getattr(request, f.name)
        if f.name == "conditions":
            value = [[c, op, list(v) if isinstance(v, (tuple, list)) else v] for c, op, v in value]
        elif f.name in _TUPLE_FIELDS:
            value = list(value)
        elif f.name in _TUPLE_OF_TUPLE_FIELDS:
            value = [list(item) for item in value]
        elif isinstance(value, Mapping):
            value = dict(value)
        out[f.name] = value
    return out


def request_from_json(data: Mapping[str, Any]) -> TableReportRequest:
    values = dict(data)
    for name in _TUPLE_FIELDS:
        if name in values:
            values[name] = tuple(values[name])
    for name in _TUPLE_OF_TUPLE_FIELDS:
        if name in values:
            values[name] = tuple(tuple(item) for item in values[name])
    if "conditions" in values:
        values["conditions"] = tuple(
            (c, op, tuple(v) if isinstance(v, list) else v) for c, op, v in values["conditions"])
    if "columns" in values:
        values["columns"] = dict(values["columns"])
    if "filters" in values:
        values["filters"] = dict(values["filters"])
    return TableReportRequest(**values)  # its own __post_init__ validates the shape again


def _relations_to_json(relations) -> list[dict[str, str]]:
    return [{"table": r.table, "column": r.column, "ref_table": r.ref_table,
             "ref_column": r.ref_column, "source": r.source} for r in relations]


def _relations_from_json(items) -> tuple[TableRelation, ...]:
    return tuple(TableRelation(**item) for item in items)


def plan_result_from_task_plan(plan: Mapping[str, Any]) -> TablePlanResult:
    return TablePlanResult(
        plan["question"], request_from_json(plan["request"]), None, int(plan.get("attempts", 0)),
        plan.get("provider"), (), proposed_relations=_relations_from_json(plan["proposed_relations"]))


# --------------------------------------------------------------------------- in-memory cache

class OutcomeCache:
    """Full outcomes (with rows) and response bodies for the replay window. In memory only."""

    def __init__(self, max_items: int = 200) -> None:
        self._max = max_items
        self._lock = threading.Lock()
        self._outcomes: "OrderedDict[str, AskOutcome]" = OrderedDict()
        self._bodies: "OrderedDict[str, dict]" = OrderedDict()

    def _put(self, store: OrderedDict, key: str, value: Any) -> None:
        with self._lock:
            store[key] = value
            store.move_to_end(key)
            while len(store) > self._max:
                store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._outcomes.clear()
            self._bodies.clear()

    def put_outcome(self, task_id: str, outcome: AskOutcome) -> None:
        self._put(self._outcomes, task_id, outcome)

    def outcome(self, task_id: str) -> Optional[AskOutcome]:
        with self._lock:
            return self._outcomes.get(task_id)

    def put_body(self, task_id: str, body: dict) -> None:
        self._put(self._bodies, task_id, body)

    def body(self, task_id: str) -> Optional[dict]:
        with self._lock:
            return self._bodies.get(task_id)


def outcome_metadata(outcome: AskOutcome) -> dict[str, Any]:
    """What the task store keeps of an outcome: everything except the rows."""
    return jsonable_encoder({
        "status": outcome.status, "reason": outcome.reason, "workflow_id": outcome.workflow_id,
        "error_code": outcome.error_code, "failed_task": outcome.failed_task_key,
        "artifact": outcome.artifact, "provider": outcome.provider, "attempts": outcome.attempts,
        "kind": outcome.kind, "row_count": outcome.row_count, "limit_reached": outcome.limit_reached,
    })


def outcome_from_metadata(plan: Mapping[str, Any], meta: Mapping[str, Any]) -> AskOutcome:
    """Rebuild an outcome after a restart: same facts, no rows (they are in the report file)."""
    return AskOutcome(
        meta["status"], plan["question"], reason=meta.get("reason"), workflow_id=meta.get("workflow_id"),
        error_code=meta.get("error_code"), failed_task_key=meta.get("failed_task"),
        artifact=meta.get("artifact"), plan=plan.get("summary"), provider=meta.get("provider"),
        attempts=int(meta.get("attempts") or 0), rows=(), kind=meta.get("kind") or "aggregate",
        row_count=meta.get("row_count"), limit_reached=bool(meta.get("limit_reached")))


# --------------------------------------------------------------------------- the feature

# resolve(database) -> (service, exact connection name or None, approvals token or None)
Resolver = Callable[[Optional[str]], tuple[TableReportService, Optional[str], Optional[str]]]


def build_report_operation(resolve: Resolver, cache: OutcomeCache) -> Operation:
    def plan(request: Mapping[str, Any], ctx) -> dict[str, Any]:
        service, name, token = resolve(request.get("database"))
        result = service.plan(request["question"], request.get("format"))
        if not result.ok:
            raise PlanUnsupported(result.reason or "The question cannot be answered.", {
                "question": result.question, "provider": result.provider, "attempts": result.attempts,
                "needs_join": bool(getattr(result, "needs_join", False))})
        proposed = tuple(getattr(result, "proposed_relations", ()) or ())
        return {
            "question": result.question, "database": name, "connection": token,
            "request": request_to_json(result.request),
            "proposed_relations": _relations_to_json(proposed),
            "provider": result.provider, "attempts": result.attempts,
            "summary": jsonable_encoder(service.summarize(result.request, proposed)),
        }

    def preview(plan_doc: Mapping[str, Any], ctx) -> dict[str, Any]:
        return {"summary": plan_doc["summary"], "database": plan_doc["database"]}

    def execute(plan_doc: Mapping[str, Any], ctx) -> dict[str, Any]:
        service, _name, token = resolve(plan_doc["database"])
        if token != plan_doc["connection"]:
            raise ConnectionChanged("the approved tables changed since this plan was made")
        outcome = service.run_plan(
            plan_doc["question"], plan_result_from_task_plan(plan_doc),
            execution_context=TrustedExecutionContext(ctx.principal))
        cache.put_outcome(ctx.task_id, outcome)
        return outcome_metadata(outcome)

    def verify(plan_doc: Mapping[str, Any], result: Mapping[str, Any], ctx) -> Verification:
        evidence = {"workflow_id": result.get("workflow_id"), "row_count": result.get("row_count"),
                    "kind": result.get("kind")}
        artifact = result.get("artifact") or {}
        if artifact:
            evidence["artifact_size_bytes"] = artifact.get("size_bytes")
        if result.get("status") != "succeeded":
            return Verification(Verdict.FAILED, {**evidence, "status": result.get("status"),
                                                 "error_code": result.get("error_code")})
        # The job runner already validated the data and refused truncated reads; here we only confirm
        # that what it reports is complete: a workflow to trace it by and, if a file was made, content.
        if not result.get("workflow_id") or (artifact and not artifact.get("size_bytes")):
            return Verification(Verdict.UNVERIFIED, evidence)
        return Verification(Verdict.VERIFIED, evidence)

    return Operation(
        feature=FEATURE, name="report", effect=Effect.READ, permission=REPORT_PERMISSION, risk=0.1,
        approval=ApprovalPolicy.none(), description="Read-only table report from a question",
        plan=plan, preview=preview, execute=execute, verify=verify, max_active=200)


def build_report_feature(resolve: Resolver, cache: OutcomeCache) -> Feature:
    return Feature(FEATURE, "Database Agent", (build_report_operation(resolve, cache),))


__all__ = [
    "ConnectionChanged", "FEATURE", "OutcomeCache", "REPORT_OPERATION", "REPORT_PERMISSION",
    "build_report_feature", "build_report_operation", "outcome_from_metadata", "outcome_metadata", "plan_result_from_task_plan",
    "request_from_json", "request_to_json",
]
