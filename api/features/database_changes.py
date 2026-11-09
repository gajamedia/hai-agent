"""The Database Agent's data-change operation (``database.change``) on the shared platform.

Insert, update and delete keep every protection they had; the platform now carries the ones that are
not specific to databases:

* approval policy of a plan (exact phrase + fresh login, and a second person for the risky ones) is
  decided when it is planned and stored with the task;
* the task, its approval and its audit are durable (``platform.sqlite3``) and tied together;
* the plan holds business values, so it is **erased when the task ends** (``sensitive``);
* after the commit the table is read again and the result is verified.

What stays in the Database Agent: the planner, the SQL builder, the all-or-nothing transaction, the
before-image and the JSON-lines audit that ``DataChangeService`` has always written (those still go to
``<work_dir>/audit/changes-<database>.jsonl``, next to the platform's own audit).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from fastapi.encoders import jsonable_encoder

from agent.platform import (
    ApprovalPolicy, Effect, Operation, PlanUnsupported, Verdict, Verification,
)
from agent.workflow.data_change import ChangeRefused, ChangeRequest
from api.data_change_settings import APPROVE, EXECUTE, PLAN, requires_approval
from api.database_runtime_registry import DatabaseRuntimeRegistry

FEATURE = "database"
CHANGE_OPERATION = "database.change"


class ConnectionChanged(ValueError):
    """The writable approvals of the connection changed after the plan was made."""


# --------------------------------------------------------------------------- plan <-> JSON

def change_request_to_json(request: ChangeRequest) -> dict[str, Any]:
    return {
        "operation": request.operation, "table": request.table, "title": request.title,
        "max_rows": request.max_rows,
        "set_values": [[c, v] for c, v in request.set_values],
        "rows": [[[c, v] for c, v in row] for row in request.rows],
        "conditions": [[c, op, list(v) if isinstance(v, (tuple, list)) else v]
                       for c, op, v in request.conditions],
    }


def change_request_from_json(data: Mapping[str, Any]) -> ChangeRequest:
    return ChangeRequest(
        operation=data["operation"], table=data["table"], title=data.get("title", ""),
        max_rows=int(data["max_rows"]),
        set_values=tuple((c, v) for c, v in data["set_values"]),
        rows=tuple(tuple((c, v) for c, v in row) for row in data["rows"]),
        conditions=tuple((c, op, tuple(v) if isinstance(v, list) else v)
                         for c, op, v in data["conditions"]))


# --------------------------------------------------------------------------- the feature

_VERB = {"insert": "Tambah data", "update": "Ubah data", "delete": "Hapus data"}


def describe_change(plan: Optional[Mapping[str, Any]], result: Optional[Mapping[str, Any]]) -> str:
    """One line that never needs the erased plan: the outcome once it ran, the kind and table before."""
    if result:
        return f"{_VERB.get(result.get('operation'), 'Ubah data')}: {result.get('rows_affected')} baris di {result.get('table')}"
    summary = (plan or {}).get("summary")
    if summary:
        return f"{_VERB.get(summary.get('operation'), 'Ubah data')} pada tabel {summary.get('table')}"
    return "Perubahan data (rincian sudah dihapus)"


def build_change_operation(get_registry: Callable[[], DatabaseRuntimeRegistry]) -> Operation:
    """``get_registry`` returns the live database registry (raising if it is not configured)."""

    def plan(request: Mapping[str, Any], ctx) -> dict[str, Any]:
        registry = get_registry()
        name = registry.resolve_change_name(request.get("database"))
        service = registry.data_changes(name)
        result = service.planner.plan(request["question"], max_rows=request["max_rows"])
        if result.request is None:
            raise PlanUnsupported(result.reason or "The request cannot be done.", {"kind": "unsupported"})
        try:
            preview = service.preview(result.request)
        except ChangeRefused as exc:
            raise PlanUnsupported(str(exc), {"kind": "refused",
                                             "plan": service.summarize(result.request)}) from exc
        return {
            "question": result.question, "database": name, "connection": registry.change_token(name),
            "request": change_request_to_json(result.request),
            "summary": service.summarize(result.request),
            "expected_rows": preview.expected_rows,
            "before": jsonable_encoder(preview.before), "after": jsonable_encoder(preview.after),
            "note": preview.sample_note,
            "confirm_phrase": service.confirm_phrase(result.request, preview.expected_rows),
            "provider": result.provider,
        }

    def approval_for(plan_doc: Mapping[str, Any], ctx) -> ApprovalPolicy:
        # Exact phrase and a fresh login always; a second person for the operations the server names.
        return ApprovalPolicy(confirm_phrase=True, fresh_login=True,
                              second_approver=requires_approval(plan_doc["summary"]["operation"]))

    def preview(plan_doc: Mapping[str, Any], ctx) -> dict[str, Any]:
        return {"confirm_phrase": plan_doc["confirm_phrase"], "summary": plan_doc["summary"],
                "before": plan_doc["before"], "after": plan_doc["after"], "note": plan_doc["note"],
                "expected_rows": plan_doc["expected_rows"]}

    def execute(plan_doc: Mapping[str, Any], ctx) -> dict[str, Any]:
        registry = get_registry()
        name = plan_doc["database"]
        if registry.change_token(name) != plan_doc["connection"]:
            raise ConnectionChanged("the approved tables changed since this plan was made")
        done = registry.data_changes(name).execute(
            change_request_from_json(plan_doc["request"]), expected_rows=plan_doc["expected_rows"],
            actor=ctx.actor_id, approver=ctx.approved_by, plan_id=ctx.task_id,
            question=plan_doc["question"])
        return {"rows_affected": done.rows_affected, "operation": plan_doc["summary"]["operation"],
                "table": plan_doc["summary"]["table"]}

    def verify(plan_doc: Mapping[str, Any], result: Mapping[str, Any], ctx) -> Verification:
        service = get_registry().data_changes(plan_doc["database"])
        seen = service.verify(change_request_from_json(plan_doc["request"]), plan_doc["expected_rows"])
        evidence = {**seen, "rows_affected": result.get("rows_affected")}
        if result.get("rows_affected") != plan_doc["expected_rows"] or not seen["confirmed"]:
            # Committed, but the table does not look the way it should now: say so, do not claim success.
            return Verification(Verdict.UNVERIFIED, evidence)
        return Verification(Verdict.VERIFIED, evidence)

    return Operation(
        feature=FEATURE, name="change", effect=Effect.WRITE, permission=EXECUTE, risk=0.8,
        approval=ApprovalPolicy(confirm_phrase=True, fresh_login=True),
        description="Insert, update or delete rows in tables approved for writing",
        plan=plan, preview=preview, execute=execute, verify=verify, approver_permission=APPROVE,
        plan_permission=PLAN, approval_for=approval_for, sensitive=True, max_active=100,
        label="Ubah data", describe=describe_change)


__all__ = ["CHANGE_OPERATION", "ConnectionChanged", "build_change_operation", "describe_change",
           "change_request_from_json", "change_request_to_json"]
