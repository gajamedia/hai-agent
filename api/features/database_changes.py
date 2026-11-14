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
    ApprovalPolicy, Effect, Operation, PlanUnsupported, TaskStatus, Verdict, Verification,
)
from agent.workflow.data_change import MAX_IMPORT_ROWS, ChangePlanResult, ChangeRefused, ChangeRequest
from agent.workflow.table_report_planner import PlanValidationError
from api.data_change_settings import APPROVE, ENTRY, EXECUTE, PLAN, requires_approval
from api.database_runtime_registry import DatabaseRuntimeRegistry

FEATURE = "database"
CHANGE_OPERATION = "database.change"
ENTRY_OPERATION = "database.entry"
ENTRY_ONLY_INSERT = (
    "Ini bukan pencatatan data baru. Untuk mengubah atau menghapus data yang sudah ada, "
    "gunakan menu Ubah data (perlu izin penulis data).")


class ConnectionChanged(ValueError):
    """The writable approvals of the connection changed after the plan was made."""


# --------------------------------------------------------------------------- plan <-> JSON

def change_request_to_json(request: ChangeRequest) -> dict[str, Any]:
    return {
        "operation": request.operation, "table": request.table, "title": request.title,
        "max_rows": request.max_rows, "match_column": request.match_column,
        "set_values": [[c, v] for c, v in request.set_values],
        "rows": [[[c, v] for c, v in row] for row in request.rows],
        "conditions": [[c, op, list(v) if isinstance(v, (tuple, list)) else v]
                       for c, op, v in request.conditions],
    }


def change_request_from_json(data: Mapping[str, Any]) -> ChangeRequest:
    return ChangeRequest(
        operation=data["operation"], table=data["table"], title=data.get("title", ""),
        max_rows=int(data["max_rows"]), match_column=data.get("match_column", ""),
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
    return _build_operation(get_registry, entry=False)


def build_entry_operation(get_registry: Callable[[], DatabaseRuntimeRegistry]) -> Operation:
    """Recording new data (``database.entry``): what a clerk does all day, in plain words or from a file.

    Insert only, with its own permission. No typed phrase, no fresh login and no second person: the person
    sees what will be recorded and presses "Simpan". The row limit, the preview, the all-or-nothing
    transaction, the read-back check and the audit are the same as for every other change. Updating and
    deleting stay with ``database.change`` and its stricter approvals."""
    return _build_operation(get_registry, entry=True)


def _build_operation(get_registry: Callable[[], DatabaseRuntimeRegistry], *, entry: bool) -> Operation:

    def plan(request: Mapping[str, Any], ctx) -> dict[str, Any]:
        registry = get_registry()
        name = registry.resolve_change_name(request.get("database"))
        service = registry.data_changes(name)
        file = request.get("file")
        if file is not None:
            # Rows read from an uploaded file by our own parser; no language model sees them. The planner's
            # validation (table approved for the operation, writable columns, value types) still decides.
            try:
                if entry:
                    checked = service.planner.validate(
                        {"operation": "insert", "table": file["table"], "rows": file["rows"],
                         "title": f"Impor {file['name']}"[:200]},
                        max_rows=MAX_IMPORT_ROWS, row_limit=MAX_IMPORT_ROWS)
                else:
                    checked = service.planner.validate_bulk_update(
                        file["table"], file["match"], file["rows"], max_rows=request["max_rows"],
                        title=f"Ubah dari {file['name']}"[:200])
            except PlanValidationError as exc:
                raise PlanUnsupported(str(exc), {"kind": "unsupported"}) from exc
            result = ChangePlanResult(f"[file] {file['name']} ({len(file['rows'])} rows)", checked, None, "file")
        else:
            result = service.planner.plan(request["question"], max_rows=request["max_rows"])
        if result.request is None:
            raise PlanUnsupported(result.reason or "The request cannot be done.", {"kind": "unsupported"})
        if entry and result.request.operation != "insert":
            raise PlanUnsupported(ENTRY_ONLY_INSERT, {"kind": "unsupported"})
        try:
            preview = service.preview(result.request)
        except ChangeRefused as exc:
            raise PlanUnsupported(str(exc), {"kind": "refused",
                                             "plan": service.summarize(result.request)}) from exc
        return {
            "question": result.question, "database": name, "connection": registry.change_token(name),
            "request": change_request_to_json(result.request),
            "summary": service.summarize(result.request),
            "column_order": list(service.table(result.request.table).entry.columns),
            "expected_rows": preview.expected_rows,
            "before": jsonable_encoder(preview.before), "after": jsonable_encoder(preview.after),
            "note": preview.sample_note,
            "confirm_phrase": service.confirm_phrase(result.request, preview.expected_rows),
            "provider": result.provider,
            **({"file": {"name": file["name"], "sha256": file["sha256"], "rows": len(file["rows"])}}
               if file is not None else {}),
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
        out = {"rows_affected": done.rows_affected, "operation": plan_doc["summary"]["operation"],
               "table": plan_doc["summary"]["table"]}
        if plan_doc.get("file"):
            # What the file was, never what it held: lets a second upload of the same file be noticed.
            out["file"] = {"name": plan_doc["file"]["name"], "sha256": plan_doc["file"]["sha256"]}
        if done.inserted:
            # Enough to take exactly these rows back later: their keys and fingerprints, never their values.
            out["undo"] = {"database": name, "table": plan_doc["summary"]["table"],
                           "rows": [dict(row) for row in done.inserted]}
        return out

    def verify(plan_doc: Mapping[str, Any], result: Mapping[str, Any], ctx) -> Verification:
        service = get_registry().data_changes(plan_doc["database"])
        seen = service.verify(change_request_from_json(plan_doc["request"]), plan_doc["expected_rows"])
        evidence = {**seen, "rows_affected": result.get("rows_affected")}
        if result.get("rows_affected") != plan_doc["expected_rows"] or not seen["confirmed"]:
            # Committed, but the table does not look the way it should now: say so, do not claim success.
            return Verification(Verdict.UNVERIFIED, evidence)
        return Verification(Verdict.VERIFIED, evidence)

    if entry:
        return Operation(
            feature=FEATURE, name="entry", effect=Effect.WRITE, permission=ENTRY, risk=0.3,
            approval=ApprovalPolicy.none(),
            description="Record new rows in tables approved for writing (insert only, no extra approval)",
            plan=plan, preview=preview, execute=execute, verify=verify, plan_permission=ENTRY,
            sensitive=True, max_active=100, label="Catat data", describe=describe_change)
    return Operation(
        feature=FEATURE, name="change", effect=Effect.WRITE, permission=EXECUTE, risk=0.8,
        approval=ApprovalPolicy(confirm_phrase=True, fresh_login=True),
        description="Insert, update or delete rows in tables approved for writing",
        plan=plan, preview=preview, execute=execute, verify=verify, approver_permission=APPROVE,
        plan_permission=PLAN, approval_for=approval_for, sensitive=True, max_active=100,
        label="Ubah data", describe=describe_change)


UNDO_OPERATION = "database.undo"
_ALREADY = {TaskStatus.COMPLETED, TaskStatus.UNVERIFIED, TaskStatus.EXECUTING, TaskStatus.VERIFYING}


def undone_entry_ids(store, actor: str, *, limit: int = 200) -> set[str]:
    """Ids of the entries this person has already taken back (or is taking back right now)."""
    found: set[str] = set()
    for task in store.list_for_owner(actor, limit=limit):
        if task.operation_id == UNDO_OPERATION and task.status in _ALREADY and task.plan:
            found.add(task.plan.get("entry_id"))
    return found


def describe_undo(plan: Optional[Mapping[str, Any]], result: Optional[Mapping[str, Any]]) -> str:
    if result and result.get("rows_affected") is not None:
        return f"Batalkan pencatatan: {result['rows_affected']} baris di {result.get('table')}"
    summary = (plan or {}).get("summary") or {}
    if summary.get("table"):
        return f"Batalkan pencatatan: {summary.get('rows')} baris di {summary['table']}"
    return "Batalkan pencatatan"


def build_undo_operation(get_registry: Callable[[], DatabaseRuntimeRegistry],
                         get_store: Callable[[], Any], *, window_minutes: Callable[[], int],
                         clock: Callable[[], float]) -> Operation:
    """Take back an entry (``database.undo``): remove exactly the rows it made, once, by its own author, within
    a time window, and only if none of them was edited since. The author needs no delete approval for this: the
    rows are named by key, so nothing else in the table can be touched."""

    def plan(request: Mapping[str, Any], ctx) -> dict[str, Any]:
        store = get_store()
        try:
            source = store.get(request["entry_id"])
        except Exception:  # an unknown id is the same as someone else's
            source = None
        if source is None or source.owner != ctx.actor_id or source.operation_id != ENTRY_OPERATION:
            raise PlanUnsupported("Pencatatan itu tidak ditemukan.", {"kind": "unsupported"})
        info = (source.result or {}).get("undo")
        if source.status is not TaskStatus.COMPLETED or not info:
            raise PlanUnsupported("Pencatatan ini tidak bisa dibatalkan (tabelnya tidak punya kunci angka, "
                                  "atau hasilnya belum terverifikasi).", {"kind": "unsupported"})
        if clock() - source.updated_at > window_minutes() * 60:
            raise PlanUnsupported(f"Batas waktu pembatalan ({window_minutes()} menit) sudah lewat.",
                                  {"kind": "unsupported"})
        if source.task_id in undone_entry_ids(store, ctx.actor_id):
            raise PlanUnsupported("Pencatatan ini sudah dibatalkan.", {"kind": "unsupported"})
        registry = get_registry()
        return {"entry_id": source.task_id, "database": info["database"], "table": info["table"],
                "rows": info["rows"], "connection": registry.change_token(info["database"]),
                "summary": {"operation": "undo", "table": info["table"], "rows": len(info["rows"])}}

    def preview(plan_doc: Mapping[str, Any], ctx) -> dict[str, Any]:
        return {"summary": plan_doc["summary"]}

    def execute(plan_doc: Mapping[str, Any], ctx) -> dict[str, Any]:
        registry = get_registry()
        if registry.change_token(plan_doc["database"]) != plan_doc["connection"]:
            raise ConnectionChanged("the approved tables changed since this entry was recorded")
        removed = registry.data_changes(plan_doc["database"]).undo_insert(
            plan_doc["table"], plan_doc["rows"], actor=ctx.actor_id, plan_id=ctx.task_id,
            entry_id=plan_doc["entry_id"])
        return {"rows_affected": removed, "operation": "undo", "table": plan_doc["table"],
                "entry_id": plan_doc["entry_id"]}

    def verify(plan_doc: Mapping[str, Any], result: Mapping[str, Any], ctx) -> Verification:
        service = get_registry().data_changes(plan_doc["database"])
        key = service.table(plan_doc["table"]).entry.key_column
        keys = tuple(row["key"] for row in plan_doc["rows"])
        request = ChangeRequest(operation="delete", table=plan_doc["table"],
                                conditions=((key, "in", keys),), max_rows=len(keys))
        seen = service.verify(request, len(keys))
        evidence = {**seen, "rows_affected": result.get("rows_affected")}
        return Verification(Verdict.VERIFIED if seen["confirmed"] else Verdict.UNVERIFIED, evidence)

    return Operation(
        feature=FEATURE, name="undo", effect=Effect.WRITE, permission=ENTRY, risk=0.3,
        approval=ApprovalPolicy.none(),
        description="Take back an entry: remove exactly the rows it made, if none was edited since",
        plan=plan, preview=preview, execute=execute, verify=verify, plan_permission=ENTRY,
        max_active=50, label="Batalkan catat", describe=describe_undo)


__all__ = ["CHANGE_OPERATION", "ENTRY_OPERATION", "UNDO_OPERATION", "ConnectionChanged", "build_change_operation",
           "build_undo_operation", "describe_undo", "undone_entry_ids",
           "build_entry_operation", "describe_change",
           "change_request_from_json", "change_request_to_json"]
