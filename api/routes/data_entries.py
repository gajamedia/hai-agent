from __future__ import annotations

"""Recording new data: say it in words, see what will be recorded, press "Simpan".

This is the everyday counterpart of ``/data-changes``. It can only *add* rows (insert), it has its own
permission (``data:entry:record``, the ``data-entry`` role) and it asks for no typed phrase, no fresh
sign-in and no second person: a clerk recording a day's sales cannot be slowed by those. What it keeps:
the writer account that only the server knows, the per-table write approval, the preview and the row
limit, one all-or-nothing transaction, the read-back check and the audit trail. Updating and deleting
stay on ``/data-changes`` with its stricter approvals.

Planning and running go through the shared platform pipeline (``database.entry``).
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, Body, Request
from starlette.concurrency import run_in_threadpool

from agent.llm.fallback_provider import AllProvidersFailedError
from agent.platform import (
    ApprovalRequired, ConfirmationMismatch, ExecutionFailed, PermissionDenied, PlanError, PlanUnsupported,
    PolicyViolation, TaskConflict, TaskExpired, TaskNotFound, TaskStatus, TooManyTasks,
)
from agent.workflow.data_change import MAX_IMPORT_ROWS, MAX_INSERT_ROWS, ChangeRefused
from agent.workflow.column_fill import apply_fill, propose_fill
from agent.workflow.uploaded_table import (
    MAX_UPLOAD_BYTES, SUPPORTED, MappingNeeded, UploadRefused, read_table, rows_for_table,
)
from api.data_change_settings import ENTRY, settings
from api.database_runtime_registry import RegistryError, RegistryNotFound
from api.features.database_changes import ENTRY_OPERATION, UNDO_OPERATION, ConnectionChanged, undone_entry_ids
from api.routes.uploads import needs_mapping, parse_mapping, read_upload
from api.routes.data_changes import LLM_UNAVAILABLE, _err, _pipeline, _principal, _registry, _view

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-entries", tags=["data-entries"])
_now = time.time  # replaced in tests


def _mine(request: Request, plan_id: str, principal, *, allow_finished: bool = False):
    """The person's own entry; anything else (unknown, someone else's, cancelled, expired) is 'not found'."""
    pipeline = _pipeline(request)
    try:
        task = pipeline.get_task(plan_id, principal)
    except TaskNotFound as exc:
        raise _err(404, "plan_not_found", "entry not found or expired") from exc
    waiting = task.status in (TaskStatus.PLANNED, TaskStatus.AWAITING_APPROVAL, TaskStatus.APPROVED)
    if (task.owner != principal.actor_id or task.operation_id != ENTRY_OPERATION
            or task.status is TaskStatus.CANCELLED
            or (waiting and pipeline.seconds_left(task) == 0 and not allow_finished)):
        raise _err(404, "plan_not_found", "entry not found or expired")
    return task


def _entry_view(task, pipeline) -> dict[str, Any]:
    view = _view(task, pipeline, with_phrase=False)
    view.pop("requires_approval", None)
    return view


@router.get("/config")
def get_config(request: Request):
    _principal(request, ENTRY)
    return {"max_rows": MAX_INSERT_ROWS, "max_file_rows": MAX_IMPORT_ROWS,
            "max_file_mb": MAX_UPLOAD_BYTES // (1024 * 1024), "file_types": list(SUPPORTED)}


@router.get("/databases")
def list_databases(request: Request):
    """Where this person may record new data (names and tables only; connecting is someone else's job)."""
    _principal(request, ENTRY)
    return {"databases": _registry(request).entry_databases()}


@router.post("/plans", status_code=201)
def create_entry(request: Request, payload: dict[str, Any] = Body(...)):
    """Turn what the person wrote into rows to record, and show them. Nothing is recorded yet."""
    principal = _principal(request, ENTRY)
    question, database = payload.get("question"), payload.get("database")
    if (set(payload) - {"question", "database"} or not isinstance(question, str) or not question.strip()
            or not (database is None or isinstance(database, str))):
        raise _err(422, "invalid_request",
                   'Body must be {"question": "<text>", "database": "<name, optional>"}.')
    registry = _registry(request)
    try:
        name = registry.resolve_entry_name(database)
        service = registry.data_changes(name)
    except RegistryNotFound as exc:
        raise _err(404, "database_not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "changes_not_available", str(exc)) from exc
    try:
        task = _pipeline(request).submit(
            ENTRY_OPERATION, {"question": question, "database": name, "max_rows": MAX_INSERT_ROWS}, principal)
    except PlanUnsupported as exc:
        if exc.details.get("kind") == "refused":
            return {"status": "refused", "reason": exc.reason, "plan_id": None,
                    "plan": exc.details.get("plan")}
        return {"status": "unsupported", "reason": exc.reason, "plan_id": None}
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {ENTRY} permission.") from exc
    except TooManyTasks as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        cause = exc.__cause__
        if isinstance(cause, ValueError):
            raise _err(422, "invalid_question", str(cause)) from exc
        if isinstance(cause, AllProvidersFailedError):
            logger.warning("Data entry planning: no LLM provider answered: %s", cause)
            raise _err(503, "llm_unavailable", LLM_UNAVAILABLE) from exc
        logger.error("Data entry planning failed.", exc_info=exc)
        raise _err(503, "data_entry_failed", "The entry could not be prepared right now.") from exc
    service.audit("entry_planned", plan_id=task.task_id, actor=principal.actor_id,
                  plan=task.plan["summary"], expected_rows=task.plan["expected_rows"])
    return {"status": "planned", **_entry_view(task, _pipeline(request))}


def _earlier_upload(pipeline, principal, sha256: str, table: str):
    """This person's earlier, finished recording of the very same file into the same table, if any."""
    tasks, _more = pipeline.list_tasks(principal, statuses=[TaskStatus.COMPLETED], limit=100)
    for task, _op in tasks:
        info = (task.result or {}).get("file")
        if (task.operation_id == ENTRY_OPERATION and info and info.get("sha256") == sha256
                and task.result.get("table") == table):
            return task
    return None


@router.post("/file-plans", status_code=201)
async def create_entry_from_file(request: Request, filename: str, table: str | None = None,
                                 database: str | None = None, mapping: str | None = None,
                                 question: str | None = None, header_row: int = 1):
    """Rows from an uploaded file (the raw body: CSV, TSV, TXT or .xlsx). The file is read by our own code,
    never by a language model; what comes back is the same preview as for a typed entry, and nothing is
    recorded until "Simpan". When the file's column names do not all match the table's, the answer is
    ``needs_mapping`` with a proposal (a model sees the column names only, never the values); the person
    corrects it and sends the file again with ``mapping`` (a JSON object: file column -> table column or null)."""
    principal = _principal(request, ENTRY)
    data = await read_upload(request)
    registry = _registry(request)
    try:
        name = registry.resolve_entry_name(database)
        service = registry.data_changes(name)
    except RegistryNotFound as exc:
        raise _err(404, "database_not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "changes_not_available", str(exc)) from exc
    able = next((d["tables"] for d in registry.entry_databases() if d["name"] == name), [])
    if table is None and len(able) == 1:
        table = able[0]
    if table not in able:
        raise _err(422, "table_required", "Choose the table to record the file into: " + ", ".join(able) + ".")
    chosen = parse_mapping(mapping)
    try:
        parsed = await run_in_threadpool(read_table, filename, data, header_row)
        writable = service.table(table)
        rows = await run_in_threadpool(rows_for_table, parsed, writable.entry.columns, writable.writable_columns,
                                       chosen)
    except MappingNeeded as exc:
        return await needs_mapping(exc, parsed, service.planner.llm, writable.entry.columns,
                                   list(writable.writable_columns), table=table)
    except UploadRefused as exc:
        return {"status": "refused", "reason": str(exc), "plan_id": None}
    instruction = (question or "").strip()[:2000]
    by_file = sorted({c for r in rows for c in r})
    left_empty: list[str] = []
    if instruction:
        # The typed text is an instruction about the file. A model turns it into fill rules from column names
        # only; our code draws the values. Every resulting row is shown before anything is recorded.
        try:
            rules, extra = await run_in_threadpool(
                propose_fill, service.planner.llm, instruction, writable.entry.columns,
                list(writable.writable_columns), by_file, table)
            rows = await run_in_threadpool(apply_fill, rows, rules, writable.entry.columns)
            left_empty = [c for c in writable.writable_columns if c not in by_file and c not in rules]
            rows = rows + extra
        except UploadRefused as exc:
            return {"status": "refused", "reason": str(exc), "plan_id": None}
    pipeline = _pipeline(request)
    earlier = _earlier_upload(pipeline, principal, parsed.sha256, table)
    try:
        task = pipeline.submit(ENTRY_OPERATION, {
            "question": f"[file] {parsed.filename}", "database": name, "max_rows": MAX_IMPORT_ROWS,
            "file": {"name": parsed.filename, "sha256": parsed.sha256, "table": table, "rows": rows}}, principal)
    except PlanUnsupported as exc:
        return {"status": "refused" if exc.details.get("kind") == "refused" else "unsupported",
                "reason": exc.reason, "plan_id": None}
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {ENTRY} permission.") from exc
    except TooManyTasks as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        logger.error("Data entry from a file failed.", exc_info=exc)
        raise _err(503, "data_entry_failed", "The file could not be prepared right now.") from exc
    service.audit("entry_planned", plan_id=task.task_id, actor=principal.actor_id,
                  plan=task.plan["summary"], expected_rows=task.plan["expected_rows"],
                  file={"name": parsed.filename, "sha256": parsed.sha256})
    view = _entry_view(task, pipeline)
    view["file"] = {"name": parsed.filename, "rows": len(rows), "sheet": parsed.sheet,
                    "columns": list(parsed.headers),
                    "already_recorded_at": earlier.updated_at if earlier else None,
                    "left_empty": left_empty, "filled": by_file}
    return {"status": "planned", **view}


@router.get("/plans/{plan_id}")
def get_entry(plan_id: str, request: Request):
    principal = _principal(request, ENTRY)
    task = _mine(request, plan_id, principal)
    view = _entry_view(task, _pipeline(request))
    return {"status": "planned" if view["state"] in {"planned", "approved"} else view["state"], **view}


@router.delete("/plans/{plan_id}")
def cancel_entry(plan_id: str, request: Request):
    principal = _principal(request, ENTRY)
    _mine(request, plan_id, principal)
    try:
        _pipeline(request).cancel(plan_id, principal)
    except TaskNotFound as exc:
        raise _err(404, "plan_not_found", "entry not found or expired") from exc
    except TaskConflict as exc:
        raise _err(409, "plan_not_runnable", str(exc)) from exc
    return {"plan_id": plan_id, "status": "cancelled"}


@router.post("/plans/{plan_id}/run")
def run_entry(plan_id: str, request: Request):
    """Record exactly the rows that were shown ("Simpan"). Send an ``Idempotency-Key`` header to make a
    repeated press (or a repeated upload) record them only once."""
    principal = _principal(request, ENTRY)
    key = request.headers.get("Idempotency-Key")
    if key is not None:
        key = key.strip()
        if not key or len(key) > 255:
            raise _err(400, "invalid_idempotency_key",
                       "Idempotency-Key must contain 1 to 255 non-whitespace characters.")
    # With a key, an entry that already ran may be asked for again; it must still be the person's own.
    task = _mine(request, plan_id, principal, allow_finished=key is not None)
    file_info = (task.plan or {}).get("file")
    if file_info and task.status is TaskStatus.APPROVED:
        earlier = _earlier_upload(_pipeline(request), principal, file_info["sha256"], task.plan["summary"]["table"])
        if earlier is not None and request.headers.get("X-Confirm-Duplicate", "").lower() != "yes":
            raise _err(409, "file_already_recorded",
                       "This same file was already recorded into this table. Confirm to record it again.")
    if task.status is TaskStatus.APPROVED:
        try:
            current = _registry(request).change_token(task.plan["database"])
        except RegistryError:
            current = None
        if current is None or current != task.plan["connection"]:
            raise _err(409, "connection_changed",
                       "The database connection changed since this entry was prepared; prepare it again.")
    try:
        outcome = _pipeline(request).run(plan_id, principal, idempotency_key=key)
    except (TaskNotFound, TaskExpired) as exc:
        raise _err(404, "plan_not_found", "entry not found or expired") from exc
    except (ApprovalRequired, TaskConflict, ConfirmationMismatch, PolicyViolation) as exc:
        raise _err(409, "plan_not_runnable", str(exc)) from exc
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {ENTRY} permission.") from exc
    except ExecutionFailed as exc:
        cause = exc.__cause__
        if isinstance(cause, ChangeRefused):
            raise _err(409, "change_refused", str(cause)) from exc
        if isinstance(cause, ConnectionChanged):
            raise _err(409, "connection_changed",
                       "The database connection changed since this entry was prepared; "
                       "prepare it again.") from exc
        logger.error("Data entry failed.", exc_info=exc)
        raise _err(503, "data_entry_failed",
                   "Recording failed and was rolled back if it had not been committed. Check the data "
                   "before trying again.") from exc
    done = outcome.task
    verdict = (done.verification or {}).get("verdict", "unverified")
    return {"status": "committed", "entry_id": done.task_id, "rows_affected": done.result["rows_affected"],
            "table": done.result["table"], "verification": verdict, "replayed": outcome.replayed,
            "undoable": bool(done.result.get("undo")) and verdict == "verified"}


@router.get("/recent")
def recent_entries(request: Request):
    """The person's latest recorded entries, and which of them may still be taken back."""
    principal = _principal(request, ENTRY)
    pipeline = _pipeline(request)
    window = settings()["entry_undo_minutes"] * 60
    undone = undone_entry_ids(pipeline.store, principal.actor_id)
    items, now = [], _now()
    tasks, _more = pipeline.list_tasks(principal, statuses=[TaskStatus.COMPLETED], limit=100)
    for task, _op in tasks:
        info = (task.result or {}).get("undo")
        if task.operation_id != ENTRY_OPERATION or not info:
            continue
        gone = task.task_id in undone
        items.append({"entry_id": task.task_id, "database": info["database"], "table": info["table"],
                      "rows": len(info["rows"]), "recorded_at": task.updated_at, "undone": gone,
                      "undoable": not gone and now - task.updated_at <= window,
                      "undo_seconds_left": 0 if gone else max(0, int(window - (now - task.updated_at)))})
        if len(items) >= 20:
            break
    return {"entries": items, "undo_minutes": settings()["entry_undo_minutes"]}


@router.post("/{entry_id}/undo")
def undo_entry(entry_id: str, request: Request):
    """Take back an entry: remove exactly the rows it recorded, unless one was edited since (then nothing is)."""
    principal = _principal(request, ENTRY)
    _mine(request, entry_id, principal, allow_finished=True)
    pipeline = _pipeline(request)
    try:
        task = pipeline.submit(UNDO_OPERATION, {"entry_id": entry_id}, principal)
    except PlanUnsupported as exc:
        raise _err(409, "undo_not_possible", exc.reason) from exc
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {ENTRY} permission.") from exc
    except TooManyTasks as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        logger.error("Undo planning failed.", exc_info=exc)
        raise _err(503, "data_entry_failed", "The entry could not be taken back right now.") from exc
    try:
        outcome = pipeline.run(task.task_id, principal)
    except ExecutionFailed as exc:
        cause = exc.__cause__
        if isinstance(cause, (ChangeRefused, ConnectionChanged)):
            raise _err(409, "undo_refused", str(cause)) from exc
        logger.error("Undo failed.", exc_info=exc)
        raise _err(503, "data_entry_failed",
                   "Taking the entry back failed and was rolled back if it had not been committed. "
                   "Check the data before trying again.") from exc
    except (TaskNotFound, TaskExpired, TaskConflict, ApprovalRequired, PolicyViolation) as exc:
        raise _err(409, "undo_not_possible", str(exc)) from exc
    done = outcome.task
    verdict = (done.verification or {}).get("verdict", "unverified")
    return {"status": "undone", "rows_removed": done.result["rows_affected"], "table": done.result["table"],
            "verification": verdict}


__all__ = ["router"]
