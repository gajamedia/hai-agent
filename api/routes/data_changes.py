from __future__ import annotations

"""Insert / update / delete, behind five separate protections.

1. its own permissions (``data:change:plan`` / ``execute`` / ``approve``), apart from reading;
2. a write-capable database account that only the server knows, and per-table write approval;
3. a preview and the user's own row limit (the server only sets a ceiling);
4. a fresh login (step-up): the token's ``auth_time`` must be recent, so the person has just typed
   their credentials into Keycloak; this application never sees the password;
5. the exact confirmation phrase, and for the risky operations a second person's approval.

Everything runs in one transaction that is rolled back unless it touches exactly the rows previewed.

Planning, approval, running, verification and the platform audit go through the shared platform
pipeline (``database.change``, see ``api/features/database_changes.py``); this module only turns
HTTP into pipeline calls and keeps the response shapes the browser already knows.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from agent.llm.fallback_provider import AllProvidersFailedError
from agent.platform import (
    ApprovalRequired, ConfirmationMismatch, ExecutionFailed, PermissionDenied, PlanError, PlanUnsupported,
    PolicyViolation, StepUpRequired, TaskConflict, TaskExpired, TaskNotFound, TaskStatus, TooManyTasks,
)
from agent.workflow.column_fill import apply_fill, propose_fill
from agent.workflow.data_change import MAX_IMPORT_ROWS, ChangeRefused
from agent.workflow.uploaded_table import MappingNeeded, UploadRefused, read_table, rows_for_table
from api.data_change_settings import APPROVE, EXECUTE, PLAN, requires_approval, settings
from api.database_runtime_registry import (
    DatabaseRuntimeRegistry, RegistryError, RegistryNotFound,
)
from api.features.database_changes import CHANGE_OPERATION, ConnectionChanged
from api.platform_runtime import get_pipeline
from api.routes.uploads import needs_mapping, parse_mapping, read_upload
from api.security.authentication import AuthenticationError, get_attached_principal
from api.security.authorization import AuthorizationDenied, AuthorizationPolicy

logger = logging.getLogger(__name__)
LLM_UNAVAILABLE = (
    "The AI model did not answer in time, so no plan was made. Nothing was changed. Try again in a minute; "
    "if it keeps happening an administrator can run scripts/check_llm_provider.py.")
router = APIRouter(prefix="/data-changes", tags=["data-changes"])
_now = time.time  # replaced in tests
_requires_approval = requires_approval  # kept for callers that imported it from here


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": code, "message": message})


def _principal(request: Request, permission: str):
    try:
        principal = get_attached_principal(request)
    except AuthenticationError as exc:
        raise _err(401, "authentication_required",
                   "A verified authenticated principal is required.") from exc
    policy = getattr(request.app.state, "authorization_policy", None)
    if not isinstance(policy, AuthorizationPolicy):
        raise _err(503, "authorization_policy_unavailable",
                   "The application authorization policy is not configured.")
    try:
        policy.require(principal, permission)
    except AuthorizationDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {permission} permission.") from exc
    return principal


def _require_recent_login(principal) -> None:
    """The person must have typed their credentials recently; otherwise the browser logs in again."""
    window = settings()["step_up_max_age_seconds"]
    auth_time = getattr(principal, "auth_time", None)
    if auth_time is None or not (0 <= _now() - auth_time <= window):
        raise _err(403, "step_up_required",
                   f"Sign in again to confirm it is you (your last sign-in is older than {window} seconds).")


def _registry(request: Request) -> DatabaseRuntimeRegistry:
    registry = getattr(request.app.state, "database_runtime_registry", None)
    if not isinstance(registry, DatabaseRuntimeRegistry):
        raise _err(503, "database_connections_unavailable", "Database connections are not configured.")
    return registry


def _pipeline(request: Request):
    return get_pipeline(request.app)


_STATE = {
    TaskStatus.EXECUTING: "running", TaskStatus.VERIFYING: "running",
    TaskStatus.COMPLETED: "done", TaskStatus.UNVERIFIED: "done", TaskStatus.REJECTED: "rejected",
    TaskStatus.CANCELLED: "cancelled",
}


def _state(task) -> str:
    """The state names the browser has always seen (planned | approved | rejected | running | done | unknown)."""
    requires = bool((task.approval or {}).get("second_approver"))
    if task.status is TaskStatus.APPROVED:
        return "approved" if requires else "planned"
    if task.status is TaskStatus.AWAITING_APPROVAL:
        return "planned"
    if task.status is TaskStatus.FAILED:
        return "done" if task.result else "unknown"
    return _STATE.get(task.status, "planned")


def _view(task, pipeline, *, with_phrase: bool, viewer: str | None = None) -> dict[str, Any]:
    plan = task.plan or {}
    out = {"plan_id": task.task_id, "database": plan.get("database"), "question": plan.get("question"),
           "plan": plan.get("summary"), "column_order": plan.get("column_order") or [],
           "preview": {"before": plan.get("before", []), "after": plan.get("after", []),
                       "note": plan.get("note", "")},
           "expected_rows": plan.get("expected_rows"),
           "requires_approval": bool((task.approval or {}).get("second_approver")),
           "state": _state(task), "approved_by": task.approved_by, "owner": task.owner,
           "own": task.owner == viewer, "expires_in_seconds": pipeline.seconds_left(task)}
    if with_phrase:
        out["confirm_phrase"] = task.confirm_phrase
    return out


def _mine(request: Request, plan_id: str, principal):
    """The owner's own change task; anything else (unknown, someone else's, cancelled, expired) is 'not found'."""
    pipeline = _pipeline(request)
    try:
        task = pipeline.get_task(plan_id, principal)
    except TaskNotFound as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    waiting = task.status in (TaskStatus.PLANNED, TaskStatus.AWAITING_APPROVAL, TaskStatus.APPROVED)
    if (task.owner != principal.actor_id or task.operation_id != CHANGE_OPERATION
            or task.status is TaskStatus.CANCELLED or (waiting and pipeline.seconds_left(task) == 0)):
        raise _err(404, "plan_not_found", "change plan not found or expired")
    return task


@router.get("/config")
def get_config(request: Request):
    _principal(request, PLAN)
    return settings()


@router.post("/plans", status_code=201)
def create_plan(request: Request, payload: dict[str, Any] = Body(...)):
    """Turn a request into a validated change and show what it would do. Nothing is changed."""
    principal = _principal(request, PLAN)
    unknown = set(payload) - {"question", "database", "max_rows"}
    question, database, max_rows = payload.get("question"), payload.get("database"), payload.get("max_rows")
    ceiling = settings()["max_rows_ceiling"]
    if (unknown or not isinstance(question, str) or not question.strip()
            or not (database is None or isinstance(database, str))
            or isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= ceiling):
        raise _err(422, "invalid_request",
                   'Body must be {"question": "<text>", "database": "<name, optional>", '
                   f'"max_rows": <1 to {ceiling}>}}.')
    registry = _registry(request)
    try:
        name = registry.resolve_change_name(database)
        service = registry.data_changes(name)
    except RegistryNotFound as exc:
        raise _err(404, "database_not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "changes_not_available", str(exc)) from exc
    try:
        task = _pipeline(request).submit(
            CHANGE_OPERATION, {"question": question, "database": name, "max_rows": max_rows}, principal)
    except PlanUnsupported as exc:
        if exc.details.get("kind") == "refused":
            return {"status": "refused", "reason": exc.reason, "plan_id": None,
                    "plan": exc.details.get("plan")}
        return {"status": "unsupported", "reason": exc.reason, "plan_id": None}
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {EXECUTE} permission.") from exc
    except TooManyTasks as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        cause = exc.__cause__
        if isinstance(cause, ValueError):
            raise _err(422, "invalid_question", str(cause)) from exc
        if isinstance(cause, AllProvidersFailedError):
            logger.warning("Data change planning: no LLM provider answered: %s", cause)
            raise _err(503, "llm_unavailable", LLM_UNAVAILABLE) from exc
        logger.error("Data change planning failed.", exc_info=exc)
        raise _err(503, "data_change_failed", "A plan could not be produced right now.") from exc
    service.audit("planned", plan_id=task.task_id, actor=principal.actor_id,
                  plan=task.plan["summary"], expected_rows=task.plan["expected_rows"])
    return {"status": "planned", **_view(task, _pipeline(request), with_phrase=True)}


@router.post("/file-plans", status_code=201)
async def create_plan_from_file(request: Request, filename: str, table: str | None = None,
                                database: str | None = None, match: str | None = None,
                                mapping: str | None = None, max_rows: int = MAX_IMPORT_ROWS,
                                question: str | None = None, header_row: int = 1):
    """Update many rows from an uploaded file (CSV, TSV, TXT or .xlsx). Each row of the file names one existing row
    by its value in the ``match`` column (the table's key unless you name another unique column) and gives the new
    values for that row; an empty cell leaves that value as it is. Read by our own code, never by a model, and
    shown as the same preview; the exact phrase, fresh login and, where the server requires it, a second person
    still apply before anything changes. A value that matches no row, or more than one, refuses the whole file."""
    principal = _principal(request, PLAN)
    ceiling = settings()["max_rows_ceiling"]
    if not 1 <= max_rows <= ceiling:
        raise _err(422, "invalid_request", f"max_rows must be 1 to {ceiling}.")
    data = await read_upload(request)
    chosen = parse_mapping(mapping)
    registry = _registry(request)
    try:
        name = registry.resolve_change_name(database)
        service = registry.data_changes(name)
    except RegistryNotFound as exc:
        raise _err(404, "database_not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "changes_not_available", str(exc)) from exc
    able = sorted(t for t, w in service.planner.tables.items() if "update" in w.operations)
    if table is None and len(able) == 1:
        table = able[0]
    if table not in able:
        raise _err(422, "table_required", "Choose the table to update: " + (", ".join(able) or "(none allows update)") + ".")
    writable = service.table(table)
    match = match or writable.entry.key_column
    if not match or match not in writable.entry.filterable_columns:
        raise _err(422, "invalid_match", "Rows are found by one of: " + ", ".join(writable.entry.filterable_columns) + ".")
    usable = list(dict.fromkeys([match, *writable.writable_columns]))
    try:
        parsed = await run_in_threadpool(read_table, filename, data, header_row)
        rows = await run_in_threadpool(rows_for_table, parsed, writable.entry.columns, usable, chosen)
        if not any(match in row for row in rows):
            raise UploadRefused(
                f"Tidak ada kolom file yang cocok dengan {match}, kolom yang dipakai untuk menemukan baris yang diubah. "
                f"Tambahkan kolom {match} di file, atau pilih kolom pencari lain yang ada di file.")
    except MappingNeeded as exc:
        return await needs_mapping(exc, parsed, service.planner.llm, writable.entry.columns, usable,
                                   table=table, match=match)
    except UploadRefused as exc:
        return {"status": "refused", "reason": str(exc), "plan_id": None}
    instruction = (question or "").strip()[:2000]
    left_alone: list[str] = []
    if instruction:
        # The typed text says what to set on the columns the file does not carry. A model turns it into rules
        # from column names only; our code applies them to every row of the file.
        by_file = sorted({c for r in rows for c in r})
        try:
            rules, _extra = await run_in_threadpool(
                propose_fill, service.planner.llm, instruction, writable.entry.columns,
                list(writable.writable_columns), by_file, table, "update")
            rows = await run_in_threadpool(apply_fill, rows, rules, writable.entry.columns)
        except UploadRefused as exc:
            return {"status": "refused", "reason": str(exc), "plan_id": None}
        left_alone = [c for c in writable.writable_columns if c not in by_file and c not in rules]
    rows = [{c: v for c, v in row.items() if v is not None or c == match} for row in rows]
    pipeline = _pipeline(request)
    try:
        task = pipeline.submit(CHANGE_OPERATION, {
            "question": f"[file] {parsed.filename}", "database": name, "max_rows": max_rows,
            "file": {"name": parsed.filename, "sha256": parsed.sha256, "table": table, "match": match,
                     "rows": rows}}, principal)
    except PlanUnsupported as exc:
        return {"status": "refused" if exc.details.get("kind") == "refused" else "unsupported",
                "reason": exc.reason, "plan_id": None, "plan": exc.details.get("plan")}
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {PLAN} permission.") from exc
    except TooManyTasks as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        logger.error("Data change from a file failed.", exc_info=exc)
        raise _err(503, "data_change_failed", "The file could not be prepared right now.") from exc
    service.audit("planned", plan_id=task.task_id, actor=principal.actor_id, plan=task.plan["summary"],
                  expected_rows=task.plan["expected_rows"], file={"name": parsed.filename, "sha256": parsed.sha256})
    view = _view(task, pipeline, with_phrase=True)
    view["file"] = {"name": parsed.filename, "rows": len(rows), "sheet": parsed.sheet, "match": match,
                    "columns": list(parsed.headers), "left_alone": left_alone}
    return {"status": "planned", **view}


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str, request: Request):
    """Show a plan again, for example after the browser returned from signing in again."""
    principal = _principal(request, PLAN)
    task = _mine(request, plan_id, principal)
    view = _view(task, _pipeline(request), with_phrase=True)
    return {"status": "planned" if view["state"] in {"planned", "approved"} else view["state"], **view}


@router.delete("/plans/{plan_id}")
def cancel_plan(plan_id: str, request: Request):
    principal = _principal(request, PLAN)
    _mine(request, plan_id, principal)
    try:
        _pipeline(request).cancel(plan_id, principal)
    except TaskNotFound as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    except TaskConflict as exc:
        raise _err(409, "plan_not_runnable", str(exc)) from exc
    return {"plan_id": plan_id, "status": "cancelled"}


@router.post("/plans/{plan_id}/run")
def run_plan(plan_id: str, request: Request, payload: dict[str, Any] = Body(...)):
    """Execute exactly the previewed change: fresh login + exact phrase (+ approval when required)."""
    principal = _principal(request, EXECUTE)
    _require_recent_login(principal)
    if set(payload) != {"confirm"} or not isinstance(payload["confirm"], str):
        raise _err(422, "invalid_request", 'Body must be {"confirm": "<the phrase shown with the plan>"}.')
    key = request.headers.get("Idempotency-Key")
    if key is not None:
        key = key.strip()
        if not key or len(key) > 255:
            raise _err(400, "invalid_idempotency_key",
                       "Idempotency-Key must contain 1 to 255 non-whitespace characters.")
    pipeline = _pipeline(request)
    try:
        task = pipeline.get_task(plan_id, principal)
    except TaskNotFound as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    if task.owner != principal.actor_id or task.operation_id != CHANGE_OPERATION:
        raise _err(404, "plan_not_found", "change plan not found or expired")
    if task.status is TaskStatus.APPROVED:
        # The connection must be the very one the plan was made for (same writable approvals).
        try:
            current = _registry(request).change_token(task.plan["database"])
        except (RegistryError, HTTPException):
            current = None
        if current is None or current != task.plan["connection"]:
            raise _err(409, "connection_changed",
                       "The database connection changed since this plan was made; plan it again.")
    try:
        outcome = pipeline.run(plan_id, principal, confirm=payload["confirm"], idempotency_key=key)
    except (TaskNotFound, TaskExpired) as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    except ApprovalRequired as exc:
        raise _err(409, "plan_not_runnable", str(exc)) from exc
    except TaskConflict as exc:
        raise _err(409, "plan_not_runnable", str(exc)) from exc
    except ConfirmationMismatch as exc:
        raise _err(422, "confirmation_mismatch", f"Type exactly: {task.confirm_phrase}") from exc
    except StepUpRequired as exc:
        raise _err(403, "step_up_required", "Sign in again to confirm it is you.") from exc
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {EXECUTE} permission.") from exc
    except ExecutionFailed as exc:
        cause = exc.__cause__
        if isinstance(cause, ChangeRefused):
            raise _err(409, "change_refused", str(cause)) from exc
        if isinstance(cause, ConnectionChanged):
            raise _err(409, "connection_changed",
                       "The database connection changed since this plan was made; plan it again.") from exc
        logger.error("Data change failed.", exc_info=exc)
        raise _err(503, "data_change_failed",
                   "The change failed and was rolled back if it had not been committed. Check the data "
                   "before trying again.") from exc
    done = outcome.task
    verdict = (done.verification or {}).get("verdict", "unverified")
    return {"status": "committed", "rows_affected": done.result["rows_affected"],
            "operation": done.result["operation"], "table": done.result["table"],
            "approved_by": done.approved_by, "verification": verdict, "replayed": outcome.replayed}


@router.get("/approvals")
def list_approvals(request: Request):
    """Changes by other people that wait for a second person (never your own)."""
    principal = _principal(request, APPROVE)
    pipeline = _pipeline(request)
    waiting = [t for t in pipeline.waiting_for_approval(principal) if t.operation_id == CHANGE_OPERATION]
    return {"approvals": [_view(t, pipeline, with_phrase=False, viewer=principal.actor_id)
                          for t in waiting]}


def _decide(plan_id: str, request: Request, *, reject: bool):
    principal = _principal(request, APPROVE)
    if not reject:
        _require_recent_login(principal)
    pipeline = _pipeline(request)
    try:
        before = pipeline.get_task(plan_id, principal)   # a rejected task's plan is erased: keep what the audit needs
    except TaskNotFound as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    if before.operation_id != CHANGE_OPERATION:
        raise _err(404, "plan_not_found", "change plan not found or expired")
    try:
        task = pipeline.approve(plan_id, principal, reject=reject)
    except (TaskNotFound, TaskExpired) as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    except (TaskConflict, PolicyViolation) as exc:
        raise _err(409, "approval_not_possible", str(exc)) from exc
    except StepUpRequired as exc:
        raise _err(403, "step_up_required", "Sign in again to confirm it is you.") from exc
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {APPROVE} permission.") from exc
    plan = before.plan
    try:
        _registry(request).data_changes(plan["database"]).audit(
            "rejected" if reject else "approved", plan_id=task.task_id, approver=principal.actor_id,
            owner=task.owner, plan=plan["summary"], expected_rows=plan["expected_rows"],
            self_approval=task.owner == principal.actor_id)
    except Exception:  # the platform audit already holds this decision
        logger.warning("Could not write the data-change audit line for %s.", plan_id, exc_info=True)
    return {"plan_id": task.task_id, "status": _state(task)}


@router.post("/approvals/{plan_id}/approve")
def approve_change(plan_id: str, request: Request):
    return _decide(plan_id, request, reject=False)


@router.post("/approvals/{plan_id}/reject")
def reject_change(plan_id: str, request: Request):
    return _decide(plan_id, request, reject=True)


__all__ = ["router"]
