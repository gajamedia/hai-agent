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
from typing import Any

from fastapi import APIRouter, Body, Request

from agent.llm.fallback_provider import AllProvidersFailedError
from agent.platform import (
    ApprovalRequired, ConfirmationMismatch, ExecutionFailed, PermissionDenied, PlanError, PlanUnsupported,
    PolicyViolation, TaskConflict, TaskExpired, TaskNotFound, TaskStatus, TooManyTasks,
)
from agent.workflow.data_change import MAX_INSERT_ROWS, ChangeRefused
from api.data_change_settings import ENTRY
from api.database_runtime_registry import RegistryError, RegistryNotFound
from api.features.database_changes import ENTRY_OPERATION, ConnectionChanged
from api.routes.data_changes import LLM_UNAVAILABLE, _err, _pipeline, _principal, _registry, _view

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-entries", tags=["data-entries"])


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
    return {"max_rows": MAX_INSERT_ROWS}


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
    return {"status": "committed", "rows_affected": done.result["rows_affected"],
            "table": done.result["table"], "verification": verdict, "replayed": outcome.replayed}


__all__ = ["router"]
