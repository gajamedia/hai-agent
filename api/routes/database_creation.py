from __future__ import annotations

"""Create a new database (and tables) from a request: preview, fresh login, exact phrase, audit.

The server profile decides whether this is possible at all (a separate account that can only CREATE
databases whose name starts with the profile's prefix) and what the prefix is. Nothing is ever dropped
or overwritten here.

Planning, running, verification and the platform audit go through the shared platform pipeline
(``database.create``, see ``api/features/database_creation.py``); this module only turns HTTP into
pipeline calls and keeps the response shapes the browser already knows.
"""

import logging
from typing import Any

from fastapi import APIRouter, Body, Request

from agent.llm.fallback_provider import AllProvidersFailedError
from agent.platform import (
    ConfirmationMismatch, ExecutionFailed, PermissionDenied, PlanError, PlanUnsupported, StepUpRequired,
    TaskConflict, TaskExpired, TaskNotFound, TaskStatus, TooManyTasks,
)
from agent.workflow.database_creation import CreationRefused
from api.database_runtime_registry import RegistryError, RegistryNotFound
from api.features.database_creation import CREATE_OPERATION
from api.platform_runtime import get_pipeline
from api.routes.data_changes import (
    LLM_UNAVAILABLE, _err, _principal, _registry, _require_recent_login,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/database-creation", tags=["database-creation"])
PERMISSION = "database:create:manage"


_STATE = {
    TaskStatus.EXECUTING: "running", TaskStatus.VERIFYING: "running",
    TaskStatus.COMPLETED: "done", TaskStatus.UNVERIFIED: "done", TaskStatus.CANCELLED: "cancelled",
}


def _state(task) -> str:
    """The state names the browser has always seen (planned | running | done | unknown)."""
    if task.status is TaskStatus.FAILED:
        return "done" if task.result else "unknown"
    return _STATE.get(task.status, "planned")


def _view(task, pipeline) -> dict[str, Any]:
    plan = task.plan or {}
    return {"plan_id": task.task_id, "profile": plan.get("profile"), "question": plan.get("question"),
            "plan": plan.get("summary"), "mode": plan.get("mode"),
            "statements": plan.get("statements", []),
            "confirm_phrase": task.confirm_phrase, "state": _state(task),
            "expires_in_seconds": pipeline.seconds_left(task)}


def _mine(request: Request, plan_id: str, principal):
    """The owner's own creation task; anything else (unknown, someone else's, cancelled, expired) is 'not found'."""
    pipeline = get_pipeline(request.app)
    try:
        task = pipeline.get_task(plan_id, principal)
    except TaskNotFound as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    waiting = task.status in (TaskStatus.PLANNED, TaskStatus.AWAITING_APPROVAL, TaskStatus.APPROVED)
    if (task.owner != principal.actor_id or task.operation_id != CREATE_OPERATION
            or task.status is TaskStatus.CANCELLED or (waiting and pipeline.seconds_left(task) == 0)):
        raise _err(404, "plan_not_found", "change plan not found or expired")
    return task


@router.get("/config")
def get_config(request: Request):
    _principal(request, PERMISSION)
    return {"profiles": _registry(request).creatable_profiles()}


@router.post("/plans", status_code=201)
def create_plan(request: Request, payload: dict[str, Any] = Body(...)):
    principal = _principal(request, PERMISSION)
    question, profile = payload.get("question"), payload.get("profile")
    if (set(payload) - {"question", "profile"} or not isinstance(question, str) or not question.strip()
            or not (profile is None or isinstance(profile, str))):
        raise _err(422, "invalid_request", 'Body must be {"question": "<text>", "profile": "<name, optional>"}.')
    registry = _registry(request)
    try:
        service = registry.database_creator(profile)
    except RegistryNotFound as exc:
        raise _err(404, "profile_not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "creation_not_available", str(exc)) from exc
    pipeline = get_pipeline(request.app)
    try:
        task = pipeline.submit(CREATE_OPERATION, {"question": question, "profile": service.profile}, principal)
    except PlanUnsupported as exc:
        if exc.details.get("kind") == "refused":
            return {"status": "refused", "reason": exc.reason, "plan_id": None}
        return {"status": "unsupported", "reason": exc.reason, "plan_id": None}
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {PERMISSION} permission.") from exc
    except TooManyTasks as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        cause = exc.__cause__
        if isinstance(cause, ValueError):
            raise _err(422, "invalid_question", str(cause)) from exc
        if isinstance(cause, AllProvidersFailedError):
            logger.warning("Database design: no LLM provider answered: %s", cause)
            raise _err(503, "llm_unavailable", LLM_UNAVAILABLE) from exc
        logger.error("Database design failed.", exc_info=exc)
        raise _err(503, "creation_failed", "A plan could not be produced right now.") from exc
    service.audit("planned", plan_id=task.task_id, actor=principal.actor_id, plan=task.plan["summary"])
    return {"status": "planned", **_view(task, pipeline)}


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str, request: Request):
    principal = _principal(request, PERMISSION)
    task = _mine(request, plan_id, principal)
    view = _view(task, get_pipeline(request.app))
    return {"status": view["state"], **view}


@router.delete("/plans/{plan_id}")
def cancel_plan(plan_id: str, request: Request):
    principal = _principal(request, PERMISSION)
    _mine(request, plan_id, principal)
    try:
        get_pipeline(request.app).cancel(plan_id, principal)
    except TaskNotFound as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    except TaskConflict as exc:
        raise _err(409, "plan_not_runnable", str(exc)) from exc
    return {"plan_id": plan_id, "status": "cancelled"}


@router.post("/plans/{plan_id}/run")
def run_plan(plan_id: str, request: Request, payload: dict[str, Any] = Body(...)):
    principal = _principal(request, PERMISSION)
    _require_recent_login(principal)
    if set(payload) != {"confirm"} or not isinstance(payload["confirm"], str):
        raise _err(422, "invalid_request", 'Body must be {"confirm": "<the phrase shown with the plan>"}.')
    key = request.headers.get("Idempotency-Key")
    if key is not None:
        key = key.strip()
        if not key or len(key) > 255:
            raise _err(400, "invalid_idempotency_key",
                       "Idempotency-Key must contain 1 to 255 non-whitespace characters.")
    pipeline = get_pipeline(request.app)
    task = _mine(request, plan_id, principal)
    try:
        outcome = pipeline.run(plan_id, principal, confirm=payload["confirm"], idempotency_key=key)
    except (TaskNotFound, TaskExpired) as exc:
        raise _err(404, "plan_not_found", "change plan not found or expired") from exc
    except TaskConflict as exc:
        raise _err(409, "plan_not_runnable", str(exc)) from exc
    except ConfirmationMismatch as exc:
        raise _err(422, "confirmation_mismatch", f"Type exactly: {task.confirm_phrase}") from exc
    except StepUpRequired as exc:
        raise _err(403, "step_up_required", "Sign in again to confirm it is you.") from exc
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {PERMISSION} permission.") from exc
    except ExecutionFailed as exc:
        cause = exc.__cause__
        if isinstance(cause, CreationRefused):
            raise _err(409, "creation_refused", str(cause)) from exc
        if isinstance(cause, RegistryError):
            raise _err(409, "creation_refused", str(cause)) from exc
        logger.error("Database creation failed.", exc_info=exc)
        raise _err(503, "creation_failed",
                   "Creating the database failed part way. Check the server before trying again; "
                   "nothing is dropped automatically.") from exc
    done = outcome.task
    return {"status": "created", **{k: done.result[k] for k in ("database", "tables", "profile", "mode")},
            "verification": (done.verification or {}).get("verdict", "unverified"),
            "replayed": outcome.replayed}


__all__ = ["router"]
