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
from starlette.concurrency import run_in_threadpool

from agent.llm.fallback_provider import AllProvidersFailedError
from agent.platform import (
    ConfirmationMismatch, ExecutionFailed, PermissionDenied, PlanError, PlanUnsupported, StepUpRequired,
    TaskConflict, TaskExpired, TaskNotFound, TaskStatus, TooManyTasks,
)
from agent.workflow.database_creation import CreationRefused
from agent.workflow.image_design import read_image
from agent.workflow.database_creation import table_prefix
from agent.workflow.file_context import describe_request, file_as_text
from agent.workflow.uploaded_table import UploadRefused, design_from_table, normalize_name, read_table
from api.database_runtime_registry import RegistryError, RegistryNotFound
from api.features.database_creation import CREATE_OPERATION, design_to_json
from api.platform_runtime import get_pipeline
from api.routes.uploads import read_upload
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


@router.post("/file-plans", status_code=201)
async def create_plan_from_file(request: Request, filename: str, database: str, table: str | None = None,
                                profile: str | None = None, question: str | None = None,
                                header_row: int = 1):
    """Design a database from the columns of an uploaded file (CSV, TSV, TXT or .xlsx): one table whose columns
    and their types come from the file. Same preview, fresh login and exact phrase as any creation; the file's
    rows are not loaded here (record them afterwards from the Catat data page, once the table is approved)."""
    principal = _principal(request, PERMISSION)
    data = await read_upload(request)
    registry = _registry(request)
    try:
        service = registry.database_creator(profile)
    except RegistryNotFound as exc:
        raise _err(404, "profile_not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "creation_not_available", str(exc)) from exc
    try:
        parsed = await run_in_threadpool(read_table, filename, data, header_row)
    except UploadRefused as exc:
        return {"status": "refused", "reason": str(exc), "plan_id": None}
    stem = filename.rsplit(".", 1)[0]
    table_name = (table or "").strip() or table_prefix() + normalize_name(stem)
    instruction = (question or "").strip()[:2000]
    sent_rows = None
    if instruction:
        # Here the file is part of the request: a model reads the typed text and the file (capped) together and
        # designs the database from both. Nothing in a design file is business data; the design is still checked
        # by our own validation and shown, with the exact statements, before anything is made.
        rows_text, sent_rows = await run_in_threadpool(file_as_text, parsed)
        payload = {"question": describe_request(
                       instruction=instruction, table=parsed, database=database.strip(),
                       table_name=(table or "").strip() or None, rows_text=rows_text, rows_sent=sent_rows),
                   "profile": service.profile, "database": database.strip(),
                   "table": (table or "").strip() or None}
        columns = [(h, "", True) for h in parsed.headers]
    else:
        detected = design_from_table(parsed)
        columns = [(n, t, True) for n, t in detected]
        payload = {"question": f"[file] {parsed.filename}", "profile": service.profile,
                   "file": {"name": parsed.filename, "sha256": parsed.sha256, "database": database.strip(),
                            "table": table_name, "columns": columns}}
    pipeline = get_pipeline(request.app)
    try:
        task = pipeline.submit(CREATE_OPERATION, payload, principal)
    except PlanUnsupported as exc:
        return {"status": "refused" if exc.details.get("kind") == "refused" else "unsupported",
                "reason": exc.reason, "plan_id": None}
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {PERMISSION} permission.") from exc
    except TooManyTasks as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        if isinstance(exc.__cause__, AllProvidersFailedError):
            logger.warning("Database design from a file: no LLM provider answered: %s", exc.__cause__)
            raise _err(503, "llm_unavailable", LLM_UNAVAILABLE) from exc
        logger.error("Database design from a file failed.", exc_info=exc)
        raise _err(503, "creation_failed", "A plan could not be produced right now.") from exc
    service.audit("planned", plan_id=task.task_id, actor=principal.actor_id, plan=task.plan["summary"],
                  file={"name": parsed.filename, "sha256": parsed.sha256})
    view = _view(task, pipeline)
    view["file"] = {"name": parsed.filename, "rows": len(parsed.rows), "sheet": parsed.sheet,
                    "columns": [{"name": n, "type": t, "nullable": null} for n, t, null in columns],
                    "sent_to_model_rows": sent_rows}
    return {"status": "planned", **view}


@router.post("/image-plans", status_code=201)
async def create_plan_from_image(request: Request, filename: str, database: str | None = None,
                                 table: str | None = None, profile: str | None = None,
                                 question: str | None = None):
    """Design a database from a picture of one (an ERD, a sketched table, a form).

    The picture is sent to the model provider so it can be read: the page says so before it is sent. It is
    never stored, and only its name and SHA-256 are kept in the history and audit. What the model reads becomes
    an ordinary design, checked by our own validation and shown with the exact statements before anything is
    made; relations between tables are kept as plain integer columns, not enforced."""
    principal = _principal(request, PERMISSION)
    data = await read_upload(request)
    registry = _registry(request)
    try:
        service = registry.database_creator(profile)
    except RegistryNotFound as exc:
        raise _err(404, "profile_not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "creation_not_available", str(exc)) from exc
    try:
        image = read_image(filename, data)
    except UploadRefused as exc:
        return {"status": "refused", "reason": str(exc), "plan_id": None}
    instruction = (question or "").strip()[:2000] or "Design the database shown in the attached image."
    try:
        read = await run_in_threadpool(service.designer.plan, instruction, [image.as_llm()])
    except AllProvidersFailedError as exc:
        logger.warning("Database design from an image: no LLM provider answered: %s", exc)
        raise _err(503, "llm_unavailable", LLM_UNAVAILABLE) from exc
    if read.design is None:
        return {"status": "unsupported", "reason": read.reason or "The image could not be turned into a design.",
                "plan_id": None}
    payload = {"question": f"[image] {image.filename}", "profile": service.profile,
               "design": design_to_json(read.design),
               "image": {"name": image.filename, "sha256": image.sha256, "provider": read.provider},
               "database": (database or "").strip() or None, "table": (table or "").strip() or None}
    pipeline = get_pipeline(request.app)
    try:
        task = pipeline.submit(CREATE_OPERATION, payload, principal)
    except PlanUnsupported as exc:
        return {"status": "refused" if exc.details.get("kind") == "refused" else "unsupported",
                "reason": exc.reason, "plan_id": None}
    except PermissionDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks the {PERMISSION} permission.") from exc
    except TooManyTasks as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        logger.error("Database design from an image failed.", exc_info=exc)
        raise _err(503, "creation_failed", "A plan could not be produced right now.") from exc
    service.audit("planned", plan_id=task.task_id, actor=principal.actor_id, plan=task.plan["summary"],
                  file={"name": image.filename, "sha256": image.sha256, "kind": "image"})
    view = _view(task, pipeline)
    view["file"] = {"name": image.filename, "kind": "image", "sent_to_model": True}
    return {"status": "planned", **view}


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
