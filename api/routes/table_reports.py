from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from api.security.authentication import AuthenticationError, get_attached_principal
from api.security.authorization import AuthorizationDenied, AuthorizationPolicy
from api.security.trusted_execution_context import TrustedExecutionContext
from agent.llm.fallback_provider import AllProvidersFailedError
from agent.workflow.table_report_service import TableReportService
from api.database_runtime_registry import (
    DatabaseRuntimeRegistry, RegistryError, RegistryNotFound,
)
from agent.platform import (
    ExecutionFailed, PermissionDenied, PlanError, PlanUnsupported, TaskConflict, TaskExpired, TaskNotFound,
    TaskStatus, TooManyTasks,
)
from agent.workflow.table_report_service import AskOutcome, REPORT_FORMATS
from api.features.database_reports import REPORT_OPERATION, outcome_from_metadata
from api.platform_runtime import get_pipeline, report_outcomes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/table-reports", tags=["table-reports"])
_PERMISSION = "report:table:run"


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": code, "message": message})


def _principal(request: Request):
    try:
        principal = get_attached_principal(request)
    except AuthenticationError as exc:
        raise _error(401, "authentication_required",
                     "A verified authenticated principal is required.") from exc
    policy = getattr(request.app.state, "authorization_policy", None)
    if not isinstance(policy, AuthorizationPolicy):
        raise _error(503, "authorization_policy_unavailable",
                     "The application authorization policy is not configured.")
    try:
        policy.require(principal, _PERMISSION)
    except AuthorizationDenied as exc:
        raise _error(403, "permission_denied",
                     f"The principal lacks the {_PERMISSION} permission.") from exc
    return principal


def _parse(payload: dict[str, Any]) -> tuple[str, str | None, str | None]:
    unknown = set(payload) - {"question", "database", "format"}
    question, database = payload.get("question"), payload.get("database")
    report_format = payload.get("format")
    if (unknown or not isinstance(question, str)
            or not (database is None or isinstance(database, str))
            or not (report_format is None or report_format in REPORT_FORMATS)):
        raise _error(422, "invalid_request",
                     "Body must be {\"question\": \"<text>\", \"database\": \"<name, optional>\", "
                     "\"format\": \"<" + "|".join(REPORT_FORMATS) + ", optional>\"}.")
    return question, database, report_format


def _service(request: Request, database: str | None) -> tuple[TableReportService, str | None]:
    """Resolve the report service and the exact connection name it belongs to."""
    registry = getattr(request.app.state, "database_runtime_registry", None)
    if isinstance(registry, DatabaseRuntimeRegistry):
        try:
            name = registry.resolve_name(database)
            return registry.service(name), name
        except RegistryNotFound as exc:
            raise _error(404, "database_not_found", str(exc)) from exc
        except RegistryError as exc:
            raise _error(422, "database_not_ready", str(exc)) from exc
    service = getattr(request.app.state, "table_report_service", None)
    if not isinstance(service, TableReportService):
        raise _error(503, "table_report_unavailable", "Table reporting is not configured.")
    return service, None


def _body(request: Request, outcome, principal=None, database: str | None = None) -> dict[str, Any]:
    # Unsupported questions are a valid answer, not a server fault.
    download = None
    registry = getattr(request.app.state, "database_runtime_registry", None)
    if (outcome.artifact and principal is not None and database
            and isinstance(registry, DatabaseRuntimeRegistry)):
        try:
            registry.record_report(owner=principal.actor_id, database=database,
                                   workflow_id=outcome.workflow_id, artifact=outcome.artifact)
            download = {
                "url": f"/api/v1/table-reports/reports/{outcome.workflow_id}/download",
                "filename": outcome.artifact.get("filename"),
                "media_type": outcome.artifact.get("media_type"),
                "size_bytes": outcome.artifact.get("size_bytes"),
            }
        except Exception:  # the on-screen result must not depend on the download index
            logger.warning("Could not index the report file.", exc_info=True)
    return {
        "download": download,
        "status": outcome.status, "question": outcome.question, "reason": outcome.reason,
        "workflow_id": outcome.workflow_id, "error_code": outcome.error_code,
        "failed_task": outcome.failed_task_key, "artifact": outcome.artifact,
        "plan": outcome.plan, "provider": outcome.provider,
        "rows": jsonable_encoder(list(outcome.rows)),
        "kind": outcome.kind, "row_count": outcome.row_count,
        "limit_reached": outcome.limit_reached,
    }


def _pipeline(request: Request):
    return get_pipeline(request.app)


def _plan_error(exc: PlanError) -> HTTPException:
    cause = exc.__cause__
    if isinstance(cause, ValueError):
        return _error(422, "invalid_question", str(cause))
    if isinstance(cause, AllProvidersFailedError):
        logger.warning("Table report planning: no LLM provider answered: %s", cause)
        return _error(503, "llm_unavailable", "The AI model did not answer in time, so no plan was made. Try again "
                      "in a minute; if it keeps happening an administrator can run scripts/check_llm_provider.py.")
    logger.error("Table report planning failed.", exc_info=exc)
    return _error(503, "table_report_failed", "A plan could not be produced right now.")


def _submit(request: Request, principal, question: str, database: str | None,
            report_format: str | None):
    """Plan through the platform pipeline. Returns the task; raises PlanUnsupported when there is no plan."""
    try:
        return _pipeline(request).submit(
            REPORT_OPERATION, {"question": question, "database": database, "format": report_format},
            principal)
    except PermissionDenied as exc:
        raise _error(403, "permission_denied", f"The principal lacks the {_PERMISSION} permission.") from exc
    except TooManyTasks as exc:
        raise _error(429, "too_many_plans", str(exc)) from exc
    except PlanError as exc:
        raise _plan_error(exc) from exc


_PROPOSED_RELATIONS_REASON = (
    "This question needs tables to be combined in a way nobody approved yet. "
    "Make a plan, check it, and run it to approve the relation.")


def _finished_body(request: Request, task, principal) -> dict[str, Any]:
    """The response for a finished task: the exact earlier body if this process still has it,
    otherwise rebuilt from the stored facts (no rows; they are in the report file)."""
    cache = report_outcomes(request.app)
    body = cache.body(task.task_id)
    if body is not None:
        return body
    outcome = cache.outcome(task.task_id) or outcome_from_metadata(task.plan, task.result)
    return _body(request, outcome, principal, task.plan.get("database"))


@router.post("")
def ask_table_report(request: Request, payload: dict[str, Any] = Body(...)):
    """Plan and run in one call: a validated, read-only table report."""
    principal = _principal(request)
    question, database, report_format = _parse(payload)
    _, name = _service(request, database)
    try:
        task = _submit(request, principal, question, name or database, report_format)
    except PlanUnsupported as exc:
        details = exc.details
        outcome = AskOutcome("unsupported", question, reason=exc.reason, provider=details.get("provider"),
                             attempts=details.get("attempts", 0))
        return _body(request, outcome, principal, name)
    except HTTPException as exc:
        if exc.status_code in (422, 403, 429):
            raise
        raise _error(503, "table_report_failed", "The report could not be produced right now.") from exc
    pipeline = _pipeline(request)
    if task.plan["proposed_relations"]:
        # Nobody has looked at the relation the AI found; only the review-then-run flow may use it.
        pipeline.cancel(task.task_id, principal)
        outcome = AskOutcome("unsupported", question, provider=task.plan.get("provider"),
                             attempts=task.plan.get("attempts", 0), reason=_PROPOSED_RELATIONS_REASON)
        return _body(request, outcome, principal, name)
    try:
        pipeline.run(task.task_id, principal)
    except Exception as exc:  # LLM unavailable, store failure, ...
        logger.exception("Table report request failed.")
        raise _error(503, "table_report_failed",
                     "The report could not be produced right now.") from exc
    done = pipeline.get_task(task.task_id, principal)
    outcome = report_outcomes(request.app).outcome(done.task_id) or outcome_from_metadata(done.plan, done.result)
    return _body(request, outcome, principal, name)


@router.post("/plans", status_code=201)
def create_plan(request: Request, payload: dict[str, Any] = Body(...)):
    """Phase 1: generate a plan for review. No data is read."""
    principal = _principal(request)
    question, database, report_format = _parse(payload)
    _, name = _service(request, database)
    try:
        task = _submit(request, principal, question, name or database, report_format)
    except PlanUnsupported as exc:
        return JSONResponse(status_code=200, content={
            "status": "unsupported", "question": exc.details.get("question", question), "reason": exc.reason,
            "provider": exc.details.get("provider"), "plan_id": None,
            "needs_join": bool(exc.details.get("needs_join", False))})
    plan = task.plan
    return {"status": "planned", "plan_id": task.task_id, "question": plan["question"],
            "database": plan["database"], "plan": plan["summary"], "provider": plan["provider"],
            "expires_in_seconds": _pipeline(request).seconds_left(task)}


_DONE = (TaskStatus.COMPLETED, TaskStatus.UNVERIFIED, TaskStatus.FAILED)


@router.post("/plans/{plan_id}/run")
def run_plan(plan_id: str, request: Request, payload: dict[str, Any] = Body(default={})):
    """Phase 2: run exactly the plan that was reviewed. Supports Idempotency-Key."""
    principal = _principal(request)
    if payload:
        raise _error(422, "invalid_request", "The plan cannot be changed; send an empty body.")
    key = request.headers.get("Idempotency-Key")
    if key is not None:
        key = key.strip()
        if not key or len(key) > 255:
            raise _error(400, "invalid_idempotency_key",
                         "Idempotency-Key must contain 1 to 255 non-whitespace characters.")
    pipeline = _pipeline(request)
    try:
        task = pipeline.get_task(plan_id, principal)
    except TaskNotFound as exc:
        raise _error(404, "plan_not_found", "plan not found or expired") from exc
    if task.operation_id != REPORT_OPERATION or task.status in (TaskStatus.CANCELLED, TaskStatus.REJECTED):
        raise _error(404, "plan_not_found", "plan not found or expired")

    if task.status not in _DONE:
        # The connection must be the very one the plan was made for (same approvals).
        _service(request, task.plan["database"])  # 404 / 422 when it is gone or not ready
        registry = getattr(request.app.state, "database_runtime_registry", None)
        if isinstance(registry, DatabaseRuntimeRegistry):
            if registry.service_token(task.plan["database"]) != task.plan["connection"]:
                raise _error(409, "connection_changed",
                             "The database connection changed since this plan was made; ask again.")
    try:
        outcome = pipeline.run(plan_id, principal, idempotency_key=key)
    except (TaskNotFound, TaskExpired) as exc:
        raise _error(404, "plan_not_found", "plan not found or expired") from exc
    except TaskConflict as exc:
        raise _error(409, "plan_not_runnable", str(exc)) from exc
    except PermissionDenied as exc:
        raise _error(403, "permission_denied", f"The principal lacks the {_PERMISSION} permission.") from exc
    except ExecutionFailed as exc:
        logger.error("Table report run failed.", exc_info=exc)
        raise _error(503, "table_report_failed",
                     "The report could not be produced right now.") from exc
    if outcome.replayed:
        return {**_finished_body(request, outcome.task, principal), "replayed": True}
    cache = report_outcomes(request.app)
    result = _body(request, cache.outcome(plan_id) or outcome_from_metadata(outcome.task.plan, outcome.task.result),
                   principal, task.plan["database"])
    cache.put_body(plan_id, result)
    return {**result, "replayed": False}


_EXTENSIONS = {"application/json": "json", "text/csv": "csv", "text/html": "html",
               "application/pdf": "pdf",
               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx"}


@router.get("/reports/{workflow_id}/download")
def download_report(workflow_id: str, request: Request):
    """Download a report file you produced (Excel, PDF, CSV, HTML or JSON)."""
    principal = _principal(request)
    registry = getattr(request.app.state, "database_runtime_registry", None)
    found = (registry.find_report(principal.actor_id, workflow_id)
             if isinstance(registry, DatabaseRuntimeRegistry) else None)
    if found is None:
        raise _error(404, "report_not_found", "No such report file.")
    media = found["media_type"].split(";")[0].strip()
    extension = _EXTENSIONS.get(media)
    if extension is None:
        raise _error(404, "report_not_found", "No such report file.")
    return Response(
        content=found["path"].read_bytes(), media_type=found["media_type"],
        headers={
            "Content-Disposition": f'attachment; filename="report-{workflow_id[:8]}.{extension}"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
            # Even if a browser rendered the file, it could not run script or load anything.
            "Content-Security-Policy": "sandbox; default-src 'none'",
        })


@router.delete("/plans/{plan_id}")
def cancel_plan(plan_id: str, request: Request):
    principal = _principal(request)
    try:
        _pipeline(request).cancel(plan_id, principal)
    except TaskNotFound as exc:
        raise _error(404, "plan_not_found", "plan not found or expired") from exc
    except TaskConflict as exc:
        raise _error(409, "plan_not_runnable", str(exc)) from exc
    return {"plan_id": plan_id, "status": "cancelled"}


__all__ = ["router"]
