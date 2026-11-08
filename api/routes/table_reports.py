from __future__ import annotations

import logging
import threading
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
from api.table_report_plans import PlanConflict, PlanNotFound, PlanStore
from agent.workflow.table_report_service import REPORT_FORMATS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/table-reports", tags=["table-reports"])
_PERMISSION = "report:table:run"
_plans_lock = threading.Lock()


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


def _plans(request: Request) -> PlanStore:
    state = request.app.state
    store = getattr(state, "table_report_plans", None)
    if not isinstance(store, PlanStore):
        with _plans_lock:
            store = getattr(state, "table_report_plans", None)
            if not isinstance(store, PlanStore):
                store = PlanStore()
                state.table_report_plans = store
    return store


@router.post("")
def ask_table_report(request: Request, payload: dict[str, Any] = Body(...)):
    """Plan and run in one call: a validated, read-only table report."""
    principal = _principal(request)
    question, database, report_format = _parse(payload)
    service, name = _service(request, database)
    try:
        outcome = service.ask(question, execution_context=TrustedExecutionContext(principal),
                              report_format=report_format)
    except ValueError as exc:
        raise _error(422, "invalid_question", str(exc)) from exc
    except Exception as exc:  # LLM unavailable, store failure, ...
        logger.exception("Table report request failed.")
        raise _error(503, "table_report_failed",
                     "The report could not be produced right now.") from exc
    return _body(request, outcome, principal, name)


@router.post("/plans", status_code=201)
def create_plan(request: Request, payload: dict[str, Any] = Body(...)):
    """Phase 1: generate a plan for review. No data is read."""
    principal = _principal(request)
    question, database, report_format = _parse(payload)
    service, name = _service(request, database)
    try:
        plan = service.plan(question, report_format)
    except ValueError as exc:
        raise _error(422, "invalid_question", str(exc)) from exc
    except AllProvidersFailedError as exc:
        logger.warning("Table report planning: no LLM provider answered: %s", exc)
        raise _error(503, "llm_unavailable", "The AI model did not answer in time, so no plan was made. Try again "
                     "in a minute; if it keeps happening an administrator can run scripts/check_llm_provider.py.") from exc
    except Exception as exc:
        logger.exception("Table report planning failed.")
        raise _error(503, "table_report_failed", "A plan could not be produced right now.") from exc
    if not plan.ok:
        return JSONResponse(status_code=200, content={
            "status": "unsupported", "question": plan.question, "reason": plan.reason,
            "provider": plan.provider, "plan_id": None})
    store = _plans(request)
    try:
        stored = store.create(owner=principal.actor_id, database=name, question=plan.question,
                              plan_result=plan, summary=service.summarize(plan.request),
                              service=service)
    except PlanConflict as exc:
        raise _error(429, "too_many_plans", str(exc)) from exc
    return {"status": "planned", "plan_id": stored.plan_id, "question": stored.question,
            "database": name, "plan": stored.summary, "provider": plan.provider,
            "expires_in_seconds": store.seconds_left(stored)}


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
    store = _plans(request)
    try:
        mode, stored = store.begin_run(plan_id, principal.actor_id, key)
    except PlanNotFound as exc:
        raise _error(404, "plan_not_found", str(exc)) from exc
    except PlanConflict as exc:
        raise _error(409, "plan_not_runnable", str(exc)) from exc
    if mode == "replay":
        return {**stored.result, "replayed": True}

    # The connection must be the very one the plan was made for (same approval).
    try:
        current, _ = _service(request, stored.database)
    except HTTPException:
        store.release(stored)
        raise
    if current is not stored.service:
        store.release(stored)
        raise _error(409, "connection_changed",
                     "The database connection changed since this plan was made; ask again.")
    try:
        outcome = stored.service.run_plan(stored.question, stored.plan_result,
                                          execution_context=TrustedExecutionContext(principal))
    except Exception as exc:
        store.mark_unknown(stored)
        logger.exception("Table report run failed.")
        raise _error(503, "table_report_failed",
                     "The report could not be produced right now.") from exc
    result = _body(request, outcome, principal, stored.database)
    store.finish(stored, result)
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
        _plans(request).cancel(plan_id, principal.actor_id)
    except PlanNotFound as exc:
        raise _error(404, "plan_not_found", str(exc)) from exc
    except PlanConflict as exc:
        raise _error(409, "plan_not_runnable", str(exc)) from exc
    return {"plan_id": plan_id, "status": "cancelled"}


__all__ = ["router"]
