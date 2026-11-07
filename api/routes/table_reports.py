from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.encoders import jsonable_encoder

from api.security.authentication import AuthenticationError, get_attached_principal
from api.security.authorization import AuthorizationDenied, AuthorizationPolicy
from api.security.trusted_execution_context import TrustedExecutionContext
from agent.workflow.table_report_service import TableReportService
from api.database_runtime_registry import (
    DatabaseRuntimeRegistry, RegistryError, RegistryNotFound,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/table-reports", tags=["table-reports"])
_PERMISSION = "report:table:run"


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": code, "message": message})


@router.post("")
def ask_table_report(request: Request, payload: dict[str, Any] = Body(...)):
    """Turn a natural-language question into a validated, read-only table report."""
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

    unknown = set(payload) - {"question", "database"}
    question, database = payload.get("question"), payload.get("database")
    if unknown or not isinstance(question, str) or not (database is None or isinstance(database, str)):
        raise _error(422, "invalid_request",
                     "Body must be {\"question\": \"<text>\", \"database\": \"<name, optional>\"}.")

    registry = getattr(request.app.state, "database_runtime_registry", None)
    if isinstance(registry, DatabaseRuntimeRegistry):
        try:
            service = registry.service(database)
        except RegistryNotFound as exc:
            raise _error(404, "database_not_found", str(exc)) from exc
        except RegistryError as exc:
            raise _error(422, "database_not_ready", str(exc)) from exc
    else:
        service = getattr(request.app.state, "table_report_service", None)
    if not isinstance(service, TableReportService):
        raise _error(503, "table_report_unavailable", "Table reporting is not configured.")

    try:
        outcome = service.ask(question, execution_context=TrustedExecutionContext(principal))
    except ValueError as exc:
        raise _error(422, "invalid_question", str(exc)) from exc
    except Exception as exc:  # LLM unavailable, store failure, ...
        logger.exception("Table report request failed.")
        raise _error(503, "table_report_failed",
                     "The report could not be produced right now.") from exc

    body = {
        "status": outcome.status, "question": outcome.question, "reason": outcome.reason,
        "workflow_id": outcome.workflow_id, "error_code": outcome.error_code,
        "failed_task": outcome.failed_task_key, "artifact": outcome.artifact,
        "plan": outcome.plan, "provider": outcome.provider,
        "rows": jsonable_encoder(list(outcome.rows)),
    }
    # Unsupported questions are a valid answer, not a server fault.
    return body


__all__ = ["router"]
