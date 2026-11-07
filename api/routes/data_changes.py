from __future__ import annotations

"""Insert / update / delete, behind five separate protections.

1. its own permissions (``data:change:plan`` / ``execute`` / ``approve``), apart from reading;
2. a write-capable database account that only the server knows, and per-table write approval;
3. a preview and the user's own row limit (the server only sets a ceiling);
4. a fresh login (step-up): the token's ``auth_time`` must be recent, so the person has just typed
   their credentials into Keycloak; this application never sees the password;
5. the exact confirmation phrase, and for the risky operations a second person's approval.

Everything runs in one transaction that is rolled back unless it touches exactly the rows previewed.
"""

import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.encoders import jsonable_encoder

from agent.workflow.data_change import ChangeRefused, DEFAULT_MAX_ROWS_CEILING
from api.data_change_plans import ChangePlanConflict, ChangePlanNotFound, ChangePlanStore
from api.database_runtime_registry import (
    DatabaseRuntimeRegistry, RegistryError, RegistryNotFound,
)
from api.security.authentication import AuthenticationError, get_attached_principal
from api.security.authorization import AuthorizationDenied, AuthorizationPolicy

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-changes", tags=["data-changes"])
PLAN, EXECUTE, APPROVE = "data:change:plan", "data:change:execute", "data:change:approve"
_now = time.time  # replaced in tests


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


def _int_env(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


def settings() -> dict[str, Any]:
    mode = os.getenv("HAI_DATA_CHANGE_APPROVAL", "delete").strip().lower()
    if mode not in {"delete", "all", "none"}:
        mode = "delete"
    return {
        "max_rows_ceiling": _int_env("HAI_DATA_CHANGE_MAX_ROWS", DEFAULT_MAX_ROWS_CEILING, 1, 1_000_000),
        "default_max_rows": 100,
        "approval": mode,
        "step_up_max_age_seconds": _int_env("HAI_STEPUP_MAX_AGE_SECONDS", 120, 30, 900),
    }


def _requires_approval(operation: str) -> bool:
    mode = settings()["approval"]
    return mode == "all" or (mode == "delete" and operation == "delete")


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


def _plans(request: Request) -> ChangePlanStore:
    state = request.app.state
    store = getattr(state, "data_change_plans", None)
    if not isinstance(store, ChangePlanStore):
        store = ChangePlanStore()
        state.data_change_plans = store
    return store


def _view(stored, store, *, with_phrase: bool) -> dict[str, Any]:
    out = {"plan_id": stored.plan_id, "database": stored.database, "question": stored.question,
           "plan": stored.summary, "preview": stored.preview, "expected_rows": stored.expected_rows,
           "requires_approval": stored.requires_approval, "state": stored.state,
           "approved_by": stored.approved_by, "owner": stored.owner,
           "expires_in_seconds": store.seconds_left(stored)}
    if with_phrase:
        out["confirm_phrase"] = stored.confirm_phrase
    return out


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
        result = service.planner.plan(question, max_rows=max_rows)
    except ValueError as exc:
        raise _err(422, "invalid_question", str(exc)) from exc
    except Exception as exc:
        logger.exception("Data change planning failed.")
        raise _err(503, "data_change_failed", "A plan could not be produced right now.") from exc
    if result.request is None:
        return {"status": "unsupported", "reason": result.reason, "plan_id": None}
    try:
        preview = service.preview(result.request)
    except ChangeRefused as exc:
        return {"status": "refused", "reason": str(exc), "plan_id": None,
                "plan": service.summarize(result.request)}
    except Exception as exc:
        logger.exception("Data change preview failed.")
        raise _err(503, "data_change_failed", "The change could not be previewed right now.") from exc
    store = _plans(request)
    try:
        stored = store.create(
            owner=principal.actor_id, database=name, question=result.question, request=result.request,
            expected_rows=preview.expected_rows, summary=service.summarize(result.request),
            preview=jsonable_encoder({"before": preview.before, "after": preview.after,
                                      "note": preview.sample_note}),
            confirm_phrase=service.confirm_phrase(result.request, preview.expected_rows),
            requires_approval=_requires_approval(result.request.operation), service=service)
    except ChangePlanConflict as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    service.audit("planned", plan_id=stored.plan_id, actor=principal.actor_id,
                  plan=stored.summary, expected_rows=stored.expected_rows)
    return {"status": "planned", **_view(stored, store, with_phrase=True)}


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str, request: Request):
    """Show a plan again, for example after the browser returned from signing in again."""
    principal = _principal(request, PLAN)
    store = _plans(request)
    try:
        stored = store.get(plan_id, principal.actor_id)
    except ChangePlanNotFound as exc:
        raise _err(404, "plan_not_found", str(exc)) from exc
    return {"status": "planned" if stored.state in {"planned", "approved"} else stored.state,
            **_view(stored, store, with_phrase=True)}


@router.delete("/plans/{plan_id}")
def cancel_plan(plan_id: str, request: Request):
    principal = _principal(request, PLAN)
    try:
        _plans(request).cancel(plan_id, principal.actor_id)
    except ChangePlanNotFound as exc:
        raise _err(404, "plan_not_found", str(exc)) from exc
    except ChangePlanConflict as exc:
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
    store = _plans(request)
    try:
        mode, stored = store.begin_run(plan_id, principal.actor_id, key)
    except ChangePlanNotFound as exc:
        raise _err(404, "plan_not_found", str(exc)) from exc
    except ChangePlanConflict as exc:
        raise _err(409, "plan_not_runnable", str(exc)) from exc
    if mode == "replay":
        return {**stored.result, "replayed": True}
    if payload["confirm"].strip() != stored.confirm_phrase:
        store.release(stored)
        raise _err(422, "confirmation_mismatch",
                   f"Type exactly: {stored.confirm_phrase}")
    try:
        current = _registry(request).data_changes(stored.database)
    except (RegistryError, HTTPException):
        store.release(stored)
        raise _err(409, "connection_changed",
                   "The database connection changed since this plan was made; plan it again.")
    if current is not stored.service:
        store.release(stored)
        raise _err(409, "connection_changed",
                   "The approved tables changed since this plan was made; plan it again.")
    try:
        done = stored.service.execute(
            stored.request, expected_rows=stored.expected_rows, actor=principal.actor_id,
            approver=stored.approved_by, plan_id=stored.plan_id, question=stored.question)
    except ChangeRefused as exc:
        store.release(stored)
        raise _err(409, "change_refused", str(exc)) from exc
    except Exception as exc:
        store.mark_unknown(stored)
        logger.exception("Data change failed.")
        raise _err(503, "data_change_failed",
                   "The change failed and was rolled back if it had not been committed. Check the data "
                   "before trying again.") from exc
    result = {"status": "committed", "rows_affected": done.rows_affected,
              "operation": stored.summary["operation"], "table": stored.summary["table"],
              "approved_by": stored.approved_by}
    store.finish(stored, result)
    return {**result, "replayed": False}


@router.get("/approvals")
def list_approvals(request: Request):
    """Changes by other people that wait for a second person (never your own)."""
    principal = _principal(request, APPROVE)
    store = _plans(request)
    return {"approvals": [_view(p, store, with_phrase=False)
                          for p in store.waiting_for_approval(principal.actor_id)]}


def _decide(plan_id: str, request: Request, *, reject: bool):
    principal = _principal(request, APPROVE)
    if not reject:
        _require_recent_login(principal)
    store = _plans(request)
    try:
        stored = store.approve(plan_id, principal.actor_id, reject=reject)
    except ChangePlanNotFound as exc:
        raise _err(404, "plan_not_found", str(exc)) from exc
    except ChangePlanConflict as exc:
        raise _err(409, "approval_not_possible", str(exc)) from exc
    stored.service.audit("rejected" if reject else "approved", plan_id=stored.plan_id,
                         approver=principal.actor_id, owner=stored.owner, plan=stored.summary,
                         expected_rows=stored.expected_rows)
    return {"plan_id": stored.plan_id, "status": stored.state}


@router.post("/approvals/{plan_id}/approve")
def approve_change(plan_id: str, request: Request):
    return _decide(plan_id, request, reject=False)


@router.post("/approvals/{plan_id}/reject")
def reject_change(plan_id: str, request: Request):
    return _decide(plan_id, request, reject=True)


__all__ = ["router"]
