from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from api.database_engine_requests import (
    EngineRequestConflict, EngineRequestError, EngineRequestNotFound, EngineRequestStore,
)
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.security.authentication import AuthenticationError, get_attached_principal
from api.security.authorization import AuthorizationDenied, AuthorizationPolicy

router = APIRouter(prefix="/database-engine-requests", tags=["database-engine-requests"])
REQUEST_PERMISSION = "database:engine:request"
REVIEW_PERMISSION = "database:engine:review"
_init_lock = threading.Lock()


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": code, "message": message})


def _principal(request: Request):
    try:
        principal = get_attached_principal(request)
    except AuthenticationError as exc:
        raise _err(401, "authentication_required",
                   "A verified authenticated principal is required.") from exc
    policy = getattr(request.app.state, "authorization_policy", None)
    if not isinstance(policy, AuthorizationPolicy):
        raise _err(503, "authorization_policy_unavailable",
                   "The application authorization policy is not configured.")
    return principal, policy


def _require(policy: AuthorizationPolicy, principal, permission: str) -> None:
    try:
        policy.require(principal, permission)
    except AuthorizationDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks {permission}.") from exc


def _can(policy: AuthorizationPolicy, principal, permission: str) -> bool:
    try:
        policy.require(principal, permission)
        return True
    except AuthorizationDenied:
        return False


def _store(request: Request) -> EngineRequestStore:
    state = request.app.state
    store = getattr(state, "database_engine_requests", None)
    if isinstance(store, EngineRequestStore):
        return store
    registry = getattr(state, "database_runtime_registry", None)
    if not isinstance(registry, DatabaseRuntimeRegistry):
        raise _err(503, "engine_requests_unavailable",
                   "Database engine requests are not configured.")
    with _init_lock:
        store = getattr(state, "database_engine_requests", None)
        if not isinstance(store, EngineRequestStore):
            store = EngineRequestStore(registry.work_dir / "engine_requests.sqlite3",
                                       supported=frozenset(registry.supported_engine_names()))
            state.database_engine_requests = store
    return store


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except EngineRequestNotFound as exc:
        raise _err(404, "not_found", str(exc)) from exc
    except EngineRequestConflict as exc:
        raise _err(409, "conflict", str(exc)) from exc
    except EngineRequestError as exc:
        raise _err(422, "invalid_request", str(exc)) from exc


@router.post("", status_code=201)
def create_request(request: Request, payload: dict[str, Any]):
    """Ask for a database engine that is not supported yet (data only)."""
    principal, policy = _principal(request)
    _require(policy, principal, REQUEST_PERMISSION)
    if set(payload) - {"engine", "reason"}:
        raise _err(422, "invalid_request", "Only engine and reason are accepted.")
    return _call(_store(request).create, payload.get("engine"), payload.get("reason"),
                 requested_by=principal.actor_id)


@router.get("")
def list_requests(request: Request):
    """Reviewers see every request; everyone else sees only their own."""
    principal, policy = _principal(request)
    reviewer = _can(policy, principal, REVIEW_PERMISSION)
    if not reviewer:
        _require(policy, principal, REQUEST_PERMISSION)
    store = _store(request)
    return {"requests": store.list(requested_by=None if reviewer else principal.actor_id),
            "scope": "all" if reviewer else "own"}


@router.patch("/{request_id}")
def review_request(request_id: str, request: Request, payload: dict[str, Any]):
    principal, policy = _principal(request)
    _require(policy, principal, REVIEW_PERMISSION)
    if set(payload) - {"status", "note"}:
        raise _err(422, "invalid_request", "Only status and note are accepted.")
    return _call(_store(request).review, request_id, payload.get("status"),
                 reviewed_by=principal.actor_id, note=payload.get("note", ""))


__all__ = ["router", "REQUEST_PERMISSION", "REVIEW_PERMISSION"]
