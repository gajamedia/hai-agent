from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from api.database_runtime_registry import (
    DatabaseRuntimeRegistry, RegistryError, RegistryNotFound,
)
from api.security.authentication import AuthenticationError, get_attached_principal
from api.security.authorization import AuthorizationDenied, AuthorizationPolicy

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/database-connections", tags=["database-connections"])
PERMISSION = "database:connection:manage"


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": code, "message": message})


def _guard(request: Request):
    """Authenticate, authorize, and return (principal, registry)."""
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
        policy.require(principal, PERMISSION)
    except AuthorizationDenied as exc:
        raise _err(403, "permission_denied", f"The principal lacks {PERMISSION}.") from exc
    registry = getattr(request.app.state, "database_runtime_registry", None)
    if not isinstance(registry, DatabaseRuntimeRegistry):
        raise _err(503, "database_connections_unavailable",
                   "Runtime database connections are not configured.")
    return principal, registry


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RegistryNotFound as exc:
        raise _err(404, "not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "invalid_request", str(exc)) from exc


@router.post("")
def connect_database(request: Request, payload: dict[str, Any]):
    """Step 1: connect a SQLite file (by name in the data folder) or a named profile."""
    principal, registry = _guard(request)
    if set(payload) - {"name", "filename", "profile", "type", "database"}:
        raise _err(422, "invalid_request",
                   "Only name, filename (sqlite) or profile (server-defined) and type are accepted.")
    has_file, has_profile = "filename" in payload, "profile" in payload
    if has_file == has_profile:
        raise _err(422, "invalid_request", "Provide exactly one of filename or profile.")
    engine = "sqlite" if has_file else registry.profile_engine(payload.get("profile"))
    if engine is None:
        raise _err(422, "unknown_profile", "That database profile does not exist.")
    if payload.get("type", engine) != engine:
        raise _err(422, "unsupported_database_type",
                   f"'{payload.get('type')}' does not match the source you provided ({engine}).")
    source = payload.get("filename" if has_file else "profile")
    database = payload.get("database")
    if database is not None:
        # A database made by this application (name starts with the profile's prefix); the engine
        # refuses any other name, so this can never point a profile at an unrelated database.
        if has_file or not isinstance(database, str) or "/" in database or not isinstance(source, str):
            raise _err(422, "invalid_request", "database can only accompany a profile, as a plain name.")
        source = f"{source}/{database}"
    return _call(registry.connect, payload.get("name"), engine=engine, source=source,
                 actor_id=principal.actor_id)


@router.get("/profiles")
def list_profiles(request: Request):
    """Server-defined connection profiles (name and type only)."""
    _, registry = _guard(request)
    return {"profiles": registry.profiles(), "engines": registry.available_engines(),
            "server_registration": registry.can_register_servers,
            "account_setup": registry.can_setup_accounts}


@router.post("/servers")
def add_server(request: Request, payload: dict[str, Any]):
    """Add a database server from a form: tested first, saved only if the test passes."""
    _, registry = _guard(request)
    return _call(registry.register_server, payload)


def _guard_admin(request: Request):
    """Account setup with an administrator password: the manage AND the create permission, and a fresh login."""
    principal, registry = _guard(request)
    try:
        request.app.state.authorization_policy.require(principal, "database:create:manage")
    except AuthorizationDenied as exc:
        raise _err(403, "permission_denied", "The principal lacks database:create:manage.") from exc
    from api.routes.data_changes import _require_recent_login
    _require_recent_login(principal)
    return registry


@router.post("/servers/inspect")
def inspect_server(request: Request, payload: dict[str, Any]):
    """Look at a server with a one-time administrator login. The password is used for this request only."""
    registry = _guard_admin(request)
    return _call(registry.inspect_server, payload)


@router.post("/servers/auto")
def auto_server(request: Request, payload: dict[str, Any]):
    """Create the limited accounts with a one-time administrator login and register the server."""
    registry = _guard_admin(request)
    return _call(registry.provision_server, payload)


@router.delete("/servers/{name}")
def remove_server(name: str, request: Request):
    _, registry = _guard(request)
    _call(registry.remove_server, name)
    return {"name": name, "status": "removed"}


@router.get("")
def list_connections(request: Request):
    _, registry = _guard(request)
    return {"connections": registry.list()}


@router.get("/{name}/schema")
def connection_schema(name: str, request: Request):
    _, registry = _guard(request)
    return _call(registry.schema, name)


@router.post("/{name}/approve")
def approve_tables(name: str, request: Request, payload: dict[str, Any]):
    """Step 2: the user approves tables/columns; only then can reports run."""
    principal, registry = _guard(request)
    if set(payload) - {"tables"}:
        raise _err(422, "invalid_request", "Only 'tables' is accepted.")
    return _call(registry.activate, name, payload.get("tables"), actor_id=principal.actor_id)


@router.post("/{name}/refresh")
def refresh_connection(name: str, request: Request):
    """Read the schema again, e.g. after the agent added tables. Approvals are kept."""
    principal, registry = _guard(request)
    return _call(registry.refresh, name, actor_id=principal.actor_id)


@router.delete("/{name}")
def disconnect_database(name: str, request: Request):
    _, registry = _guard(request)
    _call(registry.disconnect, name)
    return {"name": name, "status": "disconnected"}


__all__ = ["router", "PERMISSION"]
