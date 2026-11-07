"""Authenticated read-only schema discovery for an approved database profile."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agent.database import (
    ConnectionManager,
    SQLiteAllowlistError,
    SQLiteSchemaDiscoveryError,
    build_readonly_table_specs,
    discover_sqlite_schema,
)

router = APIRouter(prefix="/database-onboarding", tags=["database-onboarding"])


def _manager(request: Request) -> ConnectionManager:
    manager = getattr(request.app.state, "business_database_connection_manager", None)
    if not isinstance(manager, ConnectionManager):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "database_onboarding_unavailable",
                "message": "No application-owned read-only database profile is configured.",
            },
        )
    return manager


@router.get("/schema")
def get_approved_sqlite_schema(request: Request, profile_name: str) -> dict[str, object]:
    """Return SQLite metadata only for a profile configured at app startup.

    This endpoint cannot accept a filesystem path, credentials, SQL, or a
    database type.  It only uses an application-owned read-only profile.
    """
    try:
        return discover_sqlite_schema(_manager(request), profile_name=profile_name).to_dict()
    except SQLiteSchemaDiscoveryError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "sqlite_schema_discovery_failed", "message": str(exc)},
        ) from exc


@router.post("/allowlists")
def save_pending_allowlist(request: Request, payload: dict[str, Any]) -> dict[str, object]:
    """Validate and retain a user-selected allowlist for bootstrap review.

    The active gateway is intentionally not replaced at runtime.  The stored
    configuration must be reviewed and supplied to the application's gateway
    factory on its next controlled startup.
    """
    unknown = set(payload) - {"profile_name", "tables"}
    profile_name = payload.get("profile_name")
    selections = payload.get("tables")
    if unknown or not isinstance(profile_name, str) or not profile_name.strip() or not isinstance(selections, Mapping):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_database_allowlist",
                "message": "profile_name and tables must be supplied; unsupported fields are not allowed.",
            },
        )
    try:
        schema = discover_sqlite_schema(_manager(request), profile_name=profile_name.strip())
        specs = build_readonly_table_specs(schema, selections)
    except (SQLiteSchemaDiscoveryError, SQLiteAllowlistError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_database_allowlist", "message": str(exc)},
        ) from exc

    state = request.app.state
    lock = getattr(state, "database_onboarding_lock", None)
    if not callable(getattr(lock, "acquire", None)):
        # FastAPI's app state can also be used by a minimal test app that has
        # not passed through the main lifespan; initialize safely in that case.
        lock = RLock()
        state.database_onboarding_lock = lock
    if not isinstance(getattr(state, "database_onboarding_allowlists", None), dict):
        state.database_onboarding_allowlists = {}
    actor = getattr(getattr(request.state, "authenticated_principal", None), "actor_id", None)
    record = {
        "profile_name": schema.profile_name,
        "tables": {name: list(spec.columns) for name, spec in specs.items()},
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "saved_by": actor,
        "activation": "pending_controlled_restart",
    }
    with lock:
        state.database_onboarding_allowlists[schema.profile_name] = record
    return record


__all__ = ["router"]
