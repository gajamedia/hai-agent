from __future__ import annotations

"""Create a new database (and tables) from a request: preview, fresh login, exact phrase, audit.

The server profile decides whether this is possible at all (a separate account that can only CREATE
databases whose name starts with the profile's prefix) and what the prefix is. Nothing is ever dropped
or overwritten here.
"""

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from agent.workflow.database_creation import CreationRefused
from api.data_change_plans import ChangePlanConflict, ChangePlanNotFound, ChangePlanStore
from api.database_runtime_registry import RegistryError, RegistryNotFound
from api.routes.data_changes import (
    _err, _principal, _registry, _require_recent_login,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/database-creation", tags=["database-creation"])
PERMISSION = "database:create:manage"


def _plans(request: Request) -> ChangePlanStore:
    state = request.app.state
    store = getattr(state, "database_creation_plans", None)
    if not isinstance(store, ChangePlanStore):
        store = ChangePlanStore(max_plans=50)
        state.database_creation_plans = store
    return store


def _view(stored, store) -> dict[str, Any]:
    return {"plan_id": stored.plan_id, "profile": stored.database, "question": stored.question,
            "plan": stored.summary, "statements": stored.preview["statements"],
            "confirm_phrase": stored.confirm_phrase, "state": stored.state,
            "expires_in_seconds": store.seconds_left(stored)}


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
    try:
        service = _registry(request).database_creator(profile)
    except RegistryNotFound as exc:
        raise _err(404, "profile_not_found", str(exc)) from exc
    except RegistryError as exc:
        raise _err(422, "creation_not_available", str(exc)) from exc
    try:
        result = service.designer.plan(question)
    except ValueError as exc:
        raise _err(422, "invalid_question", str(exc)) from exc
    except Exception as exc:
        logger.exception("Database design failed.")
        raise _err(503, "creation_failed", "A plan could not be produced right now.") from exc
    if result.design is None:
        return {"status": "unsupported", "reason": result.reason, "plan_id": None}
    try:
        statements = service.preview(result.design)
    except CreationRefused as exc:
        return {"status": "refused", "reason": str(exc), "plan_id": None}
    except Exception as exc:
        logger.exception("Database creation preview failed.")
        raise _err(503, "creation_failed", "The plan could not be checked right now.") from exc
    store = _plans(request)
    try:
        stored = store.create(
            owner=principal.actor_id, database=service.profile, question=result.question,
            request=result.design, expected_rows=len(result.design.tables),
            summary=service.summarize(result.design), preview={"statements": statements},
            confirm_phrase=service.confirm_phrase(result.design), requires_approval=False, service=service)
    except ChangePlanConflict as exc:
        raise _err(429, "too_many_plans", str(exc)) from exc
    service.audit("planned", plan_id=stored.plan_id, actor=principal.actor_id, plan=stored.summary)
    return {"status": "planned", **_view(stored, store)}


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str, request: Request):
    principal = _principal(request, PERMISSION)
    store = _plans(request)
    try:
        stored = store.get(plan_id, principal.actor_id)
    except ChangePlanNotFound as exc:
        raise _err(404, "plan_not_found", str(exc)) from exc
    return {"status": "planned" if stored.state == "planned" else stored.state, **_view(stored, store)}


@router.delete("/plans/{plan_id}")
def cancel_plan(plan_id: str, request: Request):
    principal = _principal(request, PERMISSION)
    try:
        _plans(request).cancel(plan_id, principal.actor_id)
    except ChangePlanNotFound as exc:
        raise _err(404, "plan_not_found", str(exc)) from exc
    except ChangePlanConflict as exc:
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
        raise _err(422, "confirmation_mismatch", f"Type exactly: {stored.confirm_phrase}")
    try:
        created = stored.service.execute(stored.request, actor=principal.actor_id,
                                         plan_id=stored.plan_id, question=stored.question)
    except CreationRefused as exc:
        store.release(stored)
        raise _err(409, "creation_refused", str(exc)) from exc
    except Exception as exc:
        store.mark_unknown(stored)
        logger.exception("Database creation failed.")
        raise _err(503, "creation_failed",
                   "Creating the database failed part way. Check the server before trying again; "
                   "nothing is dropped automatically.") from exc
    result = {"status": "created", **created}
    store.finish(stored, result)
    return {**result, "replayed": False}


__all__ = ["router"]
