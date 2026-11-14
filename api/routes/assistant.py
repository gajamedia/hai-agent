from __future__ import annotations

"""The one-page assistant's front door: which page's pipeline does this message (and file) belong to?

This only *classifies*. It plans nothing and writes nothing: the browser then calls the same plan, preview and
run routes the pages already use, so every permission, fresh login, typed phrase and approval stays exactly as
it is there. The answer also says which intents this person may use at all, so the chat never offers a pipeline
that would only answer "forbidden".
"""

from typing import Any

from fastapi import APIRouter, Body, Request
from starlette.concurrency import run_in_threadpool

from agent.workflow.intent_router import (
    CHANGE_DATA, CREATE_DATABASE, RECORD_DATA, UNCLEAR, VIEW_DATA, classify_intent,
)
from agent.workflow.uploaded_table import UploadRefused, read_table
from api.routes.data_changes import _err, _registry
from api.routes.uploads import read_upload
from api.security.authentication import AuthenticationError, get_attached_principal
from api.security.authorization import AuthorizationPolicy

router = APIRouter(prefix="/assistant", tags=["assistant"])

# The permission each intent's own pipeline asks for (the same ones the middleware maps for those routes).
INTENT_PERMISSION = {
    CREATE_DATABASE: "database:create:manage",
    RECORD_DATA: "data:entry:record",
    CHANGE_DATA: "data:change:plan",
    VIEW_DATA: "report:table:run",
}
MAX_MESSAGE = 2000
ENTRY_PERMISSION = "task:history:read"   # carried by every role that runs a feature


def _who(request: Request):
    try:
        principal = get_attached_principal(request)
    except AuthenticationError as exc:
        raise _err(401, "authentication_required", "A verified authenticated principal is required.") from exc
    policy = getattr(request.app.state, "authorization_policy", None)
    if not isinstance(policy, AuthorizationPolicy):
        raise _err(503, "authorization_policy_unavailable", "The application authorization policy is not configured.")
    if not policy.is_allowed(principal, ENTRY_PERMISSION):
        raise _err(403, "permission_denied", f"The principal lacks the {ENTRY_PERMISSION} permission.")
    return principal, policy


def _answer(request: Request, message: str, parsed) -> dict[str, Any]:
    principal, policy = _who(request)
    allowed = {intent: policy.is_allowed(principal, perm) for intent, perm in INTENT_PERMISSION.items()}
    registry = _registry(request)
    known = registry.entry_databases() if allowed[RECORD_DATA] else []
    verdict = classify_intent(registry.language_model(), message,
                              filename=parsed.filename if parsed else None,
                              headers=parsed.headers if parsed else (), known=known)
    intent = verdict["intent"]
    out: dict[str, Any] = {"intent": intent, "allowed": allowed, "confidence": verdict["confidence"],
                           "database": verdict["database"], "table": verdict["table"], "format": verdict["format"],
                           "question": verdict["question"] or message}
    if intent != UNCLEAR and not allowed[intent]:
        out.update(intent=UNCLEAR, denied=intent)       # understood, but this person may not run it
    if parsed:
        out["file"] = {"name": parsed.filename, "rows": len(parsed.rows), "headers": list(parsed.headers)}
    return out


@router.post("/route")
async def route_message(request: Request, payload: dict[str, Any] = Body(...)):
    message = payload.get("message")
    if set(payload) != {"message"} or not isinstance(message, str) or not message.strip():
        raise _err(422, "invalid_request", 'Body must be {"message": "<text>"}.')
    return await run_in_threadpool(_answer, request, message.strip()[:MAX_MESSAGE], None)


@router.post("/route-file")
async def route_message_with_file(request: Request, filename: str, message: str = "", header_row: int = 1):
    """Same, for a message that comes with a file (raw body). Only the file's name and headers are read here."""
    _who(request)
    data = await read_upload(request)
    try:
        parsed = await run_in_threadpool(read_table, filename, data, header_row)
    except UploadRefused as exc:
        return {"intent": UNCLEAR, "refused": str(exc), "allowed": {}}
    return await run_in_threadpool(_answer, request, message.strip()[:MAX_MESSAGE], parsed)


__all__ = ["router"]
