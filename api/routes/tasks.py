"""The person's own task history: what they asked the agent to do, where each task stands, and how it went.

Read-only. It lists tasks of every feature that runs on the platform pipeline (reports, data changes,
and whatever comes next), shows a timeline from the audit trail, and links to the report file when
there is one. It never shows another person's task, never shows a task for an operation the person
may no longer use, and never shows data the pipeline has already erased.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agent.platform import Operation, TaskNotFound, TaskRecord, TaskStatus
from api.platform_runtime import get_pipeline
from api.security.authentication import AuthenticationError, get_attached_principal

router = APIRouter(prefix="/tasks", tags=["tasks"])

# Filter names the browser uses -> the statuses they stand for.
_GROUPS = {
    "waiting": [TaskStatus.PLANNED, TaskStatus.AWAITING_APPROVAL, TaskStatus.APPROVED],
    "running": [TaskStatus.EXECUTING, TaskStatus.VERIFYING],
    "done": [TaskStatus.COMPLETED, TaskStatus.UNVERIFIED],
    "problem": [TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.REJECTED],
}
# What of a task's stored outcome may be shown (metadata only; report rows are never stored).
_RESULT_KEYS = ("status", "rows_affected", "operation", "table", "row_count", "kind", "limit_reached",
                "workflow_id", "error_code")
# What of an audit record's detail may be shown.
_DETAIL_KEYS = ("from", "to", "verdict", "self_approved")
_MAX_LIMIT = 200


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": code, "message": message})


def _principal(request: Request):
    try:
        return get_attached_principal(request)
    except AuthenticationError as exc:
        raise _error(401, "authentication_required",
                     "A verified authenticated principal is required.") from exc


def _title(op: Operation, task: TaskRecord) -> str:
    if op.describe is None:
        return op.label or op.id
    try:
        return op.describe(task.plan or {}, task.result) or op.label or op.id
    except Exception:  # a describer must never break the list
        return op.label or op.id


def _view(task: TaskRecord, op: Operation, pipeline) -> dict[str, Any]:
    waiting = task.status in (TaskStatus.PLANNED, TaskStatus.AWAITING_APPROVAL, TaskStatus.APPROVED)
    verdict = (task.verification or {}).get("verdict")
    return {
        "task_id": task.task_id, "operation": op.id, "label": op.label or op.id,
        "title": _title(op, task), "status": task.status.value,
        "created_at": task.created_at, "updated_at": task.updated_at,
        "expires_in_seconds": pipeline.seconds_left(task) if waiting else None,
        "approved_by": task.approved_by, "verification": verdict, "error": task.error,
    }


def _download(task: TaskRecord) -> dict[str, Any] | None:
    artifact = (task.result or {}).get("artifact") or {}
    workflow_id = (task.result or {}).get("workflow_id")
    if not (workflow_id and artifact.get("filename")):
        return None
    return {"url": f"/api/v1/table-reports/reports/{workflow_id}/download",
            "filename": artifact.get("filename"), "media_type": artifact.get("media_type"),
            "size_bytes": artifact.get("size_bytes")}


@router.get("")
def list_tasks(request: Request, status: str | None = None, limit: int = 50):
    """Your tasks, newest first. ``status``: waiting, running, done or problem (default: all)."""
    principal = _principal(request)
    if status is not None and status not in _GROUPS:
        raise _error(422, "invalid_request", f"status must be one of {', '.join(_GROUPS)}.")
    if isinstance(limit, bool) or not 1 <= limit <= _MAX_LIMIT:
        raise _error(422, "invalid_request", f"limit must be between 1 and {_MAX_LIMIT}.")
    pipeline = get_pipeline(request.app)
    items, has_more = pipeline.list_tasks(
        principal, statuses=_GROUPS[status] if status else None, limit=limit)
    return {"tasks": [_view(task, op, pipeline) for task, op in items], "has_more": has_more,
            "server_time": time.time()}


@router.get("/{task_id}")
def get_task(task_id: str, request: Request):
    """One of your tasks with its timeline. Anyone else's, or an unknown id, is 'not found'."""
    principal = _principal(request)
    pipeline = get_pipeline(request.app)
    try:
        task = pipeline.get_task(task_id, principal)
        if task.owner != principal.actor_id:
            raise TaskNotFound("task not found")
        op = pipeline.registry.lookup(task.operation_id)
        if not pipeline._may_use(principal, op):
            raise TaskNotFound("task not found")
        records = pipeline.timeline(task_id, principal)
    except (TaskNotFound, KeyError, LookupError) as exc:
        raise _error(404, "task_not_found", "Task not found.") from exc
    timeline = [{"at": r["at"], "event": r["event"], "actor": r["actor"],
                 **{k: r["detail"][k] for k in _DETAIL_KEYS if k in (r["detail"] or {})}}
                for r in records]
    return {**_view(task, op, pipeline),
            "result": {k: task.result[k] for k in _RESULT_KEYS if k in (task.result or {})} or None,
            "download": _download(task), "timeline": timeline}


__all__ = ["router"]
