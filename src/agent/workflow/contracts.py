from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    PLANNED = "planned"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Workflow:
    workflow_id: str
    idempotency_key: str
    goal: str
    status: WorkflowStatus
    version: int
    created_at: str
    updated_at: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "idempotency_key": self.idempotency_key,
            "goal": self.goal,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorkflowTask:
    task_id: str
    workflow_id: str
    task_key: str
    task_type: str
    description: str
    status: TaskStatus
    version: int
    attempt_count: int
    max_attempts: int
    input_data: Mapping[str, Any]
    result_data: Mapping[str, Any] | None
    error_code: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "task_key": self.task_key,
            "task_type": self.task_type,
            "description": self.description,
            "status": self.status.value,
            "version": self.version,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "input_data": dict(self.input_data),
            "result_data": None if self.result_data is None else dict(self.result_data),
            "error_code": self.error_code,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class WorkflowEvent:
    event_id: str
    workflow_id: str
    task_id: str | None
    event_type: str
    from_status: str | None
    to_status: str
    occurred_at: str
    workflow_version: int | None
    task_version: int | None
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "occurred_at": self.occurred_at,
            "workflow_version": self.workflow_version,
            "task_version": self.task_version,
            "metadata": dict(self.metadata),
        }
