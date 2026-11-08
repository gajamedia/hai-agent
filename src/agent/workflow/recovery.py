"""Read-only inspection of potentially stale workflow tasks.

This module never changes workflow state, retries work, invokes capabilities,
or accesses business databases. A stale RUNNING task is only a candidate for
operator reconciliation; age alone cannot prove that its worker is dead.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from agent.workflow.contracts import TaskStatus, WorkflowStatus


TRUSTED_CAPABILITY_TASK_TYPE = "trusted_capability.invoke"
ANALYTICAL_TASK_TYPE = "workflow.analytical.execute"
REPORT_TASK_TYPE = "report.generate"


class WorkflowRecoveryInspectionError(RuntimeError):
    """Raised when recovery inspection cannot safely interpret workflow state."""


@dataclass(frozen=True)
class RecoveryInspectionConfig:
    stale_after_seconds: float = 900.0
    max_workflows: int = 100
    max_tasks_per_workflow: int = 1000

    def __post_init__(self) -> None:
        if isinstance(self.stale_after_seconds, bool) or not isinstance(
            self.stale_after_seconds, (int, float)
        ) or self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be a positive number")
        if isinstance(self.max_workflows, bool) or not isinstance(
            self.max_workflows, int
        ) or not 1 <= self.max_workflows <= 1000:
            raise ValueError("max_workflows must be between 1 and 1000")
        if isinstance(self.max_tasks_per_workflow, bool) or not isinstance(
            self.max_tasks_per_workflow, int
        ) or not 1 <= self.max_tasks_per_workflow <= 10000:
            raise ValueError("max_tasks_per_workflow must be between 1 and 10000")


@dataclass(frozen=True)
class WorkflowRecoveryCandidate:
    workflow_id: str
    workflow_status: str
    task_id: str
    task_key: str
    task_type: str
    task_status: str
    task_version: int
    attempt_count: int
    max_attempts: int
    updated_at: str
    age_seconds: float
    classification: str
    recommendation: str
    automatic_retry_allowed: bool
    outcome_known: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_status": self.workflow_status,
            "task_id": self.task_id,
            "task_key": self.task_key,
            "task_type": self.task_type,
            "task_status": self.task_status,
            "task_version": self.task_version,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "updated_at": self.updated_at,
            "age_seconds": self.age_seconds,
            "classification": self.classification,
            "recommendation": self.recommendation,
            "automatic_retry_allowed": self.automatic_retry_allowed,
            "outcome_known": self.outcome_known,
        }


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowRecoveryInspectionError("Task updated_at is missing or invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkflowRecoveryInspectionError("Task updated_at is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WorkflowRecoveryInspectionError("Task updated_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _classify(task_type: str) -> tuple[str, str, bool | None]:
    if task_type == TRUSTED_CAPABILITY_TASK_TYPE:
        return (
            "external_outcome_requires_reconciliation",
            "Inspect canonical execution evidence and the external system before deciding what to do. Never replay automatically.",
            None,
        )
    if task_type == REPORT_TASK_TYPE:
        return (
            "artifact_reconciliation_required",
            "Check the configured artifact store for an already-published artifact before any explicit restart. Never regenerate automatically.",
            False,
        )
    if task_type == ANALYTICAL_TASK_TYPE:
        return (
            "deterministic_work_requires_operator_reconciliation",
            "Verify the succeeded source result and prior task evidence, then explicitly authorize a fresh attempt if no result was committed.",
            False,
        )
    return (
        "unknown_task_requires_manual_review",
        "Task type has no registered recovery contract. Keep blocked from automatic replay and request operator review.",
        None,
    )


class WorkflowRecoveryInspector:
    """Read-only stale-task scanner over the existing workflow store interface."""

    def __init__(self, *, store: Any, config: RecoveryInspectionConfig | None = None):
        if store is None:
            raise TypeError("store is required")
        for name in ("get_workflow", "list_tasks"):
            if not callable(getattr(store, name, None)):
                raise TypeError(f"store must expose callable {name}(...)")
        self.store = store
        self.config = config or RecoveryInspectionConfig()

    def inspect_workflow(
        self, workflow_id: str, *, now: datetime | None = None
    ) -> tuple[WorkflowRecoveryCandidate, ...]:
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise WorkflowRecoveryInspectionError("Workflow does not exist")
        if workflow.status in {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }:
            return ()

        reference_time = now or datetime.now(timezone.utc)
        if reference_time.tzinfo is None or reference_time.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        reference_time = reference_time.astimezone(timezone.utc)

        tasks = self.store.list_tasks(workflow_id)
        if len(tasks) > self.config.max_tasks_per_workflow:
            raise WorkflowRecoveryInspectionError("Workflow task count exceeds inspection limit")

        candidates = []
        for task in tasks:
            if task.status is not TaskStatus.RUNNING:
                continue
            updated = _parse_timestamp(task.updated_at)
            age = (reference_time - updated).total_seconds()
            if age < self.config.stale_after_seconds:
                continue
            classification, recommendation, outcome_known = _classify(task.task_type)
            candidates.append(
                WorkflowRecoveryCandidate(
                    workflow_id=workflow.workflow_id,
                    workflow_status=workflow.status.value,
                    task_id=task.task_id,
                    task_key=task.task_key,
                    task_type=task.task_type,
                    task_status=task.status.value,
                    task_version=task.version,
                    attempt_count=task.attempt_count,
                    max_attempts=task.max_attempts,
                    updated_at=task.updated_at,
                    age_seconds=round(age, 3),
                    classification=classification,
                    recommendation=recommendation,
                    automatic_retry_allowed=False,
                    outcome_known=outcome_known,
                )
            )
        return tuple(sorted(candidates, key=lambda item: (item.age_seconds, item.task_key), reverse=True))

    def inspect_all(self, *, now: datetime | None = None) -> tuple[WorkflowRecoveryCandidate, ...]:
        list_workflows = getattr(self.store, "list_workflows", None)
        if not callable(list_workflows):
            raise TypeError("store must expose callable list_workflows(...) for inspect_all")
        workflows = list_workflows(limit=self.config.max_workflows)
        if len(workflows) > self.config.max_workflows:
            raise WorkflowRecoveryInspectionError("Workflow count exceeds inspection limit")
        results = []
        for workflow in workflows:
            results.extend(self.inspect_workflow(workflow.workflow_id, now=now))
        return tuple(sorted(results, key=lambda item: (item.age_seconds, item.workflow_id, item.task_key), reverse=True))


__all__ = [
    "TRUSTED_CAPABILITY_TASK_TYPE",
    "ANALYTICAL_TASK_TYPE",
    "REPORT_TASK_TYPE",
    "WorkflowRecoveryInspectionError",
    "RecoveryInspectionConfig",
    "WorkflowRecoveryCandidate",
    "WorkflowRecoveryInspector",
]
