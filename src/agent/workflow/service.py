from __future__ import annotations

"""Application orchestration for persisted workflows.

This layer coordinates the existing store and deterministic planner only.
It does not execute tasks, invoke capabilities, open business databases, or
create background workers.
"""
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from agent.workflow.contracts import Workflow, WorkflowEvent, WorkflowStatus, WorkflowTask
from agent.workflow.planner import TaskDependencyPlanner, TaskSpec, WorkflowPlan, WorkflowPlanError
from agent.workflow.sqlite_store import (
    WorkflowAlreadyExistsError, WorkflowConcurrencyError, WorkflowNotFoundError,
    WorkflowStateError,
)


class WorkflowServiceError(RuntimeError):
    def __init__(self, message: str, *, code: str = "workflow_service_error"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkflowSnapshot:
    workflow: Workflow
    tasks: tuple[WorkflowTask, ...]
    events: tuple[WorkflowEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
            "events": [event.to_dict() for event in self.events],
        }


class WorkflowApplicationService:
    """Use-case facade over the existing workflow persistence/planning contracts."""

    def __init__(self, *, store: Any, planner: TaskDependencyPlanner | None = None):
        if store is None:
            raise TypeError("store is required")
        required = (
            "create_workflow", "get_workflow", "list_workflows",
            "transition_workflow", "create_tasks_from_plan", "list_tasks",
            "refresh_task_readiness", "list_events",
        )
        for method in required:
            if not callable(getattr(store, method, None)):
                raise TypeError(f"store must expose callable {method}(...)")
        self.store = store
        self.planner = planner or TaskDependencyPlanner()

    def create_workflow(
        self, *, goal: str, idempotency_key: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Workflow:
        try:
            return self.store.create_workflow(
                goal=goal, idempotency_key=idempotency_key, metadata=metadata
            )
        except WorkflowAlreadyExistsError as exc:
            raise WorkflowServiceError(str(exc), code="workflow_already_exists") from exc
        except (TypeError, ValueError) as exc:
            raise WorkflowServiceError(str(exc), code="invalid_workflow_input") from exc

    def plan_workflow(
        self, workflow_id: str, tasks: Iterable[TaskSpec | Mapping[str, Any]],
    ) -> WorkflowSnapshot:
        workflow = self._require_workflow(workflow_id)
        if workflow.status is not WorkflowStatus.DRAFT:
            raise WorkflowServiceError(
                "Only a draft workflow can be planned.", code="workflow_not_draft"
            )
        try:
            plan = self.planner.plan(tasks)
            persisted = self.store.create_tasks_from_plan(workflow_id, plan)
            planned = self.store.transition_workflow(
                workflow_id, WorkflowStatus.PLANNED,
                expected_version=workflow.version,
                metadata={
                    "reason": "workflow_plan_persisted",
                    "task_count": len(persisted),
                    "execution_order": list(plan.execution_order),
                },
            )
            ready_tasks = self.store.refresh_task_readiness(workflow_id)
            if any(task.status.value == "ready" for task in ready_tasks):
                planned = self.store.transition_workflow(
                    workflow_id, WorkflowStatus.READY,
                    expected_version=planned.version,
                    metadata={"reason": "initial_tasks_ready"},
                )
            return self.get_workflow(workflow_id)
        except WorkflowPlanError as exc:
            raise WorkflowServiceError(str(exc), code="invalid_workflow_plan") from exc
        except (WorkflowConcurrencyError, WorkflowStateError) as exc:
            raise WorkflowServiceError(str(exc), code="workflow_state_conflict") from exc

    def get_workflow(self, workflow_id: str) -> WorkflowSnapshot:
        workflow = self._require_workflow(workflow_id)
        tasks = tuple(self.store.list_tasks(workflow_id))
        events = tuple(self.store.list_events(workflow_id))
        return WorkflowSnapshot(workflow=workflow, tasks=tasks, events=events)

    def list_workflows(self, *, limit: int = 100) -> tuple[Workflow, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise WorkflowServiceError("limit must be between 1 and 1000.", code="invalid_limit")
        return tuple(self.store.list_workflows(limit=limit))

    def get_workflow_events(self, workflow_id: str, *, limit: int = 1000) -> tuple[WorkflowEvent, ...]:
        self._require_workflow(workflow_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise WorkflowServiceError("limit must be between 1 and 10000.", code="invalid_limit")
        return tuple(self.store.list_events(workflow_id, limit=limit))

    def start_workflow(self, workflow_id: str) -> Workflow:
        workflow = self._require_workflow(workflow_id)
        if workflow.status is not WorkflowStatus.READY:
            raise WorkflowServiceError(
                "Workflow must be ready before it can start.", code="workflow_not_ready"
            )
        tasks = tuple(self.store.list_tasks(workflow_id))
        if not tasks or not any(task.status.value == "ready" for task in tasks):
            raise WorkflowServiceError(
                "At least one task must be ready before the workflow can start.",
                code="workflow_tasks_not_ready",
            )
        if any(task.status.value in {"blocked", "cancelled"} for task in tasks):
            raise WorkflowServiceError(
                "A blocked or cancelled task prevents workflow start.",
                code="workflow_tasks_not_ready",
            )
        try:
            return self.store.transition_workflow(
                workflow_id, WorkflowStatus.RUNNING,
                expected_version=workflow.version,
                metadata={"reason": "workflow_started"},
            )
        except (WorkflowConcurrencyError, WorkflowStateError) as exc:
            raise WorkflowServiceError(str(exc), code="workflow_state_conflict") from exc

    def cancel_workflow(self, workflow_id: str) -> Workflow:
        workflow = self._require_workflow(workflow_id)
        if workflow.status in {
            WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED
        }:
            raise WorkflowServiceError(
                "Terminal workflows cannot be cancelled.", code="workflow_terminal"
            )
        try:
            return self.store.transition_workflow(
                workflow_id, WorkflowStatus.CANCELLED,
                expected_version=workflow.version,
                metadata={"reason": "cancelled_by_application_service"},
            )
        except (WorkflowConcurrencyError, WorkflowStateError) as exc:
            raise WorkflowServiceError(str(exc), code="workflow_state_conflict") from exc

    def _require_workflow(self, workflow_id: str) -> Workflow:
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            raise WorkflowServiceError("workflow_id is required.", code="invalid_workflow_id")
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise WorkflowServiceError("Workflow does not exist.", code="workflow_not_found")
        return workflow


__all__ = ["WorkflowServiceError", "WorkflowSnapshot", "WorkflowApplicationService"]
