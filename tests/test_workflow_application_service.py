from __future__ import annotations

import uuid

import pytest

from agent.workflow.planner import TaskSpec
from agent.workflow.service import WorkflowApplicationService, WorkflowServiceError
from agent.workflow.sqlite_store import SQLiteWorkflowStore


@pytest.fixture
def service(tmp_path):
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    return WorkflowApplicationService(store=store)


def test_create_is_idempotent(service):
    key = "create-" + str(uuid.uuid4())
    first = service.create_workflow(goal="database to report", idempotency_key=key)
    second = service.create_workflow(goal="different retry payload", idempotency_key=key)
    assert first.workflow_id == second.workflow_id
    assert second.status.value == "draft"


def test_plan_persists_dag_and_marks_workflow_ready(service):
    workflow = service.create_workflow(goal="report", idempotency_key=str(uuid.uuid4()))
    snapshot = service.plan_workflow(workflow.workflow_id, [
        TaskSpec("read", "trusted_capability.invoke", "Read allowlisted source"),
        TaskSpec("analyze", "workflow.analytical.execute", "Analyze data", ("read",)),
        TaskSpec("report", "report.generate", "Generate report", ("analyze",)),
    ])
    assert snapshot.workflow.status.value == "ready"
    by_key = {task.task_key: task for task in snapshot.tasks}
    assert by_key["read"].status.value == "ready"
    assert by_key["analyze"].status.value == "pending"
    assert by_key["report"].status.value == "pending"
    assert any(event.event_type == "workflow.created" for event in snapshot.events)


def test_invalid_plan_does_not_transition_workflow(service):
    workflow = service.create_workflow(goal="report", idempotency_key=str(uuid.uuid4()))
    with pytest.raises(WorkflowServiceError) as error:
        service.plan_workflow(workflow.workflow_id, [
            TaskSpec("a", "x", "A", ("missing",))
        ])
    assert error.value.code == "invalid_workflow_plan"
    assert service.get_workflow(workflow.workflow_id).workflow.status.value == "draft"
    assert service.get_workflow(workflow.workflow_id).tasks == ()


def test_start_requires_ready_workflow(service):
    workflow = service.create_workflow(goal="report", idempotency_key=str(uuid.uuid4()))
    with pytest.raises(WorkflowServiceError) as error:
        service.start_workflow(workflow.workflow_id)
    assert error.value.code == "workflow_not_ready"


def test_start_and_cancel_lifecycle(service):
    workflow = service.create_workflow(goal="report", idempotency_key=str(uuid.uuid4()))
    service.plan_workflow(workflow.workflow_id, [
        TaskSpec("read", "trusted_capability.invoke", "Read")
    ])
    running = service.start_workflow(workflow.workflow_id)
    assert running.status.value == "running"
    cancelled = service.cancel_workflow(workflow.workflow_id)
    assert cancelled.status.value == "cancelled"
    with pytest.raises(WorkflowServiceError) as error:
        service.cancel_workflow(workflow.workflow_id)
    assert error.value.code == "workflow_terminal"


def test_lists_and_events_validate_limits(service):
    workflow = service.create_workflow(goal="report", idempotency_key=str(uuid.uuid4()))
    assert len(service.list_workflows()) == 1
    assert service.get_workflow_events(workflow.workflow_id)
    with pytest.raises(WorkflowServiceError):
        service.list_workflows(limit=0)
    with pytest.raises(WorkflowServiceError):
        service.get_workflow_events(workflow.workflow_id, limit=0)


def test_missing_workflow_is_normalized(service):
    with pytest.raises(WorkflowServiceError) as error:
        service.get_workflow(str(uuid.uuid4()))
    assert error.value.code == "workflow_not_found"
