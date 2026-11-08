from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from agent.workflow.contracts import TaskStatus, WorkflowStatus
from agent.workflow.recovery import (
    RecoveryInspectionConfig,
    WorkflowRecoveryInspectionError,
    WorkflowRecoveryInspector,
)


@dataclass
class FakeWorkflow:
    workflow_id: str
    status: WorkflowStatus


@dataclass
class FakeTask:
    task_id: str
    workflow_id: str
    task_key: str
    task_type: str
    status: TaskStatus
    version: int
    attempt_count: int
    max_attempts: int
    updated_at: str


class FakeStore:
    def __init__(self, workflows, tasks):
        self.workflows = {w.workflow_id: w for w in workflows}
        self.tasks = tasks
        self.write_calls = 0

    def get_workflow(self, workflow_id):
        return self.workflows.get(workflow_id)

    def list_workflows(self, *, limit=100):
        return tuple(self.workflows.values())[:limit]

    def list_tasks(self, workflow_id):
        return tuple(t for t in self.tasks if t.workflow_id == workflow_id)


NOW = datetime(2026, 11, 7, 8, 0, tzinfo=timezone.utc)
OLD = (NOW - timedelta(minutes=30)).isoformat()
RECENT = (NOW - timedelta(minutes=2)).isoformat()


def task(task_type="trusted_capability.invoke", status=TaskStatus.RUNNING, updated=OLD):
    return FakeTask("task-1", "wf-1", "read", task_type, status, 4, 1, 3, updated)


def store_with(*tasks, status=WorkflowStatus.RUNNING):
    return FakeStore([FakeWorkflow("wf-1", status)], list(tasks))


def test_inspects_only_stale_running_tasks():
    store = store_with(task(), task(status=TaskStatus.READY, updated=OLD),
                       task(status=TaskStatus.RUNNING, updated=RECENT))
    result = WorkflowRecoveryInspector(
        store=store, config=RecoveryInspectionConfig(stale_after_seconds=600)
    ).inspect_workflow("wf-1", now=NOW)
    assert len(result) == 1
    assert result[0].task_key == "read"


def test_capability_task_is_unknown_outcome_and_never_retried():
    candidate = WorkflowRecoveryInspector(store=store_with(task())).inspect_workflow("wf-1", now=NOW)[0]
    assert candidate.outcome_known is None
    assert candidate.automatic_retry_allowed is False
    assert candidate.classification == "external_outcome_requires_reconciliation"


def test_report_task_requires_artifact_reconciliation():
    candidate = WorkflowRecoveryInspector(
        store=store_with(task("report.generate"))
    ).inspect_workflow("wf-1", now=NOW)[0]
    assert candidate.classification == "artifact_reconciliation_required"
    assert candidate.automatic_retry_allowed is False


def test_analytical_task_requires_explicit_reconciliation():
    candidate = WorkflowRecoveryInspector(
        store=store_with(task("workflow.analytical.execute"))
    ).inspect_workflow("wf-1", now=NOW)[0]
    assert candidate.classification == "deterministic_work_requires_operator_reconciliation"
    assert candidate.automatic_retry_allowed is False


def test_unknown_task_type_fails_closed():
    candidate = WorkflowRecoveryInspector(
        store=store_with(task("unknown.task"))
    ).inspect_workflow("wf-1", now=NOW)[0]
    assert candidate.classification == "unknown_task_requires_manual_review"
    assert candidate.automatic_retry_allowed is False


def test_terminal_workflow_has_no_recovery_candidates():
    result = WorkflowRecoveryInspector(
        store=store_with(task(), status=WorkflowStatus.SUCCEEDED)
    ).inspect_workflow("wf-1", now=NOW)
    assert result == ()


def test_missing_workflow_raises():
    with pytest.raises(WorkflowRecoveryInspectionError):
        WorkflowRecoveryInspector(store=store_with()).inspect_workflow("missing", now=NOW)


def test_naive_now_rejected():
    with pytest.raises(ValueError):
        WorkflowRecoveryInspector(store=store_with()).inspect_workflow(
            "wf-1", now=datetime(2026, 11, 7, 8, 0)
        )


def test_naive_persisted_timestamp_fails_closed():
    with pytest.raises(WorkflowRecoveryInspectionError):
        WorkflowRecoveryInspector(store=store_with(task(updated="2026-11-07T07:00:00"))).inspect_workflow(
            "wf-1", now=NOW
        )


def test_inspection_does_not_mutate_store():
    store = store_with(task())
    WorkflowRecoveryInspector(store=store).inspect_workflow("wf-1", now=NOW)
    assert store.write_calls == 0


def test_inspect_all_uses_bounded_workflow_listing():
    store = store_with(task())
    result = WorkflowRecoveryInspector(store=store).inspect_all(now=NOW)
    assert len(result) == 1


def test_config_rejects_invalid_limits():
    with pytest.raises(ValueError):
        RecoveryInspectionConfig(stale_after_seconds=0)
    with pytest.raises(ValueError):
        RecoveryInspectionConfig(max_workflows=0)
