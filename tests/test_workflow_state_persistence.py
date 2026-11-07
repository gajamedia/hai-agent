import sqlite3
import pytest

from agent.workflow import (
    SQLiteWorkflowStore, WorkflowStatus, TaskStatus,
    WorkflowConcurrencyError, WorkflowStateError,
    WorkflowAlreadyExistsError, WorkflowNotFoundError,
    TaskAlreadyExistsError,
)


def store(tmp_path):
    return SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")


def make_workflow(s):
    return s.create_workflow(goal="Buat laporan penjualan", idempotency_key="request-001")


def test_create_and_reload_workflow(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    reopened = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    assert reopened.get_workflow(w.workflow_id) == w
    assert w.status == WorkflowStatus.DRAFT
    assert w.version == 1


def test_create_is_idempotent_by_key(tmp_path):
    s = store(tmp_path)
    first = make_workflow(s)
    second = s.create_workflow(goal="different payload", idempotency_key="request-001")
    assert first.workflow_id == second.workflow_id
    assert second.goal == "Buat laporan penjualan"


def test_duplicate_workflow_id_is_rejected(tmp_path):
    s = store(tmp_path)
    first = s.create_workflow(
        goal="first", idempotency_key="first-key",
        workflow_id="11111111-1111-4111-8111-111111111111",
    )
    assert first.workflow_id == "11111111-1111-4111-8111-111111111111"
    with pytest.raises(WorkflowAlreadyExistsError):
        s.create_workflow(goal="other", idempotency_key="other-key",
                          workflow_id=first.workflow_id)


def test_workflow_transitions_and_version(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    w = s.transition_workflow(w.workflow_id, "planned", expected_version=1)
    assert (w.status, w.version) == (WorkflowStatus.PLANNED, 2)
    w = s.transition_workflow(w.workflow_id, "ready", expected_version=2)
    assert (w.status, w.version) == (WorkflowStatus.READY, 3)
    w = s.transition_workflow(w.workflow_id, "running", expected_version=3)
    assert w.status == WorkflowStatus.RUNNING


def test_illegal_workflow_transition_rejected(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    with pytest.raises(WorkflowStateError):
        s.transition_workflow(w.workflow_id, "succeeded", expected_version=1)


def test_optimistic_concurrency_rejects_stale_version(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    s.transition_workflow(w.workflow_id, "planned", expected_version=1)
    with pytest.raises(WorkflowConcurrencyError):
        s.transition_workflow(w.workflow_id, "ready", expected_version=1)


def test_terminal_workflow_cannot_transition(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    for target in ("planned", "ready", "running", "succeeded"):
        w = s.transition_workflow(w.workflow_id, target, expected_version=w.version)
    with pytest.raises(WorkflowStateError):
        s.transition_workflow(w.workflow_id, "running", expected_version=w.version)


def test_task_create_and_lifecycle(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    task = s.create_task(w.workflow_id, task_key="fetch_sales",
                         task_type="database_read", description="Ambil penjualan")
    assert task.status == TaskStatus.PENDING
    task = s.transition_task(task.task_id, "ready", expected_version=1)
    task = s.transition_task(task.task_id, "running", expected_version=2)
    assert task.attempt_count == 1
    task = s.transition_task(task.task_id, "succeeded", expected_version=3,
                              result_data={"row_count": 12})
    assert task.result_data == {"row_count": 12}


def test_task_cannot_be_added_after_workflow_ready(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    w = s.transition_workflow(w.workflow_id, "planned", expected_version=1)
    w = s.transition_workflow(w.workflow_id, "ready", expected_version=2)
    with pytest.raises(WorkflowStateError):
        s.create_task(w.workflow_id, task_key="x", task_type="analysis", description="x")


def test_task_duplicate_key_rejected(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    s.create_task(w.workflow_id, task_key="fetch", task_type="db", description="Fetch")
    with pytest.raises(TaskAlreadyExistsError):
        s.create_task(w.workflow_id, task_key="fetch", task_type="db", description="Again")


def test_task_max_attempts_enforced(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    t = s.create_task(w.workflow_id, task_key="fetch", task_type="db",
                      description="Fetch", max_attempts=1)
    t = s.transition_task(t.task_id, "ready", expected_version=1)
    t = s.transition_task(t.task_id, "running", expected_version=2)
    t = s.transition_task(t.task_id, "failed", expected_version=3, error_code="timeout")
    t = s.transition_task(t.task_id, "ready", expected_version=4)
    with pytest.raises(WorkflowStateError, match="maximum attempts"):
        s.transition_task(t.task_id, "running", expected_version=5)


def test_failed_task_can_be_requeued_within_attempt_budget(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    t = s.create_task(w.workflow_id, task_key="fetch", task_type="db",
                      description="Fetch", max_attempts=2)
    t = s.transition_task(t.task_id, "ready", expected_version=1)
    t = s.transition_task(t.task_id, "running", expected_version=2)
    t = s.transition_task(t.task_id, "failed", expected_version=3, error_code="temporary")
    t = s.transition_task(t.task_id, "ready", expected_version=4)
    t = s.transition_task(t.task_id, "running", expected_version=5)
    assert t.attempt_count == 2


def test_events_are_persisted_and_append_only(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    t = s.create_task(w.workflow_id, task_key="fetch", task_type="db", description="Fetch")
    events = s.list_events(w.workflow_id)
    assert [e.event_type for e in events] == ["workflow.created", "task.created"]
    with sqlite3.connect(s.database_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM workflow_events")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE workflow_events SET metadata_json='{}'")


def test_failed_transition_rolls_back_without_event(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    before = len(s.list_events(w.workflow_id))
    with pytest.raises(WorkflowStateError):
        s.transition_workflow(w.workflow_id, "succeeded", expected_version=1)
    assert s.get_workflow(w.workflow_id).status == WorkflowStatus.DRAFT
    assert len(s.list_events(w.workflow_id)) == before


def test_task_result_is_json_object(tmp_path):
    s = store(tmp_path)
    w = make_workflow(s)
    t = s.create_task(w.workflow_id, task_key="analysis", task_type="analysis",
                      description="Analyze")
    t = s.transition_task(t.task_id, "ready", expected_version=1)
    t = s.transition_task(t.task_id, "running", expected_version=2)
    with pytest.raises(TypeError):
        s.transition_task(t.task_id, "succeeded", expected_version=3, result_data=["not", "object"])


def test_missing_workflow_is_reported(tmp_path):
    s = store(tmp_path)
    with pytest.raises(WorkflowNotFoundError):
        s.create_task("00000000-0000-4000-8000-000000000000",
                      task_key="x", task_type="x", description="x")


def test_invalid_limits_rejected(tmp_path):
    s = store(tmp_path)
    with pytest.raises(ValueError):
        s.list_workflows(limit=0)
    with pytest.raises(ValueError):
        s.list_events("x", limit=10001)
