import sqlite3

import pytest

from agent.workflow import (
    SQLiteWorkflowStore, TaskDependencyPlanner, TaskSpec, TaskStatus,
    WorkflowPlanError, WorkflowStateError, WorkflowStatus,
)


def specs():
    return [
        TaskSpec("extract", "db.read", "Read source data"),
        TaskSpec("validate", "data.validate", "Validate rows", ("extract",)),
        TaskSpec("analyze", "data.analyze", "Analyze data", ("validate",)),
        TaskSpec("report", "report.build", "Build report", ("analyze",)),
        TaskSpec("metadata", "report.metadata", "Build metadata", ("extract",)),
    ]


def make_store(tmp_path):
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    workflow = store.create_workflow(goal="database to report", idempotency_key="case-1")
    return store, workflow


def test_deterministic_topological_order_and_initial_ready():
    plan = TaskDependencyPlanner().plan(specs())
    assert plan.execution_order == ("extract", "validate", "analyze", "report", "metadata")
    assert plan.initial_ready == ("extract",)
    assert plan.dependency_map["report"] == ("analyze",)


@pytest.mark.parametrize("items", [
    [TaskSpec("a", "x", "A"), TaskSpec("a", "x", "A2")],
    [TaskSpec("a", "x", "A", ("missing",))],
    [TaskSpec("a", "x", "A", ("a",))],
    [TaskSpec("a", "x", "A", ("b",)), TaskSpec("b", "x", "B", ("a",))],
])
def test_invalid_graphs_rejected(items):
    with pytest.raises(WorkflowPlanError):
        TaskDependencyPlanner().plan(items)


def test_empty_plan_rejected():
    with pytest.raises(WorkflowPlanError):
        TaskDependencyPlanner().plan([])


def test_persist_plan_and_gate_readiness(tmp_path):
    store, wf = make_store(tmp_path)
    plan = TaskDependencyPlanner().plan(specs())
    persisted = store.create_tasks_from_plan(wf.workflow_id, plan)
    assert [task.task_key for task in persisted] == list(plan.execution_order)
    assert store.get_task_dependencies(wf.workflow_id, "report") == ("analyze",)
    assert {t.task_key: t.status for t in store.refresh_task_readiness(wf.workflow_id)} == {"extract": TaskStatus.READY}
    extract = next(t for t in store.list_tasks(wf.workflow_id) if t.task_key == "extract")
    extract = store.transition_task(extract.task_id, TaskStatus.RUNNING, expected_version=extract.version)
    extract = store.transition_task(extract.task_id, TaskStatus.SUCCEEDED, expected_version=extract.version, result_data={"rows": 10})
    newly_ready = store.refresh_task_readiness(wf.workflow_id)
    assert {t.task_key for t in newly_ready} == {"validate", "metadata"}
    assert all(t.status == TaskStatus.READY for t in newly_ready)


def test_cannot_bypass_dependency_gate(tmp_path):
    store, wf = make_store(tmp_path)
    plan = TaskDependencyPlanner().plan(specs())
    tasks = store.create_tasks_from_plan(wf.workflow_id, plan)
    target = next(t for t in tasks if t.task_key == "validate")
    with pytest.raises(WorkflowStateError, match="dependencies"):
        store.transition_task(target.task_id, TaskStatus.READY, expected_version=target.version)


def test_failed_dependency_blocks_downstream_task(tmp_path):
    store, wf = make_store(tmp_path)
    plan = TaskDependencyPlanner().plan(specs())
    tasks = store.create_tasks_from_plan(wf.workflow_id, plan)
    root = next(t for t in tasks if t.task_key == "extract")
    root = store.transition_task(root.task_id, TaskStatus.READY, expected_version=root.version)
    root = store.transition_task(root.task_id, TaskStatus.RUNNING, expected_version=root.version)
    root = store.transition_task(root.task_id, TaskStatus.FAILED, expected_version=root.version, error_code="read_failed")
    changed = store.refresh_task_readiness(wf.workflow_id)
    assert {t.task_key for t in changed} == {"validate", "metadata"}
    assert all(t.status == TaskStatus.BLOCKED for t in changed)


def test_schema_v1_migrates_additively_and_preserves_data(tmp_path):
    db = tmp_path / "legacy.sqlite3"
    legacy = SQLiteWorkflowStore(db)
    wf = legacy.create_workflow(goal="legacy", idempotency_key="legacy-key")
    task = legacy.create_task(wf.workflow_id, task_key="old", task_type="old.type", description="Old task")
    before_events = legacy.list_events(wf.workflow_id)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE workflow_schema SET schema_version=1 WHERE singleton=1")
        conn.execute("DROP TABLE workflow_task_dependencies")
    migrated = SQLiteWorkflowStore(db)
    assert migrated.get_workflow(wf.workflow_id) == wf
    assert migrated.get_task(task.task_id) == task
    assert migrated.list_events(wf.workflow_id) == before_events
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT schema_version FROM workflow_schema").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM workflow_tasks").fetchone()[0] == 1


def test_plan_persistence_is_atomic_on_duplicate_task(tmp_path):
    store, wf = make_store(tmp_path)
    store.create_task(wf.workflow_id, task_key="extract", task_type="existing", description="Existing")
    plan = TaskDependencyPlanner().plan(specs())
    with pytest.raises(Exception):
        store.create_tasks_from_plan(wf.workflow_id, plan)
    assert [task.task_key for task in store.list_tasks(wf.workflow_id)] == ["extract"]
    assert store.get_task_dependencies(wf.workflow_id, "extract") == ()
