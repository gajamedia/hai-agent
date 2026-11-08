from __future__ import annotations

import pytest

from agent.workflow import (
    SQLiteWorkflowStore,
    TaskDependencyPlanner,
    TaskSpec,
    TaskStatus,
    WorkflowStatus,
)
from agent.workflow.analytics import (
    ANALYTICAL_TASK_TYPE,
    WorkflowAnalyticalExecutionError,
    WorkflowAnalyticalTaskExecutor,
)
from agent.workflow.invocation import TRUSTED_CAPABILITY_TASK_TYPE


ROWS = [
    {"order_id": 1, "region": "west", "amount": 100, "status": "paid"},
    {"order_id": 2, "region": "east", "amount": 50, "status": "paid"},
    {"order_id": 3, "region": "west", "amount": 75, "status": "pending"},
]

CONFIG = {
    "source_task_key": "read_sales",
    "dataset_schema": {
        "columns": [
            {"name": "order_id", "data_type": "integer", "nullable": False},
            {"name": "region", "data_type": "string", "nullable": False},
            {"name": "amount", "data_type": "number", "nullable": False},
            {"name": "status", "data_type": "string", "nullable": False},
        ],
        "allow_extra_columns": False,
        "min_rows": 1,
        "max_rows": 100,
    },
    "quality_rules": {
        "required_non_null_columns": ["order_id", "region", "amount", "status"],
        "unique_key_columns": ["order_id"],
        "allowed_values_by_column": {"status": ["paid", "pending"]},
        "max_null_fraction_by_column": {},
        "column_comparisons": [],
    },
    "aggregation": {
        "group_by": ["region"],
        "aggregations": [
            {"output_column": "revenue", "operation": "sum", "source_column": "amount"},
            {"output_column": "orders", "operation": "count"},
        ],
    },
    "descriptive_statistics": {
        "numeric_columns": ["amount"],
        "categorical_columns": ["region", "status"],
    },
}


def make_workflow(tmp_path, *, rows=None, config=None, source_type=TRUSTED_CAPABILITY_TASK_TYPE,
                  source_read_only=True, include_report=True, source_result_override=None):
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    workflow = store.create_workflow(
        goal="Read, validate, analyze, and report sales",
        idempotency_key="analytics-workflow",
    )
    specs = [
        TaskSpec(
            task_key="read_sales",
            task_type=source_type,
            description="Read sales rows",
            input_data={"action": "read_sales", "arguments": {"table": "sales"}},
        ),
        TaskSpec(
            task_key="analyze_sales",
            task_type=ANALYTICAL_TASK_TYPE,
            description="Validate and analyze sales rows",
            depends_on=("read_sales",),
            input_data=CONFIG if config is None else config,
        ),
    ]
    if include_report:
        specs.append(TaskSpec(
            task_key="report",
            task_type="report.generate",
            description="Generate report",
            depends_on=("analyze_sales",),
        ))
    plan = TaskDependencyPlanner().plan(specs)
    tasks = store.create_tasks_from_plan(workflow.workflow_id, plan)
    read_task = next(item for item in tasks if item.task_key == "read_sales")
    read_task = store.transition_task(
        read_task.task_id, TaskStatus.READY, expected_version=read_task.version
    )
    read_task = store.transition_task(
        read_task.task_id, TaskStatus.RUNNING, expected_version=read_task.version
    )
    source_result = source_result_override
    if source_result is None:
        source_result = {
            "execution_id": "exec-read-001",
            "action": "read_sales",
            "target": "business_database",
            "capability_id": "business_database.select",
            "data": ROWS if rows is None else rows,
            "metadata": {"row_count": len(ROWS if rows is None else rows)},
            "read_only": source_read_only,
        }
    read_task = store.transition_task(
        read_task.task_id,
        TaskStatus.SUCCEEDED,
        expected_version=read_task.version,
        result_data=source_result,
    )
    store.refresh_task_readiness(workflow.workflow_id)
    current = store.get_workflow(workflow.workflow_id)
    current = store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.PLANNED, expected_version=current.version
    )
    current = store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.READY, expected_version=current.version
    )
    store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.RUNNING, expected_version=current.version
    )
    analyze_task = next(
        item for item in store.list_tasks(workflow.workflow_id)
        if item.task_key == "analyze_sales"
    )
    return store, workflow, read_task, analyze_task


def make_executor(store, *, max_result_bytes=5 * 1024 * 1024):
    return WorkflowAnalyticalTaskExecutor(
        store=store, max_result_bytes=max_result_bytes
    )


def test_analytical_task_runs_validations_aggregation_and_statistics(tmp_path):
    store, workflow, source, task = make_workflow(tmp_path)
    result = make_executor(store).execute_task(workflow.workflow_id, task.task_id)

    assert result.status is TaskStatus.SUCCEEDED
    assert result.attempt_count == 1
    assert result.result_data["source_task_key"] == "read_sales"
    assert result.result_data["source_execution_id"] == "exec-read-001"
    assert result.result_data["dataset_validation"]["valid"] is True
    assert result.result_data["data_quality"]["valid"] is True
    assert result.result_data["aggregation"]["group_count"] == 2
    assert result.result_data["descriptive_statistics"]["numeric"][0]["mean"] == 75.0
    assert store.get_task(task.task_id).result_data == result.result_data


def test_successful_analysis_unblocks_report_dependency(tmp_path):
    store, workflow, source, task = make_workflow(tmp_path)
    make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    changed = store.refresh_task_readiness(workflow.workflow_id)
    report = next(item for item in changed if item.task_key == "report")
    assert report.status is TaskStatus.READY


def test_dataset_validation_failure_fails_task_and_blocks_report(tmp_path):
    bad_rows = [dict(ROWS[0], order_id="not-an-integer")]
    store, workflow, source, task = make_workflow(tmp_path, rows=bad_rows)

    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)

    persisted = store.get_task(task.task_id)
    assert error.value.error_code == "dataset_validation_failed"
    assert persisted.status is TaskStatus.FAILED
    assert persisted.error_code == "dataset_validation_failed"
    assert persisted.result_data["dataset_validation"]["valid"] is False
    changed = store.refresh_task_readiness(workflow.workflow_id)
    report = next(item for item in changed if item.task_key == "report")
    assert report.status is TaskStatus.BLOCKED


def test_data_quality_failure_fails_task_and_persists_diagnostics(tmp_path):
    bad_rows = ROWS + [dict(ROWS[0])]
    store, workflow, source, task = make_workflow(tmp_path, rows=bad_rows)

    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)

    persisted = store.get_task(task.task_id)
    assert error.value.error_code == "data_quality_validation_failed"
    assert persisted.status is TaskStatus.FAILED
    assert any(
        item["code"] == "duplicate_key"
        for item in persisted.result_data["data_quality"]["issues"]
    )


def test_source_must_be_direct_dependency(tmp_path):
    config = dict(CONFIG)
    config["source_task_key"] = "not_a_dependency"
    store, workflow, source, task = make_workflow(tmp_path, config=config)

    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)

    assert error.value.error_code == "analytical_source_not_dependency"
    assert store.get_task(task.task_id).status is TaskStatus.READY


def test_source_must_be_trusted_capability_task(tmp_path):
    store, workflow, source, task = make_workflow(
        tmp_path, source_type="internal.fake_read"
    )
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "analytical_source_type_invalid"
    assert store.get_task(task.task_id).status is TaskStatus.READY


def test_source_must_be_explicitly_read_only(tmp_path):
    store, workflow, source, task = make_workflow(
        tmp_path, source_read_only=False
    )
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "analytical_source_not_read_only"
    assert store.get_task(task.task_id).status is TaskStatus.READY


def test_invalid_task_configuration_is_rejected_before_claim(tmp_path):
    config = dict(CONFIG)
    config["unexpected"] = "not-allowed"
    store, workflow, source, task = make_workflow(tmp_path, config=config)

    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)

    assert error.value.error_code == "invalid_analytical_task_input"
    persisted = store.get_task(task.task_id)
    assert persisted.status is TaskStatus.READY
    assert persisted.attempt_count == 0


def test_task_cannot_execute_when_workflow_not_running(tmp_path):
    store = SQLiteWorkflowStore(tmp_path / "not-running.sqlite3")
    workflow = store.create_workflow(goal="test", idempotency_key="not-running")
    task = store.create_task(
        workflow.workflow_id, task_key="analysis", task_type=ANALYTICAL_TASK_TYPE,
        description="Analyze", input_data=CONFIG,
    )
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "workflow_not_running"


def test_only_ready_task_can_execute(tmp_path):
    store, workflow, source, task = make_workflow(tmp_path)
    task = store.transition_task(
        task.task_id, TaskStatus.BLOCKED, expected_version=task.version
    )
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "task_not_ready"


def test_source_result_requires_execution_id(tmp_path):
    source_result = {
        "action": "read_sales",
        "target": "business_database",
        "capability_id": "business_database.select",
        "data": ROWS,
        "read_only": True,
    }
    store, workflow, source, task = make_workflow(
        tmp_path, source_result_override=source_result
    )
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "analytical_source_result_invalid"
    assert store.get_task(task.task_id).status is TaskStatus.READY


def test_source_result_data_must_be_array_of_row_objects(tmp_path):
    source_result = {
        "execution_id": "exec-read-001",
        "action": "read_sales",
        "target": "business_database",
        "capability_id": "business_database.select",
        "data": {"not": "rows"},
        "read_only": True,
    }
    store, workflow, source, task = make_workflow(
        tmp_path, source_result_override=source_result
    )
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "analytical_source_result_invalid"
    assert store.get_task(task.task_id).status is TaskStatus.READY

def test_result_size_limit_fails_task(tmp_path):
    store, workflow, source, task = make_workflow(tmp_path)
    executor = make_executor(store, max_result_bytes=1024)
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        executor.execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "analytical_result_too_large"
    persisted = store.get_task(task.task_id)
    assert persisted.status is TaskStatus.FAILED
    assert persisted.error_code == "analytical_result_too_large"


def test_no_database_or_capability_gateway_is_required(tmp_path):
    store, workflow, source, task = make_workflow(tmp_path)
    # Constructor intentionally accepts only the workflow state store.
    result = WorkflowAnalyticalTaskExecutor(store=store).execute_task(
        workflow.workflow_id, task.task_id
    )
    assert result.status is TaskStatus.SUCCEEDED


def test_source_task_result_is_not_modified(tmp_path):
    store, workflow, source, task = make_workflow(tmp_path)
    original = dict(source.result_data)
    make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert store.get_task(source.task_id).result_data == original


def test_unsupported_aggregation_operation_is_rejected_before_claim(tmp_path):
    config = {
        **CONFIG,
        "aggregation": {
            "group_by": ["region"],
            "aggregations": [
                {"output_column": "x", "operation": "median", "source_column": "amount"}
            ],
        },
    }
    store, workflow, source, task = make_workflow(tmp_path, config=config)
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "invalid_analytical_task_input"
    persisted = store.get_task(task.task_id)
    assert persisted.status is TaskStatus.READY
    assert persisted.attempt_count == 0


def test_wrong_workflow_task_pair_is_rejected(tmp_path):
    store, workflow, source, task = make_workflow(tmp_path)
    other = store.create_workflow(goal="other", idempotency_key="other-workflow")
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(other.workflow_id, task.task_id)
    assert error.value.error_code == "task_not_found"


def test_terminal_workflow_blocks_execution(tmp_path):
    store, workflow, source, task = make_workflow(tmp_path)
    current = store.get_workflow(workflow.workflow_id)
    store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.CANCELLED,
        expected_version=current.version,
    )
    with pytest.raises(WorkflowAnalyticalExecutionError) as error:
        make_executor(store).execute_task(workflow.workflow_id, task.task_id)
    assert error.value.error_code == "workflow_terminal"
