"""Workflow task executor for deterministic dataset validation and analytics.

This executor consumes a previously persisted, read-only capability result.
It never opens a database connection, executes SQL, calls a capability gateway,
or mutates the source dataset.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from agent.workflow.aggregation import AggregationSpec, aggregate_dataset
from agent.workflow.contracts import TaskStatus, WorkflowStatus, WorkflowTask
from agent.workflow.data_quality import (
    ColumnComparisonRule,
    DataQualityRules,
    assess_data_quality,
)
from agent.workflow.dataset import ColumnSpec, DatasetSchema, validate_dataset
from agent.workflow.descriptive_stats import describe_dataset
from agent.workflow.invocation import TRUSTED_CAPABILITY_TASK_TYPE


ANALYTICAL_TASK_TYPE = "workflow.analytical.execute"

_ALLOWED_INPUT_FIELDS = frozenset({
    "source_task_key",
    "dataset_schema",
    "quality_rules",
    "aggregation",
    "descriptive_statistics",
})


class WorkflowAnalyticalExecutionError(RuntimeError):
    """A workflow analytical task could not complete as a verified result."""

    def __init__(
        self,
        message: str,
        *,
        task: WorkflowTask | None = None,
        error_code: str = "workflow_analytical_execution_failed",
    ) -> None:
        super().__init__(message)
        self.task = task
        self.error_code = error_code


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _reject_unknown_fields(value: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {', '.join(sorted(map(str, unknown)))}")


def _string_tuple(value: Any, name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array of column names")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    result = tuple(item.strip() for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _build_dataset_schema(raw: Any) -> DatasetSchema:
    value = _require_mapping(raw, "dataset_schema")
    _reject_unknown_fields(
        value, frozenset({"columns", "allow_extra_columns", "min_rows", "max_rows"}),
        "dataset_schema",
    )
    columns_raw = value.get("columns")
    if not isinstance(columns_raw, list):
        raise ValueError("dataset_schema.columns must be a JSON array")
    columns = []
    for index, item in enumerate(columns_raw):
        item = _require_mapping(item, f"dataset_schema.columns[{index}]")
        _reject_unknown_fields(
            item, frozenset({"name", "data_type", "nullable", "required"}),
            f"dataset_schema.columns[{index}]",
        )
        columns.append(ColumnSpec(
            name=item.get("name"),
            data_type=item.get("data_type", "any"),
            nullable=item.get("nullable", True),
            required=item.get("required", True),
        ))
    return DatasetSchema(
        columns=tuple(columns),
        allow_extra_columns=value.get("allow_extra_columns", False),
        min_rows=value.get("min_rows", 0),
        max_rows=value.get("max_rows", 10000),
    )


def _build_quality_rules(raw: Any) -> DataQualityRules:
    value = _require_mapping(raw, "quality_rules")
    _reject_unknown_fields(
        value,
        frozenset({
            "required_non_null_columns", "max_null_fraction_by_column",
            "unique_key_columns", "allowed_values_by_column", "column_comparisons",
        }),
        "quality_rules",
    )

    required = _string_tuple(
        value.get("required_non_null_columns", []), "quality_rules.required_non_null_columns"
    )
    unique = _string_tuple(
        value.get("unique_key_columns", []), "quality_rules.unique_key_columns"
    )

    null_thresholds = value.get("max_null_fraction_by_column", {})
    if not isinstance(null_thresholds, Mapping):
        raise ValueError("quality_rules.max_null_fraction_by_column must be a JSON object")
    null_thresholds = dict(null_thresholds)

    allowed_raw = value.get("allowed_values_by_column", {})
    if not isinstance(allowed_raw, Mapping):
        raise ValueError("quality_rules.allowed_values_by_column must be a JSON object")
    allowed = {}
    for column, choices in allowed_raw.items():
        if not isinstance(column, str) or not column:
            raise ValueError("allowed-value column names must be non-empty strings")
        if not isinstance(choices, list):
            raise ValueError(f"allowed values for {column!r} must be a JSON array")
        allowed[column] = tuple(choices)

    comparisons_raw = value.get("column_comparisons", [])
    if not isinstance(comparisons_raw, list):
        raise ValueError("quality_rules.column_comparisons must be a JSON array")
    comparisons = []
    for index, item in enumerate(comparisons_raw):
        item = _require_mapping(item, f"quality_rules.column_comparisons[{index}]")
        _reject_unknown_fields(
            item, frozenset({"left_column", "operator", "right_column"}),
            f"quality_rules.column_comparisons[{index}]",
        )
        comparisons.append(ColumnComparisonRule(
            left_column=item.get("left_column"),
            operator=item.get("operator"),
            right_column=item.get("right_column"),
        ))

    return DataQualityRules(
        required_non_null_columns=required,
        max_null_fraction_by_column=null_thresholds,
        unique_key_columns=unique,
        allowed_values_by_column=allowed,
        column_comparisons=tuple(comparisons),
    )


def _build_aggregation(raw: Any) -> tuple[tuple[str, ...], tuple[AggregationSpec, ...]] | None:
    if raw is None:
        return None
    value = _require_mapping(raw, "aggregation")
    _reject_unknown_fields(value, frozenset({"group_by", "aggregations"}), "aggregation")
    group_by = _string_tuple(value.get("group_by", []), "aggregation.group_by")
    specs_raw = value.get("aggregations")
    if not isinstance(specs_raw, list) or not specs_raw:
        raise ValueError("aggregation.aggregations must be a non-empty JSON array")
    specs = []
    for index, item in enumerate(specs_raw):
        item = _require_mapping(item, f"aggregation.aggregations[{index}]")
        _reject_unknown_fields(
            item, frozenset({"output_column", "operation", "source_column"}),
            f"aggregation.aggregations[{index}]",
        )
        specs.append(AggregationSpec(
            output_column=item.get("output_column"),
            operation=item.get("operation"),
            source_column=item.get("source_column"),
        ))
    return group_by, tuple(specs)


def _build_descriptive_columns(raw: Any) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    if raw is None:
        return None
    value = _require_mapping(raw, "descriptive_statistics")
    _reject_unknown_fields(
        value, frozenset({"numeric_columns", "categorical_columns"}),
        "descriptive_statistics",
    )
    numeric = _string_tuple(
        value.get("numeric_columns", []), "descriptive_statistics.numeric_columns"
    )
    categorical = _string_tuple(
        value.get("categorical_columns", []), "descriptive_statistics.categorical_columns"
    )
    if not numeric and not categorical:
        raise ValueError("descriptive_statistics must select at least one column")
    return numeric, categorical


def _parse_task_configuration(task: WorkflowTask) -> dict[str, Any]:
    if task.task_type != ANALYTICAL_TASK_TYPE:
        raise WorkflowAnalyticalExecutionError(
            f"Unsupported task_type: {task.task_type}",
            task=task,
            error_code="unsupported_analytical_task_type",
        )
    payload = dict(task.input_data)
    _reject_unknown_fields(payload, _ALLOWED_INPUT_FIELDS, "analytical task input")
    source_task_key = payload.get("source_task_key")
    if not isinstance(source_task_key, str) or not source_task_key.strip():
        raise ValueError("source_task_key must be non-empty text")
    if "dataset_schema" not in payload:
        raise ValueError("dataset_schema is required")
    if "quality_rules" not in payload:
        raise ValueError("quality_rules is required")

    schema = _build_dataset_schema(payload["dataset_schema"])
    rules = _build_quality_rules(payload["quality_rules"])
    # Public pure functions provide the authoritative validation of declarative
    # rules/specifications without requiring any source rows or side effects.
    assess_data_quality([], rules)
    aggregation = _build_aggregation(payload.get("aggregation"))
    if aggregation is not None:
        aggregate_dataset([], group_by=aggregation[0], aggregations=aggregation[1])
    descriptive = _build_descriptive_columns(payload.get("descriptive_statistics"))
    if descriptive is not None:
        describe_dataset(
            [], numeric_columns=descriptive[0], categorical_columns=descriptive[1]
        )
    if aggregation is None and descriptive is None:
        raise ValueError("at least one of aggregation or descriptive_statistics is required")

    return {
        "source_task_key": source_task_key.strip(),
        "schema": schema,
        "quality_rules": rules,
        "aggregation": aggregation,
        "descriptive_statistics": descriptive,
    }


class WorkflowAnalyticalTaskExecutor:
    """Run validation and analytics from a succeeded direct capability dependency.

    Task input JSON contract:
    {
      "source_task_key": "read_sales",
      "dataset_schema": {
        "columns": [
          {"name": "order_id", "data_type": "integer", "nullable": false},
          {"name": "region", "data_type": "string"},
          {"name": "amount", "data_type": "number"}
        ],
        "allow_extra_columns": false,
        "min_rows": 1,
        "max_rows": 10000
      },
      "quality_rules": {
        "required_non_null_columns": ["order_id"],
        "unique_key_columns": ["order_id"],
        "max_null_fraction_by_column": {"region": 0.05},
        "allowed_values_by_column": {"region": ["west", "east"]},
        "column_comparisons": []
      },
      "aggregation": {
        "group_by": ["region"],
        "aggregations": [
          {"output_column": "revenue", "operation": "sum", "source_column": "amount"}
        ]
      },
      "descriptive_statistics": {
        "numeric_columns": ["amount"],
        "categorical_columns": ["region"]
      }
    }

    `source_task_key` must name a direct dependency whose task type is the
    existing trusted capability invocation and whose persisted result is
    explicitly marked read_only. No gateway or business database is accessed.
    """

    def __init__(self, *, store: Any, max_result_bytes: int = 5 * 1024 * 1024) -> None:
        if store is None:
            raise TypeError("store is required")
        for name in (
            "get_workflow", "get_task", "list_tasks",
            "get_task_dependencies", "transition_task",
        ):
            if not callable(getattr(store, name, None)):
                raise TypeError(f"store must expose callable {name}(...).")
        if (
            isinstance(max_result_bytes, bool)
            or not isinstance(max_result_bytes, int)
            or max_result_bytes < 1024
        ):
            raise ValueError("max_result_bytes must be an integer of at least 1024")
        self.store = store
        self.max_result_bytes = max_result_bytes

    def _fail(
        self,
        claimed: WorkflowTask,
        *,
        error_code: str,
        message: str,
        result_data: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowTask:
        try:
            return self.store.transition_task(
                claimed.task_id,
                TaskStatus.FAILED,
                expected_version=claimed.version,
                result_data=result_data,
                error_code=error_code,
                metadata={
                    "reason": error_code,
                    **dict(metadata or {}),
                },
            )
        except Exception as exc:
            raise WorkflowAnalyticalExecutionError(
                "Analytical execution failed and its failure state could not be persisted.",
                task=claimed,
                error_code="analytical_failure_persistence_failed",
            ) from exc

    def execute_task(self, workflow_id: str, task_id: str) -> WorkflowTask:
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise WorkflowAnalyticalExecutionError(
                "Workflow does not exist.", error_code="workflow_not_found"
            )
        task = self.store.get_task(task_id)
        if task is None or task.workflow_id != workflow_id:
            raise WorkflowAnalyticalExecutionError(
                "Task does not belong to the requested workflow.",
                error_code="task_not_found",
            )
        if workflow.status in {
            WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED
        }:
            raise WorkflowAnalyticalExecutionError(
                "Terminal workflows cannot execute analytical tasks.",
                error_code="workflow_terminal",
            )
        if workflow.status is not WorkflowStatus.RUNNING:
            raise WorkflowAnalyticalExecutionError(
                "Workflow must be running before analytical execution.",
                error_code="workflow_not_running",
            )
        if task.status is not TaskStatus.READY:
            raise WorkflowAnalyticalExecutionError(
                "Only a ready analytical task can execute.",
                task=task,
                error_code="task_not_ready",
            )

        # Validate configuration before claiming the task; invalid configuration
        # is a planning/input error and must not consume an execution attempt.
        try:
            config = _parse_task_configuration(task)
        except WorkflowAnalyticalExecutionError:
            raise
        except (TypeError, ValueError) as exc:
            raise WorkflowAnalyticalExecutionError(
                str(exc),
                task=task,
                error_code="invalid_analytical_task_input",
            ) from exc

        tasks_by_key = {
            item.task_key: item for item in self.store.list_tasks(workflow_id)
        }
        source_task_key = config["source_task_key"]
        dependencies = self.store.get_task_dependencies(workflow_id, task.task_key)
        if source_task_key not in dependencies:
            raise WorkflowAnalyticalExecutionError(
                "Source task must be an explicit direct dependency of the analytical task.",
                task=task,
                error_code="analytical_source_not_dependency",
            )
        source_task = tasks_by_key.get(source_task_key)
        if source_task is None or source_task.status is not TaskStatus.SUCCEEDED:
            raise WorkflowAnalyticalExecutionError(
                "Source task must exist and have succeeded.",
                task=task,
                error_code="analytical_source_not_succeeded",
            )
        if source_task.task_type != TRUSTED_CAPABILITY_TASK_TYPE:
            raise WorkflowAnalyticalExecutionError(
                "Analytical source must be a trusted capability invocation task.",
                task=task,
                error_code="analytical_source_type_invalid",
            )

        source_result = source_task.result_data
        if not isinstance(source_result, Mapping) or source_result.get("read_only") is not True:
            raise WorkflowAnalyticalExecutionError(
                "Source task result is not a verified read-only capability result.",
                task=task,
                error_code="analytical_source_not_read_only",
            )
        source_execution_id = source_result.get("execution_id")
        rows = source_result.get("data")
        if not isinstance(source_execution_id, str) or not source_execution_id.strip():
            raise WorkflowAnalyticalExecutionError(
                "Source capability result has no execution_id.",
                task=task,
                error_code="analytical_source_result_invalid",
            )
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise WorkflowAnalyticalExecutionError(
                "Source capability result data must be an array of row objects.",
                task=task,
                error_code="analytical_source_result_invalid",
            )

        try:
            claimed = self.store.transition_task(
                task.task_id,
                TaskStatus.RUNNING,
                expected_version=task.version,
                metadata={
                    "reason": "analytical_execution_started",
                    "source_task_key": source_task_key,
                    "source_execution_id": source_execution_id,
                },
            )
        except Exception as exc:
            raise WorkflowAnalyticalExecutionError(
                "Analytical task could not be claimed; no analysis was performed.",
                task=task,
                error_code="analytical_task_claim_failed",
            ) from exc

        try:
            dataset_report = validate_dataset(rows, config["schema"])
            if not dataset_report.valid:
                failure_result = {
                    "source_task_key": source_task_key,
                    "source_execution_id": source_execution_id,
                    "row_count": dataset_report.row_count,
                    "dataset_validation": dataset_report.to_dict(),
                    "data_quality": None,
                    "aggregation": None,
                    "descriptive_statistics": None,
                }
                failed = self._fail(
                    claimed,
                    error_code="dataset_validation_failed",
                    message="Dataset structural/type validation failed.",
                    result_data=failure_result,
                    metadata={
                        "issue_count": len(dataset_report.issues),
                        "issue_codes": sorted({issue.code for issue in dataset_report.issues}),
                    },
                )
                raise WorkflowAnalyticalExecutionError(
                    "Dataset structural/type validation failed.",
                    task=failed,
                    error_code="dataset_validation_failed",
                )

            quality_report = assess_data_quality(rows, config["quality_rules"])
            if not quality_report.valid:
                failure_result = {
                    "source_task_key": source_task_key,
                    "source_execution_id": source_execution_id,
                    "row_count": len(rows),
                    "dataset_validation": dataset_report.to_dict(),
                    "data_quality": quality_report.to_dict(),
                    "aggregation": None,
                    "descriptive_statistics": None,
                }
                failed = self._fail(
                    claimed,
                    error_code="data_quality_validation_failed",
                    message="Dataset quality validation failed.",
                    result_data=failure_result,
                    metadata={
                        "issue_count": quality_report.issue_count,
                        "issue_codes": sorted({issue.code for issue in quality_report.issues}),
                    },
                )
                raise WorkflowAnalyticalExecutionError(
                    "Dataset quality validation failed.",
                    task=failed,
                    error_code="data_quality_validation_failed",
                )

            aggregation_result = None
            if config["aggregation"] is not None:
                group_by, specs = config["aggregation"]
                aggregation_result = aggregate_dataset(
                    rows, group_by=group_by, aggregations=specs
                ).to_dict()

            descriptive_result = None
            if config["descriptive_statistics"] is not None:
                numeric_columns, categorical_columns = config["descriptive_statistics"]
                descriptive_result = describe_dataset(
                    rows,
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                ).to_dict()

            result_payload = {
                "source_task_key": source_task_key,
                "source_execution_id": source_execution_id,
                "source_row_count": len(rows),
                "dataset_validation": dataset_report.to_dict(),
                "data_quality": quality_report.to_dict(),
                "aggregation": aggregation_result,
                "descriptive_statistics": descriptive_result,
            }
            encoded = json.dumps(
                result_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if len(encoded.encode("utf-8")) > self.max_result_bytes:
                failed = self._fail(
                    claimed,
                    error_code="analytical_result_too_large",
                    message="Analytical result exceeds the configured persistence limit.",
                    metadata={"max_result_bytes": self.max_result_bytes},
                )
                raise WorkflowAnalyticalExecutionError(
                    "Analytical result exceeds the configured persistence limit.",
                    task=failed,
                    error_code="analytical_result_too_large",
                )
            persisted_payload = json.loads(encoded)
            completed = self.store.transition_task(
                claimed.task_id,
                TaskStatus.SUCCEEDED,
                expected_version=claimed.version,
                result_data=persisted_payload,
                metadata={
                    "reason": "analytical_execution_completed",
                    "source_task_key": source_task_key,
                    "source_execution_id": source_execution_id,
                    "source_row_count": len(rows),
                    "aggregation_group_count": (
                        None if aggregation_result is None
                        else aggregation_result["group_count"]
                    ),
                },
            )
            return completed

        except WorkflowAnalyticalExecutionError:
            raise
        except Exception as exc:
            failed = self._fail(
                claimed,
                error_code="analytical_execution_failed",
                message="Analytical computation failed.",
                metadata={"exception_type": type(exc).__name__},
            )
            raise WorkflowAnalyticalExecutionError(
                "Analytical computation failed.",
                task=failed,
                error_code="analytical_execution_failed",
            ) from exc


__all__ = [
    "ANALYTICAL_TASK_TYPE",
    "WorkflowAnalyticalExecutionError",
    "WorkflowAnalyticalTaskExecutor",
]
