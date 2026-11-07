from __future__ import annotations

"""End-to-end job: read an approved table, validate + analyze it, write a report.

This is the thin orchestration layer that connects three existing, separately
tested executors into one runnable job:

    read (trusted capability) -> analyze (quality + aggregation) -> report

It owns no gateway, database connection, credentials or approval. Everything
risky is injected and enforced by the existing boundaries (authenticated
principal, read-only gateway, execution security policy, artifact store).

Truncation guard: the read adapter caps rows per request, so a large table
would otherwise yield silently partial totals. The read asks for ``row_limit``
rows while the dataset schema allows at most ``row_limit - 1``; a table that
reaches the cap therefore fails validation instead of producing a wrong report.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from agent.workflow.analytics import ANALYTICAL_TASK_TYPE, analytical_column_order, order_columns
from agent.workflow.contracts import TaskStatus, WorkflowStatus
from agent.workflow.invocation import TRUSTED_CAPABILITY_TASK_TYPE
from agent.workflow.planner import TaskSpec
from agent.workflow.reports import REPORT_TASK_TYPE

READ_ACTION = "read_business_data"
MAX_FILTERS = 3
MAX_LIST_COLUMNS = 30
MAX_ORDER_BY = 3
DEFAULT_LIST_LIMIT = 100
MAX_RESULT_ROWS = 200
MAX_FILTER_VALUE_LENGTH = 200


@dataclass(frozen=True)
class TableReportRequest:
    """Declarative description of one table -> report job."""

    table: str
    columns: Mapping[str, str]  # column name -> data type (see dataset.SUPPORTED_DATA_TYPES)
    group_by: tuple[str, ...]
    metrics: tuple[tuple[str, str, str], ...]  # (output_column, operation, source_column)
    title: str = "Table Report"
    report_format: str = "json"
    row_limit: int = 500
    required_non_null_columns: tuple[str, ...] = ()
    unique_key_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    # Equality filters only (column == value); applied by the read-only adapter
    # with bound parameters and only on columns it allowlists as filterable.
    filters: Mapping[str, Any] = field(default_factory=dict)
    # "aggregate" summarizes (group_by + metrics); "list" shows the rows themselves.
    kind: str = "aggregate"
    select_columns: tuple[str, ...] = ()
    order_by: tuple[tuple[str, str], ...] = ()  # (column, "asc" | "desc")
    limit: int = DEFAULT_LIST_LIMIT

    def __post_init__(self) -> None:
        if not self.table.strip() or not self.columns:
            raise ValueError("table and columns are required")
        if self.kind not in {"aggregate", "list"}:
            raise ValueError("kind must be 'aggregate' or 'list'")
        if isinstance(self.row_limit, bool) or not isinstance(self.row_limit, int) or self.row_limit < 2:
            raise ValueError("row_limit must be an integer >= 2")
        if self.kind == "aggregate":
            if not self.group_by or not self.metrics:
                raise ValueError("group_by and metrics are required")
        else:
            if not self.select_columns or len(self.select_columns) > MAX_LIST_COLUMNS \
                    or len(set(self.select_columns)) != len(self.select_columns):
                raise ValueError(f"select_columns must be 1-{MAX_LIST_COLUMNS} distinct columns")
            if isinstance(self.limit, bool) or not isinstance(self.limit, int) \
                    or not 1 <= self.limit <= self.row_limit - 1:
                raise ValueError(f"limit must be an integer between 1 and {self.row_limit - 1}")
            if len(self.order_by) > MAX_ORDER_BY or any(
                    len(item) != 2 or item[1] not in {"asc", "desc"} for item in self.order_by):
                raise ValueError(f"order_by must have at most {MAX_ORDER_BY} (column, asc|desc) items")
        referenced = set(self.group_by) | {m[2] for m in self.metrics}
        referenced |= set(self.required_non_null_columns) | set(self.unique_key_columns)
        referenced |= set(self.numeric_columns) | set(self.categorical_columns)
        referenced |= set(self.select_columns) | {item[0] for item in self.order_by}
        unknown = referenced - set(self.columns)
        if unknown:
            raise ValueError(f"columns not declared: {sorted(unknown)}")
        if not isinstance(self.filters, Mapping) or len(self.filters) > MAX_FILTERS:
            raise ValueError(f"filters must be a mapping of at most {MAX_FILTERS} items")
        for column, value in self.filters.items():
            if not isinstance(column, str) or not column.strip():
                raise ValueError("filter column names must be non-empty text")
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not (isinstance(value, (str, bool)) or is_number):
                raise ValueError(f"filter value for {column!r} must be text, bool or a number")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"filter value for {column!r} must be finite")
            if isinstance(value, str) and len(value) > MAX_FILTER_VALUE_LENGTH:
                raise ValueError(f"filter value for {column!r} is too long")


def _build_list_task_specs(request: TableReportRequest) -> list[TaskSpec]:
    """Show rows: read ``limit + 1`` (so "more exist" is detectable), keep ``limit``, write the report.

    No type or quality rules are enforced: a listing shows the data as it is, NULLs included.
    The ORDER BY and LIMIT are applied by the database through the allowlisted adapter.
    """
    read_arguments: dict[str, Any] = {
        "table": request.table,
        "columns": list(request.select_columns),
        "limit": request.limit + 1,
    }
    if request.order_by:
        read_arguments["order_by"] = [("-" if direction == "desc" else "") + column
                                      for column, direction in request.order_by]
    title = request.title
    if request.filters:
        read_arguments["filters"] = dict(request.filters)
        shown = ", ".join(f"{k}={v}" for k, v in request.filters.items())
        title = f"{title} [filter: {shown}]"[:200]
    analytical = {
        "source_task_key": "read",
        "dataset_schema": {
            "columns": [{"name": c, "data_type": "any", "nullable": True} for c in request.select_columns],
            "allow_extra_columns": False,
            "min_rows": 1,
            "max_rows": request.limit + 1,
        },
        "quality_rules": {"required_non_null_columns": [], "unique_key_columns": []},
        "detail": {"columns": list(request.select_columns), "limit": request.limit},
    }
    return [
        TaskSpec("read", TRUSTED_CAPABILITY_TASK_TYPE, f"Read table {request.table}",
                 input_data={"action": READ_ACTION, "arguments": read_arguments}),
        TaskSpec("analyze", ANALYTICAL_TASK_TYPE, "Validate and collect rows",
                 depends_on=("read",), input_data=analytical),
        TaskSpec("report", REPORT_TASK_TYPE, "Write report", depends_on=("analyze",),
                 input_data={"source_task_key": "analyze", "title": title,
                             "format": request.report_format}),
    ]


def build_task_specs(request: TableReportRequest) -> list[TaskSpec]:
    """Translate a request into the three dependent workflow tasks."""
    if request.kind == "list":
        return _build_list_task_specs(request)
    required = set(request.required_non_null_columns)
    columns = [
        {"name": name, "data_type": data_type, "nullable": name not in required}
        for name, data_type in request.columns.items()
    ]
    descriptive: dict[str, Any] | None = None
    if request.numeric_columns or request.categorical_columns:
        descriptive = {
            "numeric_columns": list(request.numeric_columns),
            "categorical_columns": list(request.categorical_columns),
        }
    analytical: dict[str, Any] = {
        "source_task_key": "read",
        "dataset_schema": {
            "columns": columns,
            "allow_extra_columns": False,
            "min_rows": 1,
            "max_rows": request.row_limit - 1,  # truncation guard, see module docstring
        },
        "quality_rules": {
            "required_non_null_columns": list(request.required_non_null_columns),
            "unique_key_columns": list(request.unique_key_columns),
        },
        "aggregation": {
            "group_by": list(request.group_by),
            "aggregations": [
                {"output_column": out, "operation": op, "source_column": src}
                for out, op, src in request.metrics
            ],
        },
    }
    if descriptive is not None:
        analytical["descriptive_statistics"] = descriptive
    read_arguments: dict[str, Any] = {
        "table": request.table,
        "columns": list(request.columns),
        "limit": request.row_limit,
    }
    title = request.title
    if request.filters:
        read_arguments["filters"] = dict(request.filters)
        # Make the scope visible on the report itself so totals are not mistaken
        # for whole-table figures.
        shown = ", ".join(f"{k}={v}" for k, v in request.filters.items())
        title = f"{title} [filter: {shown}]"[:200]
    return [
        TaskSpec(
            "read", TRUSTED_CAPABILITY_TASK_TYPE, f"Read table {request.table}",
            input_data={"action": READ_ACTION, "arguments": read_arguments},
        ),
        TaskSpec("analyze", ANALYTICAL_TASK_TYPE, "Validate and aggregate rows",
                 depends_on=("read",), input_data=analytical),
        TaskSpec("report", REPORT_TASK_TYPE, "Write report", depends_on=("analyze",),
                 input_data={"source_task_key": "analyze", "title": title,
                             "format": request.report_format}),
    ]


@dataclass(frozen=True)
class JobOutcome:
    workflow_id: str
    succeeded: bool
    task_statuses: Mapping[str, str]
    artifact: Mapping[str, Any] | None = None
    failed_task_key: str | None = None
    error_code: str | None = None
    message: str | None = None
    rows: tuple[Mapping[str, Any], ...] = ()  # result rows shown on screen (capped)
    kind: str = "aggregate"
    row_count: int | None = None      # all rows in the result (the screen shows at most MAX_RESULT_ROWS)
    limit_reached: bool = False       # a listing was cut at its row limit: more rows exist


class TableReportJobRunner:
    """Synchronously drive one workflow to a terminal state.

    Each task is dispatched at most once per run, so a task that an executor
    rejects before claiming it cannot cause an infinite loop. Any failure ends
    the workflow as FAILED with the failing task and a machine-readable code.
    """

    def __init__(self, *, service: Any, store: Any, capability_executor: Any,
                 analytical_executor: Any, report_executor: Any) -> None:
        self.service = service
        self.store = store
        self.capability_executor = capability_executor
        self.analytical_executor = analytical_executor
        self.report_executor = report_executor

    def run(self, *, goal: str, idempotency_key: str, request: TableReportRequest,
            execution_context: Any) -> JobOutcome:
        workflow = self.service.create_workflow(
            goal=goal, idempotency_key=idempotency_key,
            metadata={"job": "table_report", "table": request.table},
        )
        wid = workflow.workflow_id
        self.service.plan_workflow(wid, build_task_specs(request))
        self.service.start_workflow(wid)

        dispatched: set[str] = set()
        while True:
            self.store.refresh_task_readiness(wid)
            tasks = {t.task_key: t for t in self.store.list_tasks(wid)}
            failed = next((t for t in tasks.values() if t.status is TaskStatus.FAILED), None)
            if failed is not None:
                code, message = self._diagnose(tasks, failed.task_key,
                                               failed.error_code or "task_failed", None)
                return self._finish(wid, tasks, False, failed.task_key, code, message)
            if all(t.status is TaskStatus.SUCCEEDED for t in tasks.values()):
                return self._finish(wid, tasks, True, None, None, None)
            ready = [t for t in tasks.values()
                     if t.status is TaskStatus.READY and t.task_key not in dispatched]
            if not ready:
                stuck = next((t for t in tasks.values() if t.status is not TaskStatus.SUCCEEDED), None)
                return self._finish(wid, tasks, False, stuck.task_key if stuck else None,
                                    "no_runnable_task", "No task is ready to run.")
            task = ready[0]
            dispatched.add(task.task_key)
            try:
                self._dispatch(wid, task, execution_context)
            except Exception as exc:  # executors raise typed errors with an error_code
                code = getattr(exc, "error_code", None) or type(exc).__name__
                tasks = {t.task_key: t for t in self.store.list_tasks(wid)}
                code, message = self._diagnose(tasks, task.task_key, code, str(exc))
                return self._finish(wid, tasks, False, task.task_key, code, message)

    @staticmethod
    def _diagnose(tasks: Mapping[str, Any], failed_key: str, code: str,
                  message: str | None) -> tuple[str, str | None]:
        """Replace an opaque analyze-stage failure with a clear one when the read was empty."""
        read = tasks.get("read")
        if (failed_key == "analyze" and read is not None and read.result_data
                and read.result_data.get("data") == []):
            return "no_matching_rows", "The query returned no rows (check the filter values)."
        return code, message

    def _dispatch(self, workflow_id: str, task: Any, execution_context: Any) -> None:
        if task.task_type == TRUSTED_CAPABILITY_TASK_TYPE:
            self.capability_executor.execute_task(
                workflow_id, task.task_id, execution_context=execution_context)
        elif task.task_type == ANALYTICAL_TASK_TYPE:
            self.analytical_executor.execute_task(workflow_id, task.task_id)
        elif task.task_type == REPORT_TASK_TYPE:
            self.report_executor.execute_task(workflow_id, task.task_id)
        else:
            raise RuntimeError(f"unsupported task type: {task.task_type}")

    def _finish(self, workflow_id: str, tasks: Mapping[str, Any], ok: bool,
                failed_key: str | None, code: str | None, message: str | None) -> JobOutcome:
        current = self.store.get_workflow(workflow_id)
        target = WorkflowStatus.SUCCEEDED if ok else WorkflowStatus.FAILED
        self.store.transition_workflow(
            workflow_id, target, expected_version=current.version,
            metadata={"reason": "job_finished", "error_code": code},
        )
        artifact = None
        report = tasks.get("report")
        if ok and report is not None and report.result_data:
            artifact = dict(report.result_data.get("artifact") or {}) or None
        rows: tuple[Mapping[str, Any], ...] = ()
        kind, row_count, limit_reached = "aggregate", None, False
        analyze = tasks.get("analyze")
        if ok and analyze is not None and analyze.result_data:
            result = analyze.result_data.get("aggregation") or {}
            raw = result.get("rows") or []
            # stored results have alphabetical keys: show the columns in the order that was asked for
            shown = [r for r in raw[:MAX_RESULT_ROWS] if isinstance(r, Mapping)]
            rows = tuple(order_columns(shown, analytical_column_order(dict(analyze.input_data or {}))))
            # the analysis step calls a pass-through result "detail"; the API calls it "list"
            kind, row_count = ("list" if result.get("kind") == "detail" else "aggregate"), len(raw)
            limit_reached = bool(result.get("truncated"))
        return JobOutcome(
            rows=rows, kind=kind, row_count=row_count, limit_reached=limit_reached,
            workflow_id=workflow_id, succeeded=ok,
            task_statuses={k: v.status.value for k, v in tasks.items()},
            artifact=artifact, failed_task_key=failed_key, error_code=code, message=message,
        )


__all__ = ["READ_ACTION", "TableReportRequest", "build_task_specs",
           "JobOutcome", "TableReportJobRunner"]
