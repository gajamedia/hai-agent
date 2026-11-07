from __future__ import annotations

"""Turn a natural-language question into a validated ``TableReportRequest``.

The LLM only *chooses* from an application-owned catalog (tables, columns,
aggregation operations, output formats). It never writes SQL, never supplies
column types, and never controls data-quality rules, row limits or which
permissions apply: those come from the trusted catalog. Anything the LLM
returns is validated; an invalid answer is sent back once with the exact error,
and after that the planner reports failure instead of guessing.

Questions that need things this job type cannot do (filters, joins, time
series) are answered with ``supported=False`` and a reason, not forced into a
wrong report.
"""

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from agent.llm.contracts import LLMMessage
from agent.workflow.dataset import SUPPORTED_DATA_TYPES
from agent.workflow.table_report_job import (
    DEFAULT_LIST_LIMIT,
    MAX_FILTER_VALUE_LENGTH,
    MAX_FILTERS,
    MAX_LIST_COLUMNS,
    MAX_ORDER_BY,
    TableReportRequest,
)

ALLOWED_OPERATIONS = ("count", "count_distinct", "sum", "avg", "min", "max")
NUMERIC_TYPES = frozenset({"integer", "number"})
REPORT_FORMATS = ("json", "csv", "html", "xlsx", "pdf")
_OUTPUT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_GROUP_BY = 3
MAX_METRICS = 5


@dataclass(frozen=True)
class TableCatalogEntry:
    """Trusted description of one approved table."""

    table: str
    columns: Mapping[str, str]  # column -> data type
    description: str = ""
    key_column: str | None = None  # unique, non-null id; used for row counts
    required_non_null_columns: tuple[str, ...] = ()
    row_limit: int = 500
    filterable_columns: tuple[str, ...] = ()

    @classmethod
    def from_table_spec(
        cls, spec: Any, column_types: Mapping[str, str], *, description: str = "",
        key_column: str | None = None,
        required_non_null_columns: tuple[str, ...] = (),
    ) -> "TableCatalogEntry":
        """Derive the catalog from the adapter's own allowlist so they cannot drift.

        Columns, filterable columns and the row cap come from ``spec`` (a
        ``ReadOnlyTableSpec``); only the data types are supplied separately.
        """
        missing = [c for c in spec.columns if c not in column_types]
        if missing:
            raise ValueError(f"data types missing for columns: {missing}")
        return cls(
            table=spec.table,
            columns={c: column_types[c] for c in spec.columns},
            description=description, key_column=key_column,
            required_non_null_columns=required_non_null_columns,
            row_limit=spec.max_limit, filterable_columns=tuple(spec.filterable_columns),
        )

    def __post_init__(self) -> None:
        bad = {c: t for c, t in self.columns.items() if t not in SUPPORTED_DATA_TYPES}
        if bad:
            raise ValueError(f"unsupported data types in catalog: {bad}")
        known = set(self.columns)
        extra = (({self.key_column} if self.key_column else set())
                 | set(self.required_non_null_columns) | set(self.filterable_columns))
        if extra - known:
            raise ValueError(f"catalog references unknown columns: {sorted(extra - known)}")


@dataclass(frozen=True)
class TablePlanResult:
    question: str
    request: TableReportRequest | None
    reason: str | None = None  # why there is no request
    attempts: int = 0
    provider: str | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.request is not None


class PlanValidationError(ValueError):
    """The LLM answer does not satisfy the catalog; the message is sent back to it."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise PlanValidationError("Answer must be a single JSON object.")
    try:
        data = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"Answer is not valid JSON: {exc}.") from exc
    if not isinstance(data, dict):
        raise PlanValidationError("Answer must be a JSON object.")
    return data


class TableReportPlanner:
    def __init__(self, llm_service: Any, catalog: Sequence[TableCatalogEntry],
                 *, max_attempts: int = 2, experience: Any = None) -> None:
        if not catalog:
            raise ValueError("catalog must not be empty")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.llm = llm_service
        self.catalog = {entry.table: entry for entry in catalog}
        if len(self.catalog) != len(catalog):
            raise ValueError("duplicate table in catalog")
        self.max_attempts = max_attempts
        # Optional ConnectionExperience: verified filter values and past plans.
        self.experience = experience

    def _apply_experience(self, request: TableReportRequest) -> TableReportRequest:
        """Correct string filter values to their verified spelling (case/space only)."""
        if self.experience is None or not request.filters:
            return request
        fixed = {
            column: (self.experience.canonical_value(request.table, column, value)
                     if isinstance(value, str) else value)
            for column, value in request.filters.items()
        }
        return request if fixed == dict(request.filters) else replace(request, filters=fixed)

    # ------------------------------------------------------------------ API
    def plan(self, question: str) -> TablePlanResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        question = question.strip()
        learned = self.experience.prompt_block() if self.experience is not None else ""
        messages = [LLMMessage("system", self._system_prompt() + learned),
                    LLMMessage("user", f"Question:\n{question}")]
        errors: list[str] = []
        provider = None
        for attempt in range(1, self.max_attempts + 1):
            response = self.llm.complete(messages, temperature=0)
            provider = response.provider
            try:
                data = _extract_json(response.content)
                if data.get("supported") is False:
                    reason = str(data.get("reason") or "The question cannot be answered.").strip()
                    return TablePlanResult(question, None, reason, attempt, provider, tuple(errors))
                request = self._apply_experience(self._validate(data))
                return TablePlanResult(question, request, None, attempt, provider, tuple(errors))
            except PlanValidationError as exc:
                errors.append(str(exc))
                messages += [LLMMessage("assistant", response.content),
                             LLMMessage("user", f"Invalid: {exc} Reply again with ONLY the corrected JSON object.")]
        return TablePlanResult(question, None,
                               f"Could not produce a valid plan: {errors[-1]}",
                               self.max_attempts, provider, tuple(errors))

    # ------------------------------------------------------------ internals
    @staticmethod
    def _filter_value_matches(kind: str, value: Any) -> bool:
        if kind == "string" or kind in {"date", "datetime"}:
            return isinstance(value, str)
        if kind == "boolean":
            return isinstance(value, bool)
        if isinstance(value, bool):
            return False
        if kind == "integer":
            return isinstance(value, int)
        if kind == "number":
            return isinstance(value, (int, float)) and math.isfinite(value)
        return isinstance(value, (str, int, float))  # "any"

    @classmethod
    def _validate_filters(cls, entry: TableCatalogEntry, raw: Any) -> dict[str, Any]:
        if raw is None or raw == {}:
            return {}
        if not isinstance(raw, Mapping) or len(raw) > MAX_FILTERS:
            raise PlanValidationError(f"filters must be an object with at most {MAX_FILTERS} items.")
        result: dict[str, Any] = {}
        for column, value in raw.items():
            if column not in entry.filterable_columns:
                raise PlanValidationError(
                    f"Cannot filter on {column!r}; filterable columns: "
                    f"{list(entry.filterable_columns)}.")
            kind = entry.columns[column]
            if not cls._filter_value_matches(kind, value):
                raise PlanValidationError(
                    f"Filter value for {column!r} must match its type ({kind}), got {value!r}.")
            if isinstance(value, str):
                if not value.strip() or len(value) > MAX_FILTER_VALUE_LENGTH:
                    raise PlanValidationError(f"Filter value for {column!r} is empty or too long.")
                if kind in {"date", "datetime"}:
                    try:
                        (date if kind == "date" else datetime).fromisoformat(value)
                    except ValueError as exc:
                        raise PlanValidationError(
                            f"Filter value for {column!r} must be an ISO {kind}.") from exc
            result[column] = value
        return result

    def _system_prompt(self) -> str:
        catalog = [
            {"table": e.table, "description": e.description,
             "columns": dict(e.columns), "key_column": e.key_column,
             "filterable_columns": list(e.filterable_columns),
             "list_limit_max": e.row_limit - 1}
            for e in self.catalog.values()
        ]
        return (
            "You plan a single-table report. Choose ONLY from this catalog; "
            "never invent tables or columns.\n"
            f"CATALOG:\n{json.dumps(catalog, ensure_ascii=False)}\n\n"
            f"Allowed operations: {', '.join(ALLOWED_OPERATIONS)} "
            "(sum and avg need an integer/number column). "
            f"Allowed formats: {', '.join(REPORT_FORMATS)} (default json).\n"
            "There are two kinds of report. kind \"aggregate\" (the default) summarizes: "
            "grouping by 1-3 columns and computing 1-5 metrics over one table, "
            "optionally with up to 3 EQUALITY filters (column = value) on that table's "
            "filterable_columns only. Use the filter value exactly as the user wrote it.\n"
            "NOT supported: ranges or comparisons (>, <, between, date ranges), OR conditions, "
            "'contains', filters on columns not in filterable_columns, joins, several tables, "
            "raw SQL (you never write SQL), forecasts.\n"
            "If the question needs anything unsupported, answer "
            '{"supported": false, "reason": "<short reason, in the question\'s language>"}.\n'
            "Otherwise answer ONLY with this JSON (no markdown):\n"
            '{"supported": true, "table": "...", "group_by": ["col"], '
            '"metrics": [{"output": "snake_case_name", "operation": "sum", "column": "col"}], '
            '"filters": {"filterable_col": "value"}, '
            '"title": "short title", "format": "json"}\n'
            'Omit "filters" when the question has no condition. '
            'For a row count use operation "count" and omit "column".\n'
            'kind "list" SHOWS THE ROWS THEMSELVES: use it for requests such as "tampilkan", '
            '"daftar", "lihat data", "show", "list", "display". Answer ONLY with this JSON:\n'
            '{"supported": true, "kind": "list", "table": "...", "columns": ["col1", "col2"], '
            '"filters": {"filterable_col": "value"}, '
            '"order_by": [{"column": "col", "direction": "desc"}], "limit": 100, '
            '"title": "short title", "format": "json"}\n'
            'columns: the columns the user asks for, or "all" for every column. order_by is optional '
            '(at most 3 items, direction asc or desc). limit is the number of rows: 100 unless the user '
            "asks for another number, and never more than the table's list_limit_max. "
            "Use kind aggregate for totals, counts and averages."
        )

    def _validate_list(self, entry: TableCatalogEntry, data: Mapping[str, Any]) -> TableReportRequest:
        """A "show the rows" plan: columns, optional equality filters, order and a row limit."""
        columns = entry.columns
        raw_columns = data.get("columns")
        if raw_columns == "all" or raw_columns == ["*"]:
            selected = list(columns)[:MAX_LIST_COLUMNS]
        else:
            if (not isinstance(raw_columns, list) or not raw_columns
                    or len(raw_columns) > MAX_LIST_COLUMNS
                    or any(not isinstance(c, str) for c in raw_columns)
                    or len(set(raw_columns)) != len(raw_columns)):
                raise PlanValidationError(
                    f"columns must be 1-{MAX_LIST_COLUMNS} distinct column names, or \"all\".")
            unknown = [c for c in raw_columns if c not in columns]
            if unknown:
                raise PlanValidationError(f"Unknown columns {unknown}; columns: {sorted(columns)}.")
            selected = list(raw_columns)

        raw_order = data.get("order_by", [])
        if raw_order is None:
            raw_order = []
        if not isinstance(raw_order, list) or len(raw_order) > MAX_ORDER_BY:
            raise PlanValidationError(f"order_by must be a list of at most {MAX_ORDER_BY} items.")
        order: list[tuple[str, str]] = []
        for item in raw_order:
            if not isinstance(item, Mapping):
                raise PlanValidationError('Each order_by item must be {"column": ..., "direction": "asc"|"desc"}.')
            column, direction = item.get("column"), str(item.get("direction", "asc")).lower()
            if not isinstance(column, str) or column not in columns:
                raise PlanValidationError(f"Unknown order_by column {column!r}; columns: {sorted(columns)}.")
            if direction not in {"asc", "desc"}:
                raise PlanValidationError("order_by direction must be asc or desc.")
            if any(column == existing for existing, _ in order):
                raise PlanValidationError(f"order_by lists {column!r} twice.")
            order.append((column, direction))

        maximum = entry.row_limit - 1
        limit = data.get("limit", min(DEFAULT_LIST_LIMIT, maximum))
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
            raise PlanValidationError(
                f"limit must be a whole number between 1 and {maximum} (list_limit_max of this table).")

        filters = self._validate_filters(entry, data.get("filters"))
        fmt = data.get("format", "json")
        if fmt not in REPORT_FORMATS:
            raise PlanValidationError(f"format must be one of {list(REPORT_FORMATS)}.")
        title = data.get("title")
        title = title.strip()[:200] if isinstance(title, str) and title.strip() else f"Rows: {entry.table}"

        used = set(selected) | {c for c, _ in order}
        try:
            return TableReportRequest(
                table=entry.table, columns={n: columns[n] for n in columns if n in used},
                group_by=(), metrics=(), title=title, report_format=fmt, row_limit=entry.row_limit,
                filters=filters, kind="list", select_columns=tuple(selected),
                order_by=tuple(order), limit=limit)
        except ValueError as exc:
            raise PlanValidationError(str(exc)) from exc

    def _validate(self, data: Mapping[str, Any]) -> TableReportRequest:
        table = data.get("table")
        entry = self.catalog.get(table) if isinstance(table, str) else None
        if entry is None:
            raise PlanValidationError(
                f"Unknown table {table!r}. Choose one of: {sorted(self.catalog)}.")
        columns = entry.columns
        kind = data.get("kind", "aggregate")
        if kind == "list":
            return self._validate_list(entry, data)
        if kind != "aggregate":
            raise PlanValidationError(f"kind must be 'aggregate' or 'list', got {kind!r}.")

        group_by = data.get("group_by")
        if (not isinstance(group_by, list) or not group_by or len(group_by) > MAX_GROUP_BY
                or any(not isinstance(c, str) for c in group_by) or len(set(group_by)) != len(group_by)):
            raise PlanValidationError(f"group_by must be 1-{MAX_GROUP_BY} distinct column names.")
        unknown = [c for c in group_by if c not in columns]
        if unknown:
            raise PlanValidationError(f"Unknown group_by columns {unknown}; columns: {sorted(columns)}.")

        raw_metrics = data.get("metrics")
        if not isinstance(raw_metrics, list) or not 1 <= len(raw_metrics) <= MAX_METRICS:
            raise PlanValidationError(f"metrics must contain 1-{MAX_METRICS} items.")
        metrics: list[tuple[str, str, str]] = []
        used_outputs = set(group_by)
        for item in raw_metrics:
            if not isinstance(item, Mapping):
                raise PlanValidationError("Each metric must be an object.")
            output, operation, column = item.get("output"), item.get("operation"), item.get("column")
            if not isinstance(output, str) or not _OUTPUT_NAME.match(output):
                raise PlanValidationError(
                    f"Metric output {output!r} must be snake_case (a-z, 0-9, _), max 40 chars.")
            if output in used_outputs:
                raise PlanValidationError(f"Duplicate output name {output!r}.")
            if operation not in ALLOWED_OPERATIONS:
                raise PlanValidationError(
                    f"Operation {operation!r} is not allowed; use one of {list(ALLOWED_OPERATIONS)}.")
            if column is None and operation == "count":
                column = entry.key_column
                if column is None:
                    raise PlanValidationError("This table has no key column; give a column for count.")
            if not isinstance(column, str) or column not in columns:
                raise PlanValidationError(f"Unknown metric column {column!r}; columns: {sorted(columns)}.")
            if operation in {"sum", "avg"} and columns[column] not in NUMERIC_TYPES:
                raise PlanValidationError(
                    f"{operation} needs a numeric column but {column!r} is {columns[column]}.")
            used_outputs.add(output)
            metrics.append((output, operation, column))

        filters = self._validate_filters(entry, data.get("filters"))

        fmt = data.get("format", "json")
        if fmt not in REPORT_FORMATS:
            raise PlanValidationError(f"format must be one of {list(REPORT_FORMATS)}.")
        title = data.get("title")
        title = title.strip()[:200] if isinstance(title, str) and title.strip() else f"Report: {entry.table}"

        used = set(group_by) | {m[2] for m in metrics}
        used_columns = {name: columns[name] for name in columns if name in used
                        or name in entry.required_non_null_columns
                        or name == entry.key_column}
        numeric = tuple(c for c in used_columns if used_columns[c] in NUMERIC_TYPES
                        and c in {m[2] for m in metrics})
        try:
            return TableReportRequest(
                table=entry.table, columns=used_columns, group_by=tuple(group_by),
                metrics=tuple(metrics), title=title, report_format=fmt,
                row_limit=entry.row_limit,
                required_non_null_columns=tuple(
                    c for c in entry.required_non_null_columns if c in used_columns)
                + ((entry.key_column,) if entry.key_column and entry.key_column not in entry.required_non_null_columns else ()),
                unique_key_columns=(entry.key_column,) if entry.key_column else (),
                numeric_columns=numeric,
                categorical_columns=tuple(c for c in group_by if columns[c] == "string"),
                filters=filters,
            )
        except ValueError as exc:
            raise PlanValidationError(str(exc)) from exc


__all__ = ["TableCatalogEntry", "TablePlanResult", "TableReportPlanner", "PlanValidationError",
           "ALLOWED_OPERATIONS", "REPORT_FORMATS"]
