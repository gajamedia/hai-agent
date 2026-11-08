from __future__ import annotations

"""Turn a natural-language question into a validated ``TableReportRequest``.

The LLM only *chooses* from an application-owned catalog (tables, columns,
aggregation operations, output formats). It never writes SQL, never supplies
column types, and never controls data-quality rules, row limits or which
permissions apply: those come from the trusted catalog. Anything the LLM
returns is validated; an invalid answer is sent back once with the exact error,
and after that the planner reports failure instead of guessing.

Joins (listings and totals) are possible only along ``TableRelation`` s the user
approved (the database's foreign keys, or relations the user declared). Questions
that need things this job type cannot do (time series, joins outside the approved
relations) are answered with ``supported=False`` and a
reason, not forced into a wrong report.
"""

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Callable, Mapping, Sequence

from agent.llm.business_database_adapter import (
    CONDITION_OPERATORS, JOIN_TYPES, MAX_CONDITIONS, MAX_IN_VALUES, MAX_JOINS, join_output_names,
    split_qualified,
)
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

_COMPARABLE = frozenset({"eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "is_null", "is_not_null"})
_TEXT = frozenset({"eq", "ne", "in", "contains", "starts_with", "ends_with", "is_null", "is_not_null"})
_FLAG = frozenset({"eq", "ne", "is_null", "is_not_null"})
# which operators make sense for which kind of column (a text column is not "greater than" anything)
OPERATORS_BY_TYPE = {"integer": _COMPARABLE, "number": _COMPARABLE, "date": _COMPARABLE,
                     "datetime": _COMPARABLE, "string": _TEXT, "boolean": _FLAG,
                     "any": frozenset({"eq", "ne", "in", "is_null", "is_not_null"})}

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
class TableRelation:
    """Two columns of two approved tables that may be joined (``source``: the database's own foreign
    key, or one the user declared because the database does not define it)."""

    table: str
    column: str
    ref_table: str
    ref_column: str
    source: str = "foreign_key"

    def __post_init__(self) -> None:
        if self.table == self.ref_table:
            raise ValueError("a relation must connect two different tables")
        if self.source not in {"foreign_key", "manual", "ai"}:
            raise ValueError("relation source must be foreign_key, manual or ai")

    @property
    def left(self) -> str:
        return f"{self.table}.{self.column}"

    @property
    def right(self) -> str:
        return f"{self.ref_table}.{self.ref_column}"


@dataclass(frozen=True)
class TablePlanResult:
    question: str
    request: TableReportRequest | None
    reason: str | None = None  # why there is no request
    attempts: int = 0
    provider: str | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)
    # unsupported only because the question needs several tables that cannot be combined (no approved relation)
    needs_join: bool = False
    # relations the AI found for this question; they are used by the plan but NOT approved until a person
    # runs the plan (see TableReportService.run_plan)
    proposed_relations: tuple["TableRelation", ...] = ()

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


def condition_scalar(kind: str, column: str, value: Any) -> Any:
    """One value of a condition, checked against the column's type."""
    if kind in {"integer", "number"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PlanValidationError(
                f"A condition on {column!r} ({kind}) needs a plain number, got {value!r}. "
                'Convert "30 ribu" to 30000.')
        return value
    if kind in {"date", "datetime"}:
        if not isinstance(value, str):
            raise PlanValidationError(f"A condition on {column!r} ({kind}) needs an ISO date string.")
        try:
            (date if kind == "date" else datetime).fromisoformat(value)
        except ValueError as exc:
            raise PlanValidationError(f"Value {value!r} for {column!r} must be an ISO {kind}.") from exc
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            raise PlanValidationError(f"A condition on {column!r} (boolean) needs true or false.")
        return value
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_FILTER_VALUE_LENGTH:
        raise PlanValidationError(f"A condition on {column!r} needs non-empty text (max {MAX_FILTER_VALUE_LENGTH} characters).")
    return value

def validate_conditions(entry: TableCatalogEntry, raw: Any) -> tuple[tuple[str, str, Any], ...]:
    """Comparisons beyond equality: a closed set of operators on approved, filterable columns."""
    if raw is None or raw == []:
        return ()
    if not isinstance(raw, list) or len(raw) > MAX_CONDITIONS:
        raise PlanValidationError(f"conditions must be a list of at most {MAX_CONDITIONS} items.")
    result: list[tuple[str, str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) - {"column", "operator", "value"}:
            raise PlanValidationError('Each condition must be {"column": ..., "operator": ..., "value": ...}.')
        column, operator, value = item.get("column"), item.get("operator"), item.get("value")
        if not isinstance(column, str) or column not in entry.filterable_columns:
            raise PlanValidationError(
                f"Cannot filter on {column!r}; filterable columns: {list(entry.filterable_columns)}.")
        if operator not in CONDITION_OPERATORS:
            raise PlanValidationError(
                f"Operator {operator!r} is not allowed; use one of {sorted(CONDITION_OPERATORS)}.")
        kind = entry.columns[column]
        allowed = OPERATORS_BY_TYPE.get(kind, OPERATORS_BY_TYPE["any"])
        if operator not in allowed:
            raise PlanValidationError(
                f"{operator!r} cannot be used on {column!r} ({kind}); allowed: {sorted(allowed)}.")
        if operator in {"is_null", "is_not_null"}:
            if value is not None:
                raise PlanValidationError(f"{operator!r} takes no value.")
            result.append((column, operator, None))
        elif operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise PlanValidationError("'between' needs a value like [low, high].")
            low, high = (condition_scalar(kind, column, v) for v in value)
            if low > high:
                raise PlanValidationError(f"'between' on {column!r}: low {low!r} is greater than high {high!r}.")
            result.append((column, operator, (low, high)))
        elif operator == "in":
            if not isinstance(value, list) or not 1 <= len(value) <= MAX_IN_VALUES:
                raise PlanValidationError(f"'in' needs a list of 1-{MAX_IN_VALUES} values.")
            result.append((column, operator, tuple(condition_scalar(kind, column, v) for v in value)))
        else:
            result.append((column, operator, condition_scalar(kind, column, value)))
    return tuple(result)


class TableReportPlanner:
    def __init__(self, llm_service: Any, catalog: Sequence[TableCatalogEntry],
                 *, max_attempts: int = 2, experience: Any = None,
                 today: Callable[[], date] = date.today,
                 relations: Sequence[TableRelation] = (),
                 relation_hints: Sequence[TableRelation] = (),
                 propose_relations: bool = True) -> None:
        if not catalog:
            raise ValueError("catalog must not be empty")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.llm = llm_service
        self.catalog = {entry.table: entry for entry in catalog}
        if len(self.catalog) != len(catalog):
            raise ValueError("duplicate table in catalog")
        self.max_attempts = max_attempts
        for rel in relations:
            for table, column in ((rel.table, rel.column), (rel.ref_table, rel.ref_column)):
                if table not in self.catalog or column not in self.catalog[table].columns:
                    raise ValueError(
                        f"relation {rel.left} = {rel.right} uses a table or column that is not in the catalog")
        self.relations = tuple(relations)
        # Guesses from column names (never approved) that help the AI propose a relation, and whether it may.
        self.relation_hints = tuple(relation_hints)
        self.propose_relations = propose_relations
        # Optional ConnectionExperience: verified filter values and past plans.
        self.experience = experience
        self._today = today

    def _apply_experience(self, request: TableReportRequest) -> TableReportRequest:
        """Correct string filter values to their verified spelling (case/space only)."""
        if self.experience is None or not request.filters or request.joins:
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
                    result = TablePlanResult(question, None, reason, attempt, provider, tuple(errors),
                                             needs_join=data.get("needs_join") is True)
                    if result.needs_join and self.propose_relations and len(self.catalog) > 1:
                        return self._plan_with_proposed_relations(question, result)
                    return result
                request = self._apply_experience(self._validate(data))
                return TablePlanResult(question, request, None, attempt, provider, tuple(errors))
            except PlanValidationError as exc:
                errors.append(str(exc))
                messages += [LLMMessage("assistant", response.content),
                             LLMMessage("user", f"Invalid: {exc} Reply again with ONLY the corrected JSON object.")]
        return TablePlanResult(question, None,
                               f"Could not produce a valid plan: {errors[-1]}",
                               self.max_attempts, provider, tuple(errors))

    # ------------------------------------------------- relations found by the AI
    def _propose_relations(self, question: str) -> tuple[TableRelation, ...]:
        """Ask the AI how the tables in the catalog are connected, and keep only what can be verified.

        A proposal must use approved tables and columns of the same type, and one side must be the key
        column of its table (a foreign key points at a key). Anything else is dropped. One attempt only:
        if the AI does not answer, the plain "unsupported" answer stands."""
        import logging
        catalog = [{"table": e.table, "columns": dict(e.columns), "key_column": e.key_column}
                   for e in self.catalog.values()]
        hints = [{"a": r.left, "b": r.right} for r in self.relation_hints]
        system = (
            "You find how the tables of a database are connected, so a question that needs several tables can "
            "be answered. Choose ONLY from this catalog; never invent tables or columns.\n"
            f"CATALOG:\n{json.dumps(catalog, ensure_ascii=False)}\n"
            f"GUESSES FROM COLUMN NAMES (may be wrong, check them against the question): {json.dumps(hints)}\n"
            "A relation says: column `column` of table `table` holds the same value as the key_column of table "
            "`ref_table` (a foreign key, for example items.id_category = categories.id). Propose only the "
            "relations the question needs, at most 3, and only when the names make the connection clear. "
            "ref_column MUST be the key_column of ref_table. Answer ONLY with JSON (no markdown):\n"
            '{"relations": [{"table": "...", "column": "...", "ref_table": "...", "ref_column": "..."}]}\n'
            'or {"relations": []} when no connection is clear.')
        try:
            response = self.llm.complete([LLMMessage("system", system),
                                          LLMMessage("user", f"Question:\n{question}")], temperature=0)
            raw = _extract_json(response.content).get("relations")
        except Exception:  # noqa: BLE001 - this step must never turn an answer into an error
            logging.getLogger(__name__).info("Relation proposal failed.", exc_info=True)
            return ()
        if not isinstance(raw, list):
            return ()
        known = {frozenset({r.left, r.right}) for r in self.relations}
        found: list[TableRelation] = []
        for item in raw[:3]:
            if not isinstance(item, Mapping) or set(item) - {"table", "column", "ref_table", "ref_column"}:
                continue
            parts = [item.get(k) for k in ("table", "column", "ref_table", "ref_column")]
            if not all(isinstance(p, str) for p in parts):
                continue
            table, column, ref_table, ref_column = parts
            left, right = self.catalog.get(table), self.catalog.get(ref_table)
            if (left is None or right is None or left is right or column not in left.columns
                    or ref_column not in right.columns or left.columns[column] != right.columns[ref_column]):
                continue
            left_key, right_key = left.key_column == column, right.key_column == ref_column
            if not (left_key or right_key):
                continue
            if left_key and not right_key:  # written the other way round: the key side is the reference
                table, column, ref_table, ref_column = ref_table, ref_column, table, column
            key = frozenset({f"{table}.{column}", f"{ref_table}.{ref_column}"})
            if key in known:
                continue
            known.add(key)
            found.append(TableRelation(table, column, ref_table, ref_column, "ai"))
        return tuple(found)

    def _plan_with_proposed_relations(self, question: str, original: TablePlanResult) -> TablePlanResult:
        """The question needs a join nobody approved: let the AI propose the relation, then plan again with it.

        The result carries ``proposed_relations`` so the person sees them in the plan; they only become
        approved when that plan is run."""
        proposed = self._propose_relations(question)
        if not proposed:
            return original
        inner = TableReportPlanner(
            self.llm, list(self.catalog.values()), max_attempts=self.max_attempts, experience=self.experience,
            today=self._today, relations=(*self.relations, *proposed), propose_relations=False)
        result = inner.plan(question)
        if not result.ok:
            return original
        pairs = {(r.left, r.right): r for r in proposed} | {(r.right, r.left): r for r in proposed}
        used: list[TableRelation] = []
        for _, _, first, second in result.request.joins:
            rel = pairs.get((first, second))
            if rel is not None and rel not in used:
                used.append(rel)
        return replace(result, proposed_relations=tuple(used),
                       attempts=original.attempts + result.attempts, errors=original.errors + result.errors)

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
        joinable = bool(self.relations)
        return (
            "You plan a table report. Choose ONLY from this catalog; "
            "never invent tables or columns.\n"
            f"CATALOG:\n{json.dumps(catalog, ensure_ascii=False)}\n\n"
            f"Allowed operations: {', '.join(ALLOWED_OPERATIONS)} "
            "(sum and avg need an integer/number column). "
            f"Allowed formats: {', '.join(REPORT_FORMATS)} (default json).\n"
            "There are two kinds of report. kind \"aggregate\" (the default) summarizes: "
            "grouping by 1-3 columns and computing 1-5 metrics over one table, "
            "optionally with up to 3 EQUALITY filters (column = value) on that table's "
            "filterable_columns only. Use the filter value exactly as the user wrote it.\n"
            "NOT supported: OR conditions, filters or conditions on columns not in filterable_columns, "
            + ("joins that are not in RELATIONS, "
               if joinable else "joins, several tables, ")
            + "raw SQL (you never write SQL), calculations inside a condition, "
            "forecasts, anything that changes data.\n"
            f"Today is {self._today().isoformat()}.\n"
            "If the question needs anything unsupported, answer "
            '{"supported": false, "reason": "<short reason, in the question\'s language>"}. '
            'Add "needs_join": true when the ONLY obstacle is that the question needs columns of several '
            "tables that cannot be combined.\n"
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
            "Use kind aggregate for totals, counts and averages.\n"
            'CONDITIONS go beyond equality, for both kinds: "conditions": [{"column": "filterable_col", '
            '"operator": "lt", "value": 30000}]. Operators: eq, ne, gt, gte, lt, lte (numbers and dates), '
            'between (value [low, high], inclusive), in (value: a list of up to 20), contains, starts_with, '
            "ends_with (text columns), is_null, is_not_null (no value). Only on filterable_columns; at most "
            f"{MAX_CONDITIONS} conditions, all combined with AND. Dates are ISO yyyy-mm-dd. Give numbers as "
            'plain numbers: "30 ribu" is 30000, "2 juta" is 2000000, "1,5 juta" is 1500000. For a month or a '
            "year use between its first and last day (use today's date to resolve \"this month\" or "
            '"last year"). Plain equality can stay in "filters".'
            + (self._join_prompt() if joinable else "")
        )

    def _join_prompt(self) -> str:
        relations = [{"a": r.left, "b": r.right} for r in self.relations]
        return (
            "\nJOINS: a listing (kind list) may combine tables, but ONLY along these approved RELATIONS "
            f"(each pair of columns may be matched): {json.dumps(relations)}\n"
            f"Use up to {MAX_JOINS} joins, each to a table that has a relation with the table the listing "
            'starts from or with a table already joined. Answer with: {"supported": true, "kind": "list", '
            '"table": "start_table", "joins": [{"table": "other_table", "type": "left"}], '
            '"columns": ["start_table.col", "other_table.col"], "filters": {"table.col": "value"}, '
            '"conditions": [{"column": "table.col", "operator": "eq", "value": 1}], '
            '"order_by": [{"column": "table.col", "direction": "asc"}], "limit": 100, "title": "...", '
            '"format": "json"}. In a joined listing EVERY column (columns, filters, conditions, order_by) is '
            'written table.col and columns may not be "all". Use type "left" when rows without a match must '
            'stay (for example items without a category), otherwise "inner" (the default). Leave "on" out: '
            "the relation is found from the tables; give it as \"on\": [\"a.col\", \"b.col\"] only when "
            "two tables are connected by more than one relation. Do not join when one table has all the "
            "columns that were asked for.\n"
            'A total, count or average over joined tables is kind "aggregate" with the same "joins": '
            '{"supported": true, "table": "start_table", "joins": [{"table": "other_table", "type": "inner"}], '
            '"group_by": ["other_table.col"], "metrics": [{"output": "snake_case_name", "operation": "sum", '
            '"column": "start_table.col"}], "filters": {...}, "title": "...", "format": "json"}. Every column '
            "is written table.col; a count without a column counts the start table's rows. Use the start "
            "table that holds the values being summed or counted. Use type \"left\" when groups or rows "
            "without a match must still appear."
        )

    @staticmethod
    def _condition_scalar(kind: str, column: str, value: Any) -> Any:
        return condition_scalar(kind, column, value)

    def _validate_conditions(self, entry: TableCatalogEntry, raw: Any) -> tuple[tuple[str, str, Any], ...]:
        return validate_conditions(entry, raw)

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
        conditions = self._validate_conditions(entry, data.get("conditions"))
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
                filters=filters, conditions=conditions, kind="list", select_columns=tuple(selected),
                order_by=tuple(order), limit=limit)
        except ValueError as exc:
            raise PlanValidationError(str(exc)) from exc

    def _join_scope(self, entry, raw):
        """The joined tables and, for each, the approved relation that connects it."""
        if not self.relations:
            raise PlanValidationError("Joins are not available: no relation between tables is approved.")
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_JOINS:
            raise PlanValidationError(f"joins must be a list of 1-{MAX_JOINS} items.")
        scope = [entry]
        joins: list[tuple[str, str, str, str]] = []
        for item in raw:
            if not isinstance(item, Mapping) or set(item) - {"table", "type", "on"}:
                raise PlanValidationError('Each join must be {"table": ..., "type": "inner"|"left"}.')
            table, kind, on = item.get("table"), item.get("type", "inner"), item.get("on")
            joined = self.catalog.get(table) if isinstance(table, str) else None
            if joined is None:
                raise PlanValidationError(f"Unknown join table {table!r}. Choose one of: {sorted(self.catalog)}.")
            if any(joined.table == e.table for e in scope):
                raise PlanValidationError(f"{table!r} is already in the query; a table can be joined once.")
            if kind not in JOIN_TYPES:
                raise PlanValidationError("A join type must be inner or left.")
            present = {e.table for e in scope}
            candidates = [(r.left, r.right) if r.table == table else (r.right, r.left)
                          for r in self.relations
                          if (r.table == table and r.ref_table in present)
                          or (r.ref_table == table and r.table in present)]
            if on is not None:
                if not isinstance(on, list) or len(on) != 2 or any(not isinstance(x, str) for x in on):
                    raise PlanValidationError('"on" must be ["table.col", "table.col"].')
                pair = (on[0], on[1]) if on[0].startswith(f"{table}.") else (on[1], on[0])
                if pair not in candidates:
                    raise PlanValidationError(
                        f"{on} is not an approved relation; approved for {table!r}: "
                        f"{[list(c) for c in candidates]}.")
                chosen = pair
            elif len(candidates) == 1:
                chosen = candidates[0]
            elif not candidates:
                raise PlanValidationError(
                    f"No approved relation connects {table!r} to {sorted(present)}. "
                    f"Approved relations: {[[r.left, r.right] for r in self.relations]}. "
                    "If the question cannot be answered with them, answer supported false.")
            else:
                raise PlanValidationError(
                    f'{table!r} has several relations {[list(c) for c in candidates]}; give "on".')
            scope.append(joined)
            joins.append((table, kind, chosen[0], chosen[1]))
        return scope, joins

    @staticmethod
    def _resolver(scope):
        by_table = {e.table: e for e in scope}

        def resolve(name, what):
            try:
                table, column = split_qualified(name)
            except ValueError:
                raise PlanValidationError(f"{what} {name!r} must be written table.column.") from None
            found = by_table.get(table)
            if found is None or column not in found.columns:
                raise PlanValidationError(
                    f"Unknown {what} {name!r}; use columns of {sorted(by_table)} written table.column.")
            return found, column
        return resolve

    def _join_conditions(self, resolve, data):
        """Equality filters and conditions of a joined report; each column is table.column."""
        raw_filters = data.get("filters") or {}
        if not isinstance(raw_filters, Mapping) or len(raw_filters) > MAX_FILTERS:
            raise PlanValidationError(f"filters must be an object with at most {MAX_FILTERS} items.")
        filters: dict[str, Any] = {}
        for name, value in raw_filters.items():
            found, column = resolve(name, "filter column")
            self._validate_filters(found, {column: value})
            filters[name] = value
        raw_conditions = data.get("conditions") or []
        if not isinstance(raw_conditions, list) or len(raw_conditions) > MAX_CONDITIONS:
            raise PlanValidationError(f"conditions must be a list of at most {MAX_CONDITIONS} items.")
        conditions: list[tuple[str, str, Any]] = []
        for item in raw_conditions:
            if not isinstance(item, Mapping):
                raise PlanValidationError('Each condition must be {"column": ..., "operator": ..., "value": ...}.')
            found, column = resolve(item.get("column"), "condition column")
            (checked,) = validate_conditions(found, [{**item, "column": column}])
            conditions.append((item["column"], checked[1], checked[2]))
        return filters, tuple(conditions)

    def _validate_join_aggregate(self, entry, data):
        """Totals, counts and averages over the start table plus joined tables."""
        scope, joins = self._join_scope(entry, data.get("joins"))
        resolve = self._resolver(scope)
        group_by = data.get("group_by")
        if (not isinstance(group_by, list) or not group_by or len(group_by) > MAX_GROUP_BY
                or len(set(map(str, group_by))) != len(group_by)):
            raise PlanValidationError(f"group_by must be 1-{MAX_GROUP_BY} distinct table.column names.")
        for name in group_by:
            resolve(name, "group_by column")
        raw_metrics = data.get("metrics")
        if not isinstance(raw_metrics, list) or not 1 <= len(raw_metrics) <= MAX_METRICS:
            raise PlanValidationError(f"metrics must contain 1-{MAX_METRICS} items.")
        metrics: list[tuple[str, str, str]] = []
        used = list(group_by)
        outputs: set[str] = set()
        for item in raw_metrics:
            if not isinstance(item, Mapping):
                raise PlanValidationError("Each metric must be an object.")
            output, operation, column = item.get("output"), item.get("operation"), item.get("column")
            if not isinstance(output, str) or not _OUTPUT_NAME.match(output):
                raise PlanValidationError(
                    f"Metric output {output!r} must be snake_case (a-z, 0-9, _), max 40 chars.")
            if output in outputs:
                raise PlanValidationError(f"Duplicate output name {output!r}.")
            if operation not in ALLOWED_OPERATIONS:
                raise PlanValidationError(
                    f"Operation {operation!r} is not allowed; use one of {list(ALLOWED_OPERATIONS)}.")
            if column is None and operation == "count":
                if entry.key_column is None:
                    raise PlanValidationError("The start table has no key column; give a column for count.")
                column = f"{entry.table}.{entry.key_column}"
            found, bare = resolve(column, "metric column")
            if operation in {"sum", "avg"} and found.columns[bare] not in NUMERIC_TYPES:
                raise PlanValidationError(f"{operation} needs a numeric column but {column!r} is {found.columns[bare]}.")
            outputs.add(output)
            metrics.append((output, operation, column))
            if column not in used:
                used.append(column)
        try:
            names = join_output_names(used)
        except ValueError as exc:
            raise PlanValidationError(str(exc)) from exc
        clash = outputs & set(names)
        if clash:
            raise PlanValidationError(f"Metric output name(s) {sorted(clash)} clash with column names; rename them.")
        filters, conditions = self._join_conditions(resolve, data)
        fmt = data.get("format", "json")
        if fmt not in REPORT_FORMATS:
            raise PlanValidationError(f"format must be one of {list(REPORT_FORMATS)}.")
        title = data.get("title")
        title = title.strip()[:200] if isinstance(title, str) and title.strip() else (
            f"Report: {' + '.join(e.table for e in scope)}")
        types = {f"{e.table}.{c}": t for e in scope for c, t in e.columns.items()}
        metric_columns = {m[2] for m in metrics}
        try:
            return TableReportRequest(
                table=entry.table, columns={n: types[n] for n in used}, group_by=tuple(group_by),
                metrics=tuple(metrics), title=title, report_format=fmt,
                row_limit=min(e.row_limit for e in scope),
                numeric_columns=tuple(n for n in used if types[n] in NUMERIC_TYPES and n in metric_columns),
                categorical_columns=tuple(n for n in group_by if types[n] == "string"),
                filters=filters, conditions=conditions, joins=tuple(joins))
        except ValueError as exc:
            raise PlanValidationError(str(exc)) from exc

    def _validate_join_list(self, entry, data):
        """A listing over the start table plus joined tables; every column is written table.column."""
        scope, joins = self._join_scope(entry, data.get("joins"))
        resolve = self._resolver(scope)

        raw_columns = data.get("columns")
        if (not isinstance(raw_columns, list) or not raw_columns or len(raw_columns) > MAX_LIST_COLUMNS
                or len(set(map(str, raw_columns))) != len(raw_columns)):
            raise PlanValidationError(
                f"columns must be 1-{MAX_LIST_COLUMNS} distinct table.column names "
                '("all" is not allowed with joins).')
        for name in raw_columns:
            resolve(name, "column")
        try:
            join_output_names(raw_columns)
        except ValueError as exc:
            raise PlanValidationError(str(exc)) from exc

        raw_order = data.get("order_by") or []
        if not isinstance(raw_order, list) or len(raw_order) > MAX_ORDER_BY:
            raise PlanValidationError(f"order_by must be a list of at most {MAX_ORDER_BY} items.")
        order: list[tuple[str, str]] = []
        for item in raw_order:
            if not isinstance(item, Mapping):
                raise PlanValidationError('Each order_by item must be {"column": ..., "direction": "asc"|"desc"}.')
            resolve(item.get("column"), "order_by column")
            direction = str(item.get("direction", "asc")).lower()
            if direction not in {"asc", "desc"}:
                raise PlanValidationError("order_by direction must be asc or desc.")
            if any(item["column"] == c for c, _ in order):
                raise PlanValidationError(f"order_by lists {item['column']!r} twice.")
            order.append((item["column"], direction))

        maximum = min(e.row_limit for e in scope) - 1
        limit = data.get("limit", min(DEFAULT_LIST_LIMIT, maximum))
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
            raise PlanValidationError(f"limit must be a whole number between 1 and {maximum}.")

        filters, conditions = self._join_conditions(resolve, data)

        fmt = data.get("format", "json")
        if fmt not in REPORT_FORMATS:
            raise PlanValidationError(f"format must be one of {list(REPORT_FORMATS)}.")
        title = data.get("title")
        title = title.strip()[:200] if isinstance(title, str) and title.strip() else (
            f"Rows: {' + '.join(e.table for e in scope)}")
        used = set(raw_columns) | {c for c, _ in order}
        types = {f"{e.table}.{c}": t for e in scope for c, t in e.columns.items()}
        try:
            return TableReportRequest(
                table=entry.table, columns={n: t for n, t in types.items() if n in used},
                group_by=(), metrics=(), title=title, report_format=fmt,
                row_limit=min(e.row_limit for e in scope), filters=filters, conditions=conditions,
                kind="list", select_columns=tuple(raw_columns), order_by=tuple(order), limit=limit,
                joins=tuple(joins))
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
        if data.get("joins"):
            if kind == "list":
                return self._validate_join_list(entry, data)
            if kind != "aggregate":
                raise PlanValidationError(f"kind must be 'aggregate' or 'list', got {kind!r}.")
            return self._validate_join_aggregate(entry, data)
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
        conditions = self._validate_conditions(entry, data.get("conditions"))

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
                filters=filters, conditions=conditions,
            )
        except ValueError as exc:
            raise PlanValidationError(str(exc)) from exc


__all__ = ["TableRelation", "TableCatalogEntry", "TablePlanResult", "TableReportPlanner", "PlanValidationError",
           "ALLOWED_OPERATIONS", "REPORT_FORMATS"]
