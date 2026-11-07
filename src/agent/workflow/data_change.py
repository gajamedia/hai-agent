from __future__ import annotations

"""Changing data (INSERT / UPDATE / DELETE) without letting an LLM write SQL.

The LLM only picks a *structured* change from the catalog the user approved for writing:
an operation, a table, the columns to set and the conditions that select the rows. Our own
code validates it, builds parameterized SQL, shows the user what would happen (how many rows,
a sample before and after), and only then, after the user has confirmed, runs it inside one
transaction that is rolled back unless it touches exactly the rows that were shown.

What this module never does: take SQL text from anyone, touch a table or column that was not
approved for writing, run an UPDATE or DELETE without conditions, or change more rows than the
user's own limit.
"""

import json
import math
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent.llm.business_database_adapter import (
    BusinessDatabaseAdapter, ReadOnlyTableSpec,
)
from agent.llm.contracts import LLMMessage
from agent.workflow.table_report_planner import (
    CONDITION_OPERATORS, MAX_CONDITIONS, PlanValidationError, TableCatalogEntry, _extract_json,
    condition_scalar, validate_conditions,
)

OPERATIONS = ("insert", "update", "delete")
MAX_INSERT_ROWS = 100
PREVIEW_ROWS = 10
DEFAULT_MAX_ROWS_CEILING = 10000


class ChangeRefused(ValueError):
    """The change is not allowed or no longer matches what the user was shown."""


@dataclass(frozen=True)
class WritableTable:
    """What the user approved for writing on one table."""
    entry: TableCatalogEntry              # approved columns (with types) and filterable columns
    operations: tuple[str, ...]           # subset of OPERATIONS
    writable_columns: tuple[str, ...]     # columns that may be set; never the key column

    @property
    def table(self) -> str:
        return self.entry.table


@dataclass(frozen=True)
class ChangeRequest:
    operation: str
    table: str
    set_values: tuple[tuple[str, Any], ...] = ()               # update
    rows: tuple[tuple[tuple[str, Any], ...], ...] = ()         # insert
    conditions: tuple[tuple[str, str, Any], ...] = ()          # update / delete
    max_rows: int = 100                                        # the USER's limit, not the LLM's
    title: str = ""


@dataclass(frozen=True)
class ChangePlanResult:
    question: str
    request: ChangeRequest | None
    reason: str | None
    provider: str | None
    attempts: int = 1


@dataclass(frozen=True)
class ChangePreview:
    expected_rows: int
    before: tuple[dict[str, Any], ...]
    after: tuple[dict[str, Any], ...]
    sample_note: str = ""


@dataclass(frozen=True)
class ChangeResult:
    rows_affected: int
    before_image: tuple[dict[str, Any], ...] = field(default=(), repr=False)


# --------------------------------------------------------------------------- SQL
class ChangeSqlBuilder:
    """Parameterized statements. Identifiers are checked and quoted; values are always bound."""

    def __init__(self, *, quote: str, paramstyle: str) -> None:
        if paramstyle not in {"qmark", "format"}:
            raise ValueError("paramstyle must be 'qmark' or 'format'")
        self.quote, self.paramstyle = quote, paramstyle

    def _adapter(self, table: str, column: str) -> BusinessDatabaseAdapter:
        spec = ReadOnlyTableSpec(table=table, columns=(column,), filterable_columns=(column,))
        return BusinessDatabaseAdapter(
            connection_factory=lambda: None, table_specs={table: spec},
            paramstyle=self.paramstyle, identifier_quote=self.quote)

    def _statement(self, table: str, conditions: Sequence[tuple[str, str, Any]]):
        params: list[Any] = []
        adapter = self._adapter(table, conditions[0][0] if conditions else "c")

        def bind(value: Any) -> str:
            params.append(value)
            return "?" if self.paramstyle == "qmark" else "%s"

        return adapter, params, bind

    def _where(self, adapter, bind, conditions) -> str:
        if not conditions:
            return ""
        parts = [adapter._condition_sql(
            {"column": c, "operator": op,
             **({} if v is None else {"value": list(v) if isinstance(v, tuple) else v})}, bind)
            for c, op, v in conditions]
        return " WHERE " + " AND ".join(parts)

    def select(self, table: str, columns: Sequence[str], conditions, *, limit: int,
               lock: str = "") -> tuple[str, list[Any]]:
        adapter, params, bind = self._statement(table, conditions)
        q = adapter._q
        sql = (f"SELECT {', '.join(q(c) for c in columns)} FROM {q(table)}"
               f"{self._where(adapter, bind, conditions)} LIMIT {int(limit)}{lock}")
        return sql, params

    def update(self, table: str, set_values, conditions) -> tuple[str, list[Any]]:
        adapter, params, bind = self._statement(table, conditions)
        q = adapter._q
        assignments = ", ".join(f"{q(c)} = {bind(v)}" for c, v in set_values)
        return (f"UPDATE {q(table)} SET {assignments}{self._where(adapter, bind, conditions)}",
                params)

    def delete(self, table: str, conditions) -> tuple[str, list[Any]]:
        adapter, params, bind = self._statement(table, conditions)
        q = adapter._q
        return f"DELETE FROM {q(table)}{self._where(adapter, bind, conditions)}", params

    def insert(self, table: str, columns: Sequence[str], rows) -> tuple[str, list[Any]]:
        adapter, params, bind = self._statement(table, ())
        q = adapter._q
        groups = ", ".join("(" + ", ".join(bind(row.get(c)) for c in columns) + ")" for row in rows)
        return f"INSERT INTO {q(table)} ({', '.join(q(c) for c in columns)}) VALUES {groups}", params


# ------------------------------------------------------------------------ planning
def _plain(value: Any) -> Any:
    """JSON friendly: dates, decimals and bytes from a database row."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "as_tuple"):  # Decimal
        return float(value)
    return value


class DataChangePlanner:
    def __init__(self, llm: Any, tables: Sequence[WritableTable], *, max_attempts: int = 2,
                 today: Callable[[], date] = date.today) -> None:
        if not tables:
            raise ValueError("at least one writable table is required")
        self.llm, self.max_attempts, self._today = llm, max_attempts, today
        self.tables = {t.table: t for t in tables}

    def _catalog(self) -> list[dict[str, Any]]:
        return [{"table": t.table, "columns": dict(t.entry.columns),
                 "operations": list(t.operations), "writable_columns": list(t.writable_columns),
                 "filterable_columns": list(t.entry.filterable_columns),
                 "key_column": t.entry.key_column} for t in self.tables.values()]

    def _system_prompt(self) -> str:
        return (
            "You plan ONE data change on ONE table. Choose ONLY from this catalog; never invent tables "
            "or columns. You never write SQL.\n"
            f"CATALOG:\n{json.dumps(self._catalog(), ensure_ascii=False)}\n"
            f"Today is {self._today().isoformat()}.\n"
            "Reply with ONLY one JSON object.\n"
            'INSERT: {"supported": true, "operation": "insert", "table": "...", "rows": [{"col": value}], '
            '"title": "..."} (at most 100 rows).\n'
            'UPDATE: {"supported": true, "operation": "update", "table": "...", "set": {"col": value}, '
            '"conditions": [{"column": "...", "operator": "eq", "value": ...}], "title": "..."}.\n'
            'DELETE: {"supported": true, "operation": "delete", "table": "...", '
            '"conditions": [{"column": "...", "operator": "eq", "value": ...}], "title": "..."}.\n'
            "Rules: only operations listed for the table; set/rows only use writable_columns (never the "
            "key column); conditions only use filterable_columns, with operators "
            f"{sorted(CONDITION_OPERATORS)} (between needs [low, high], in needs a list, is_null/is_not_null "
            f"take no value), at most {MAX_CONDITIONS}, all ANDed. UPDATE and DELETE MUST have at least one "
            "condition; never change a whole table. Numbers are plain numbers (\"30 ribu\" is 30000); dates "
            "are ISO yyyy-mm-dd. Do not decide how many rows may change: the user sets that.\n"
            'If the request cannot be done with this catalog: {"supported": false, "reason": "..."}.'
        )

    def plan(self, question: str, *, max_rows: int) -> ChangePlanResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        question = question.strip()
        messages = [LLMMessage("system", self._system_prompt()),
                    LLMMessage("user", f"Request:\n{question}")]
        errors: list[str] = []
        provider = None
        for attempt in range(1, self.max_attempts + 1):
            response = self.llm.complete(messages, temperature=0)
            provider = response.provider
            try:
                data = _extract_json(response.content)
                if data.get("supported") is False:
                    reason = str(data.get("reason") or "The request cannot be done.").strip()
                    return ChangePlanResult(question, None, reason, provider, attempt)
                return ChangePlanResult(question, self.validate(data, max_rows=max_rows),
                                        None, provider, attempt)
            except PlanValidationError as exc:
                errors.append(str(exc))
                messages += [LLMMessage("assistant", response.content),
                             LLMMessage("user", f"Invalid: {exc} Reply again with ONLY the corrected JSON object.")]
        return ChangePlanResult(question, None, f"Could not produce a valid plan: {errors[-1]}",
                                provider, self.max_attempts)

    # ---------------------------------------------------------------- validation
    def _values(self, table: WritableTable, raw: Any, *, what: str) -> tuple[tuple[str, Any], ...]:
        if not isinstance(raw, Mapping) or not raw:
            raise PlanValidationError(f"{what} must be a non-empty object of column: value.")
        out = []
        for column, value in raw.items():
            if column == table.entry.key_column:
                raise PlanValidationError(f"The key column {column!r} cannot be set.")
            if column not in table.writable_columns:
                raise PlanValidationError(
                    f"Column {column!r} is not writable; writable columns: {list(table.writable_columns)}.")
            if value is not None:
                condition_scalar(table.entry.columns[column], column, value)
            out.append((column, value))
        return tuple(out)

    def validate(self, data: Mapping[str, Any], *, max_rows: int) -> ChangeRequest:
        operation, name = data.get("operation"), data.get("table")
        if operation not in OPERATIONS:
            raise PlanValidationError(f"operation must be one of {list(OPERATIONS)}.")
        table = self.tables.get(name) if isinstance(name, str) else None
        if table is None:
            raise PlanValidationError(f"Unknown table {name!r}; choose one of {sorted(self.tables)}.")
        if operation not in table.operations:
            raise PlanValidationError(
                f"{operation!r} is not allowed on {name!r}; allowed: {list(table.operations)}.")
        title = data.get("title")
        title = title.strip()[:200] if isinstance(title, str) and title.strip() else f"{operation} {name}"
        if operation == "insert":
            raw_rows = data.get("rows")
            if (not isinstance(raw_rows, list) or not raw_rows
                    or len(raw_rows) > min(MAX_INSERT_ROWS, max_rows)):
                raise PlanValidationError(
                    f"rows must be a list of 1 to {min(MAX_INSERT_ROWS, max_rows)} objects "
                    "(the user's row limit applies).")
            rows = tuple(self._values(table, r, what="each row") for r in raw_rows)
            if len({tuple(sorted(c for c, _ in row)) for row in rows}) != 1:
                raise PlanValidationError(
                    "every inserted row must set the same columns (a missing one would become NULL).")
            if any(data.get(k) for k in ("conditions", "set")):
                raise PlanValidationError("insert takes only rows.")
            return ChangeRequest("insert", name, rows=rows, max_rows=max_rows, title=title)
        conditions = validate_conditions(table.entry, data.get("conditions"))
        if not conditions:
            raise PlanValidationError(
                f"{operation} needs at least one condition; changing a whole table is not allowed.")
        if data.get("rows"):
            raise PlanValidationError(f"{operation} does not take rows.")
        if operation == "update":
            return ChangeRequest("update", name, set_values=self._values(table, data.get("set"), what="set"),
                                 conditions=conditions, max_rows=max_rows, title=title)
        if data.get("set"):
            raise PlanValidationError("delete does not take set.")
        return ChangeRequest("delete", name, conditions=conditions, max_rows=max_rows, title=title)


# ---------------------------------------------------------------------- execution
class DataChangeService:
    """Preview and execute validated changes with a write-capable connection."""

    def __init__(self, *, planner: DataChangePlanner, connect: Callable[[], Any], quote: str,
                 paramstyle: str, audit_path: Path, lock_clause: str = " FOR UPDATE",
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.planner, self._connect = planner, connect
        self.builder = ChangeSqlBuilder(quote=quote, paramstyle=paramstyle)
        self.audit_path, self.lock_clause, self._clock = Path(audit_path), lock_clause, clock
        self._audit_lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def table(self, name: str) -> WritableTable:
        return self.planner.tables[name]

    def _columns(self, request: ChangeRequest) -> list[str]:
        table = self.table(request.table)
        return list(table.entry.columns)

    @staticmethod
    def _rows(cursor, columns: Sequence[str]) -> list[dict[str, Any]]:
        return [{c: _plain(v) for c, v in zip(columns, row)} for row in cursor.fetchall()]

    def summarize(self, request: ChangeRequest) -> dict[str, Any]:
        out: dict[str, Any] = {"operation": request.operation, "table": request.table,
                               "title": request.title, "max_rows": request.max_rows}
        if request.operation == "insert":
            out["rows"] = [dict(r) for r in request.rows]
        else:
            out["conditions"] = [{"column": c, "operator": op,
                                  "value": list(v) if isinstance(v, (tuple, list)) else v}
                                 for c, op, v in request.conditions]
        if request.operation == "update":
            out["set"] = dict(request.set_values)
        return out

    def confirm_phrase(self, request: ChangeRequest, expected_rows: int) -> str:
        verb = {"insert": "TAMBAH", "update": "UBAH", "delete": "HAPUS"}[request.operation]
        return f"{verb} {expected_rows} BARIS {request.table}"

    # ------------------------------------------------------------------ preview
    def preview(self, request: ChangeRequest) -> ChangePreview:
        """What would change, without changing anything. Raises ChangeRefused if it is over the limit."""
        if request.operation == "insert":
            after = tuple({k: _plain(v) for k, v in row} for row in request.rows[:PREVIEW_ROWS])
            return ChangePreview(len(request.rows), (), after,
                                 "" if len(request.rows) <= PREVIEW_ROWS
                                 else f"showing the first {PREVIEW_ROWS} of {len(request.rows)} rows")
        columns = self._columns(request)
        sql, params = self.builder.select(request.table, columns, request.conditions,
                                          limit=request.max_rows + 1)
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, params)
                rows = self._rows(cursor, columns)
            finally:
                cursor.close()
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()
        if len(rows) > request.max_rows:
            raise ChangeRefused(
                f"This would change more than {request.max_rows} rows, the limit you set. "
                "Narrow the conditions or raise the limit.")
        if not rows:
            raise ChangeRefused("No rows match these conditions, so nothing would change.")
        sample = rows[:PREVIEW_ROWS]
        after = tuple({**row, **{c: _plain(v) for c, v in request.set_values}} for row in sample) \
            if request.operation == "update" else ()
        return ChangePreview(len(rows), tuple(sample), after,
                             "" if len(rows) <= PREVIEW_ROWS
                             else f"showing the first {PREVIEW_ROWS} of {len(rows)} rows")

    # ------------------------------------------------------------------ execute
    def execute(self, request: ChangeRequest, *, expected_rows: int, actor: str,
                approver: str | None, plan_id: str, question: str) -> ChangeResult:
        """Run the change in one transaction; roll back unless it touches exactly expected_rows."""
        if expected_rows > request.max_rows or expected_rows < 1:
            raise ChangeRefused("The number of rows does not fit the limit.")
        connection = self._connect()
        record: dict[str, Any] = {
            "plan_id": plan_id, "actor": actor, "approver": approver, "operation": request.operation,
            "table": request.table, "expected_rows": expected_rows, "question": question[:500]}
        try:
            cursor = connection.cursor()
            try:
                before: list[dict[str, Any]] = []
                if request.operation != "insert":
                    columns = self._columns(request)
                    sql, params = self.builder.select(
                        request.table, columns, request.conditions,
                        limit=request.max_rows + 1, lock=self.lock_clause)
                    cursor.execute(sql, params)
                    before = self._rows(cursor, columns)
                    if len(before) != expected_rows:
                        raise ChangeRefused(
                            f"The data changed since the preview: {len(before)} rows match now, "
                            f"{expected_rows} were shown. Nothing was changed; plan it again.")
                if request.operation == "insert":
                    cols = sorted({c for row in request.rows for c, _ in row})
                    sql, params = self.builder.insert(
                        request.table, cols, [dict(row) for row in request.rows])
                elif request.operation == "update":
                    sql, params = self.builder.update(request.table, request.set_values, request.conditions)
                else:
                    sql, params = self.builder.delete(request.table, request.conditions)
                record.update(sql=sql, params=[_plain(p) for p in params])
                cursor.execute(sql, params)
                affected = cursor.rowcount
                if affected != expected_rows:
                    raise ChangeRefused(
                        f"The statement touched {affected} rows, not the {expected_rows} that were shown. "
                        "It was rolled back; nothing changed.")
            finally:
                cursor.close()
            connection.commit()
        except Exception as exc:
            try:
                connection.rollback()
            except Exception:  # the connection may already be gone
                pass
            record.update(status="rolled_back", reason=f"{type(exc).__name__}: {exc}"[:300])
            self._audit(record)
            raise
        finally:
            try:
                connection.close()
            except Exception:
                pass
        record.update(status="committed", rows_affected=affected, before_image=before)
        self._audit(record)
        return ChangeResult(affected, tuple(before))

    def audit(self, event: str, **fields: Any) -> None:
        """Record a decision that is not an execution (approval, rejection, refusal)."""
        self._audit({"event": event, **fields})

    def _audit(self, record: dict[str, Any]) -> None:
        record = {"at": self._clock().isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
        with self._audit_lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


__all__ = [
    "ChangePlanResult", "ChangePreview", "ChangeRefused", "ChangeRequest", "ChangeResult",
    "ChangeSqlBuilder", "DataChangePlanner", "DataChangeService", "DEFAULT_MAX_ROWS_CEILING",
    "MAX_INSERT_ROWS", "OPERATIONS", "WritableTable",
]
