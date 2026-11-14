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

import re
import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

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
MAX_IMPORT_ROWS = 1000   # rows of an uploaded file recorded in one go (typed rows stay at MAX_INSERT_ROWS)
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
    # An update with ``rows`` (from an uploaded file): each row is the match column's value followed by the
    # columns to set on the one row that holds that value.
    match_column: str = ""


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
    # For an insert into a table with a whole-number key: the key and a fingerprint of each row made, so the
    # insert can be taken back. Never the values themselves.
    inserted: tuple[dict[str, Any], ...] = ()


_UNTRACKED = object()


def _fingerprint(row: Mapping[str, Any]) -> str:
    """A short hash of a row as read from the database; equal rows give equal fingerprints."""
    text = json.dumps({k: _plain(v) for k, v in row.items()}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- SQL
class ChangeSqlBuilder:
    """Parameterized statements. Identifiers are checked and quoted; values are always bound."""

    def __init__(self, *, quote: str, paramstyle: str, dialect: str = "limit") -> None:
        if paramstyle not in {"qmark", "format"}:
            raise ValueError("paramstyle must be 'qmark' or 'format'")
        if dialect not in {"limit", "top"}:
            raise ValueError("dialect must be 'limit' or 'top'")
        self.quote, self.paramstyle, self.dialect = quote, paramstyle, dialect

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
        parts = [self._key_list(adapter, bind, c, v) if op == "in" and isinstance(v, (tuple, list)) and len(v) > 20
                 else adapter._condition_sql(
                     {"column": c, "operator": op,
                      **({} if v is None else {"value": list(v) if isinstance(v, tuple) else v})}, bind)
                 for c, op, v in conditions]
        return " WHERE " + " AND ".join(parts)

    @staticmethod
    def _key_list(adapter, bind, column: str, values) -> str:
        """``column IN (...)`` for a long list of keys that our own code produced (the rows an entry made, or the
        keys of an uploaded file). A model's conditions stay capped at 20 values by the planner; this path is
        never reachable from model output. SQL Server takes up to 2,100 parameters, and the lists here stop at
        MAX_IMPORT_ROWS."""
        if len(values) > MAX_IMPORT_ROWS or any(isinstance(v, bool) or not isinstance(v, (int, float, str))
                                                for v in values):
            raise ChangeRefused("Too many keys, or keys of an unsupported kind.")
        return f"{adapter._q(column)} IN ({', '.join(bind(v) for v in values)})"

    def select(self, table: str, columns: Sequence[str], conditions, *, limit: int,
               lock: str = "") -> tuple[str, list[Any]]:
        adapter, params, bind = self._statement(table, conditions)
        q = adapter._q
        names = ', '.join(q(c) for c in columns)
        where = self._where(adapter, bind, conditions)
        if self.dialect == "top":     # SQL Server: TOP after SELECT, the lock as a table hint after the table
            return f"SELECT TOP ({int(limit)}) {names} FROM {q(table)}{lock}{where}", params
        return f"SELECT {names} FROM {q(table)}{where} LIMIT {int(limit)}{lock}", params

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

    def max_key(self, table: str, key: str) -> str:
        adapter, _params, _bind = self._statement(table, ((key, "eq", 0),))
        q = adapter._q
        return f"SELECT MAX({q(key)}) FROM {q(table)}"

    def insert(self, table: str, columns: Sequence[str], rows) -> tuple[str, list[Any]]:
        adapter, params, bind = self._statement(table, ())
        q = adapter._q
        groups = ", ".join("(" + ", ".join(bind(row.get(c)) for c in columns) + ")" for row in rows)
        return f"INSERT INTO {q(table)} ({', '.join(q(c) for c in columns)}) VALUES {groups}", params


    def insert_chunks(self, table: str, columns: Sequence[str], rows, *, max_params: int = 2000,
                      max_rows: int = 1000) -> list[tuple[str, list[Any]]]:
        """One INSERT per chunk. SQL Server accepts at most 2,100 parameters and 1,000 rows per statement;
        every chunk runs in the caller's one transaction."""
        per = max(1, min(max_rows, max_params // max(1, len(columns))))
        rows = list(rows)
        return [self.insert(table, columns, rows[i:i + per]) for i in range(0, len(rows), per)]


# ------------------------------------------------------------------------ planning
_MISSING_COLUMN = (
    re.compile(r"Field '([^']+)' doesn't have a default value"),            # MySQL 1364
    re.compile(r"Column '([^']+)' cannot be null"),                         # MySQL 1048
    re.compile(r"NOT NULL constraint failed: [\w$]+\.(\w+)"),               # SQLite
    re.compile(r"Cannot insert the value NULL into column '([^']+)'"),      # SQL Server
)


def _missing_required_column(exc: BaseException) -> str | None:
    """The column a database named when it refused an insert for lack of a value, else None."""
    text = str(exc)
    for pattern in _MISSING_COLUMN:
        found = pattern.search(text)
        if found:
            return found.group(1)
    return None


def _plain(value: Any) -> Any:
    """JSON friendly: dates, decimals and bytes from a database row."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "as_tuple"):  # Decimal
        return float(value)
    return value


def _same(stored: Any, wanted: Any) -> bool:
    """A value read back equals the one written, allowing for how databases hand values back (1 for True,
    12.0 for 12, a date as text)."""
    stored, wanted = _plain(stored), _plain(wanted)
    if stored is None or wanted is None:
        return stored is wanted
    if isinstance(stored, bool) or isinstance(wanted, bool):
        return bool(stored) == bool(wanted)
    if isinstance(stored, (int, float)) and isinstance(wanted, (int, float)):
        return math.isclose(float(stored), float(wanted), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(stored, str) and isinstance(wanted, str):
        a, b = stored.replace("T", " "), wanted.replace("T", " ")
        return a == b or (len(b) == 10 and a[:10] == b)    # a date written as text may come back with a time
    return stored == wanted


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

    def validate_bulk_update(self, table_name: str, match_column: str, rows: Any, *, max_rows: int,
                             title: str = "") -> ChangeRequest:
        """Update many rows, each found by its own value of ``match_column`` and given its own new values (an
        uploaded file). Built by our code, never by a model. A value left out is left as it is."""
        table = self.tables.get(table_name) if isinstance(table_name, str) else None
        if table is None:
            raise PlanValidationError(f"Unknown table {table_name!r}; choose one of {sorted(self.tables)}.")
        if "update" not in table.operations:
            raise PlanValidationError(f"'update' is not allowed on {table_name!r}; allowed: {list(table.operations)}.")
        if match_column not in table.entry.filterable_columns:
            raise PlanValidationError(
                f"Rows cannot be found by {match_column!r}; use one of {list(table.entry.filterable_columns)}.")
        limit = min(MAX_IMPORT_ROWS, max_rows)
        if not isinstance(rows, list) or not 1 <= len(rows) <= limit:
            raise PlanValidationError(f"rows must be a list of 1 to {limit} objects (the row limit applies).")
        out, seen = [], set()
        for number, raw in enumerate(rows, start=1):
            if not isinstance(raw, Mapping) or raw.get(match_column) is None:
                raise PlanValidationError(f"Row {number} has no value in {match_column!r}.")
            key = raw[match_column]
            condition_scalar(table.entry.columns[match_column], match_column, key)
            if key in seen:
                raise PlanValidationError(f"The value {key!r} of {match_column!r} appears twice in the file.")
            seen.add(key)
            sets = {c: v for c, v in raw.items() if c != match_column}
            if not sets or any(v is None for v in sets.values()):
                raise PlanValidationError(f"Row {number} has nothing to change (empty values are left as they are).")
            out.append(((match_column, key),) + self._values(table, sets, what="each row"))
        return ChangeRequest("update", table_name, rows=tuple(out), max_rows=max_rows, match_column=match_column,
                             title=title or f"update {table_name}")

    def validate(self, data: Mapping[str, Any], *, max_rows: int, row_limit: int = MAX_INSERT_ROWS) -> ChangeRequest:
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
                    or len(raw_rows) > min(row_limit, max_rows)):
                raise PlanValidationError(
                    f"rows must be a list of 1 to {min(row_limit, max_rows)} objects "
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
                 paramstyle: str, audit_path: Path, lock_clause: str = " FOR UPDATE", dialect: str = "limit",
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 audit_detail: Union[str, Callable[[], str]] = "minimal") -> None:
        # ``audit_detail``: "minimal" (default) keeps no business data in the audit file - only the key of
        # each changed row and the number of values; "full" also keeps the data as it was before the change
        # and the values written. A callable is asked at every write, so a change of setting needs no restart.
        self._audit_detail = audit_detail
        self.planner, self._connect = planner, connect
        self.builder = ChangeSqlBuilder(quote=quote, paramstyle=paramstyle, dialect=dialect)
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
        if request.operation == "insert" or request.rows:
            out["rows"] = [dict(r) for r in request.rows]
            if request.match_column:
                out["match_column"] = request.match_column
        else:
            out["conditions"] = [{"column": c, "operator": op,
                                  "value": list(v) if isinstance(v, (tuple, list)) else v}
                                 for c, op, v in request.conditions]
        if request.operation == "update" and not request.rows:
            out["set"] = dict(request.set_values)
        return out

    def confirm_phrase(self, request: ChangeRequest, expected_rows: int) -> str:
        verb = {"insert": "TAMBAH", "update": "UBAH", "delete": "HAPUS"}[request.operation]
        return f"{verb} {expected_rows} BARIS {request.table}"

    @staticmethod
    def _bulk_keys(request: ChangeRequest) -> tuple:
        """The match values of an update that comes from a file, else ()."""
        if request.operation != "update" or not request.rows:
            return ()
        return tuple(dict(row)[request.match_column] for row in request.rows)

    @staticmethod
    def _check_bulk_match(request: ChangeRequest, rows: Sequence[Mapping[str, Any]]) -> None:
        """Every value in the file must find exactly one row: none missing, none found twice."""
        key = request.match_column
        wanted = {dict(r)[key] for r in request.rows}
        found: dict[Any, int] = {}
        for row in rows:
            found[row[key]] = found.get(row[key], 0) + 1
        missing = [k for k in wanted if k not in found]
        twice = [k for k, n in found.items() if n > 1]
        if missing or twice:
            parts = []
            if missing:
                parts.append(f"{len(missing)} of the {len(wanted)} values of {key} match no row "
                             f"(first: {missing[0]!r})")
            if twice:
                parts.append(f"{len(twice)} values of {key} match more than one row (first: {twice[0]!r})")
            raise ChangeRefused("; ".join(parts).capitalize() + ". Nothing would be changed. "
                                "Rows can only be updated by a column whose values are unique.")

    # ------------------------------------------------------------------ preview
    def preview(self, request: ChangeRequest) -> ChangePreview:
        """What would change, without changing anything. Raises ChangeRefused if it is over the limit."""
        if request.operation == "insert":
            after = tuple({k: _plain(v) for k, v in row} for row in request.rows[:PREVIEW_ROWS])
            return ChangePreview(len(request.rows), (), after,
                                 "" if len(request.rows) <= PREVIEW_ROWS
                                 else f"showing the first {PREVIEW_ROWS} of {len(request.rows)} rows")
        columns = self._columns(request)
        bulk = self._bulk_keys(request)
        sql, params = self.builder.select(request.table, columns,
                                          ((request.match_column, "in", bulk),) if bulk else request.conditions,
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
        if bulk:
            self._check_bulk_match(request, rows)
            wanted = {dict(r)[request.match_column]: dict(r) for r in request.rows}
            sample = rows[:PREVIEW_ROWS]
            return ChangePreview(len(rows), tuple(sample),
                                 tuple({**row, **wanted[row[request.match_column]]} for row in sample),
                                 "" if len(rows) <= PREVIEW_ROWS
                                 else f"showing the first {PREVIEW_ROWS} of {len(rows)} rows")
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
        inserted: tuple[dict[str, Any], ...] = ()
        try:
            cursor = connection.cursor()
            try:
                before: list[dict[str, Any]] = []
                bulk = self._bulk_keys(request)
                if request.operation != "insert":
                    columns = self._columns(request)
                    sql, params = self.builder.select(
                        request.table, columns,
                        ((request.match_column, "in", bulk),) if bulk else request.conditions,
                        limit=request.max_rows + 1, lock=self.lock_clause)
                    cursor.execute(sql, params)
                    before = self._rows(cursor, columns)
                    if len(before) != expected_rows:
                        raise ChangeRefused(
                            f"The data changed since the preview: {len(before)} rows match now, "
                            f"{expected_rows} were shown. Nothing was changed; plan it again.")
                    if bulk:
                        self._check_bulk_match(request, before)
                if request.operation == "insert":
                    cols = sorted({c for row in request.rows for c, _ in row})
                    rows = [dict(row) for row in request.rows]
                    statements = (self.builder.insert_chunks(request.table, cols, rows)
                                  if self.builder.dialect == "top"
                                  else [self.builder.insert(request.table, cols, rows)])
                elif bulk:
                    statements = [self.builder.update(request.table, row[1:], ((request.match_column, "eq", row[0][1]),))
                                  for row in request.rows]
                elif request.operation == "update":
                    statements = [self.builder.update(request.table, request.set_values, request.conditions)]
                else:
                    statements = [self.builder.delete(request.table, request.conditions)]
                sql = statements[0][0]
                record.update(sql=sql, params=[_plain(p) for _s, ps in statements for p in ps],
                              statements=len(statements))
                record["params_count"] = len(record["params"])
                affected = 0
                # For an insert into a table with a numeric key, remember which rows this change made, so that
                # it can be taken back later (and only those rows). Read inside the same transaction.
                track_key = self.table(request.table).entry.key_column if request.operation == "insert" else None
                max_before = self._max_key(cursor, request.table, track_key) if track_key else _UNTRACKED
                for sql, params in statements:
                    cursor.execute(sql, params)
                    if bulk and cursor.rowcount != 1:   # each key names exactly one row, which is locked
                        raise ChangeRefused(f"One update touched {cursor.rowcount} rows, not 1. "
                                            "It was rolled back; nothing changed.")
                    affected += cursor.rowcount
                if affected != expected_rows:
                    raise ChangeRefused(
                        f"The statement touched {affected} rows, not the {expected_rows} that were shown. "
                        "It was rolled back; nothing changed.")
                if track_key and max_before is not _UNTRACKED:
                    inserted = self._inserted_rows(cursor, request, track_key, max_before, affected)
            finally:
                cursor.close()
            connection.commit()
        except Exception as exc:
            try:
                connection.rollback()
            except Exception:  # the connection may already be gone
                pass
            record.update(status="rolled_back", reason=f"{type(exc).__name__}: {exc}"[:300])
            if self._detail_mode() != "full":
                record["params"] = []
            self._audit(record)
            missing = _missing_required_column(exc)
            if missing and request.operation == "insert":
                raise ChangeRefused(
                    f"Kolom {missing} wajib diisi di tabel {request.table}, tetapi belum ada nilainya. "
                    "Tidak ada yang tersimpan. Isi kolom itu (di file, atau tulis nilainya di kolom teks, "
                    f"misalnya \"{missing} diisi ...\"), lalu siapkan lagi.") from exc
            raise
        finally:
            try:
                connection.close()
            except Exception:
                pass
        record.update(status="committed", rows_affected=affected)
        record.update(self._data_for_audit(request, record.get("params", []), before))
        if inserted:
            record["inserted_keys"] = [row["key"] for row in inserted]      # keys only, never the values
        self._audit(record)
        return ChangeResult(affected, tuple(before), inserted)

    # ------------------------------------------------------------------ taking an insert back
    def _max_key(self, cursor, table: str, key: str):
        """The highest key now, or None for an empty table, or ``_UNTRACKED`` when the key is not a plain
        whole number (such a table can still be written, but its inserts cannot be taken back)."""
        cursor.execute(self.builder.max_key(table, key))
        row = cursor.fetchone()
        value = row[0] if row else None
        if value is None:
            return None
        return value if isinstance(value, int) and not isinstance(value, bool) else _UNTRACKED

    def _inserted_rows(self, cursor, request: ChangeRequest, key: str, max_before, expected: int):
        """The rows this insert made: those above the highest key seen before it, inside the same transaction.

        If their number is not exactly the number inserted (someone else inserted at the same moment) the whole
        change is refused and rolled back: it is better to try again than to guess which rows are ours."""
        columns = self._table_columns(request.table)
        conditions = () if max_before is None else ((key, "gt", max_before),)
        sql, params = self.builder.select(request.table, columns, conditions, limit=expected + 1)
        cursor.execute(sql, params)
        rows = self._rows(cursor, columns)
        if len(rows) != expected:
            raise ChangeRefused(
                "Another change was recorded at the same moment, so this one was not saved. "
                "Nothing changed; press Simpan again.")
        return tuple({"key": row[key], "fp": _fingerprint(row)} for row in sorted(rows, key=lambda r: r[key]))

    def _table_columns(self, table: str) -> list[str]:
        return list(self.table(table).entry.columns)

    def undo_insert(self, table: str, rows: Sequence[Mapping[str, Any]], *, actor: str, plan_id: str,
                    entry_id: str) -> int:
        """Remove exactly the rows an earlier insert made, if and only if every one of them is still there and
        still holds what was recorded. Whatever the table's approvals say, nothing else can be removed this way:
        the rows are named by their keys, and one that was edited since is a refusal, never a deletion."""
        key = self.table(table).entry.key_column
        keys = [row["key"] for row in rows]
        wanted = {row["key"]: row["fp"] for row in rows}
        if not key or not keys or len(keys) != len(set(keys)) or len(keys) > MAX_IMPORT_ROWS:
            raise ChangeRefused("This entry cannot be taken back.")
        columns = self._table_columns(table)
        conditions = ((key, "in", tuple(keys)),)
        record: dict[str, Any] = {"event": "undo", "plan_id": plan_id, "entry_id": entry_id, "actor": actor,
                                  "table": table, "keys": keys}
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                sql, params = self.builder.select(table, columns, conditions, limit=len(keys) + 1,
                                                  lock=self.lock_clause)
                cursor.execute(sql, params)
                found = self._rows(cursor, columns)
                if len(found) != len(keys):
                    raise ChangeRefused(
                        f"{len(keys) - len(found)} of the {len(keys)} rows are no longer there, so nothing was "
                        "taken back.")
                changed = [row[key] for row in found if _fingerprint(row) != wanted.get(row[key])]
                if changed:
                    raise ChangeRefused(
                        f"{len(changed)} of the rows were changed after they were recorded, so nothing was "
                        "taken back. Ask someone who may change data to correct them.")
                sql, params = self.builder.delete(table, conditions)
                cursor.execute(sql, params)
                if cursor.rowcount != len(keys):
                    raise ChangeRefused(f"The statement touched {cursor.rowcount} rows, not {len(keys)}. "
                                        "It was rolled back; nothing changed.")
            finally:
                cursor.close()
            connection.commit()
        except Exception as exc:
            try:
                connection.rollback()
            except Exception:
                pass
            record.update(status="rolled_back", reason=f"{type(exc).__name__}: {exc}"[:300])
            self._audit(record)
            raise
        finally:
            try:
                connection.close()
            except Exception:
                pass
        record.update(status="committed", rows_affected=len(keys))
        self._audit(record)
        return len(keys)

    def verify(self, request: ChangeRequest, expected_rows: int) -> dict[str, Any]:
        """After the commit: read the table again and say what is observed.

        * delete: no row matches the conditions any more;
        * update: when the conditions do not mention a column that was set, exactly ``expected_rows``
          rows match the conditions *and* carry the new values;
        * insert, and updates whose conditions the update itself changed: the rows cannot be found
          again, so the evidence is the row count the database reported inside the committed
          transaction (``method`` says which).

        Returns ``{"method", "expected_rows", "observed_rows", "confirmed"}``.
        """
        if self._bulk_keys(request):
            return self._verify_bulk(request, expected_rows)
        set_columns = {c for c, _ in request.set_values}
        condition_columns = {c for c, _, _ in request.conditions}
        if request.operation == "insert" or (request.operation == "update" and set_columns & condition_columns):
            return {"method": "database_reported_rowcount", "expected_rows": expected_rows,
                    "observed_rows": expected_rows, "confirmed": True}
        conditions = tuple(request.conditions)
        if request.operation == "update":
            conditions += tuple((c, "is_null", None) if v is None else (c, "eq", v)
                                for c, v in request.set_values)
        columns = self._columns(request)[:1]
        sql, params = self.builder.select(request.table, columns, conditions, limit=expected_rows + 1)
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, params)
                observed = len(cursor.fetchall())
            finally:
                cursor.close()
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()
        wanted = 0 if request.operation == "delete" else expected_rows
        return {"method": "reread_after_commit", "expected_rows": wanted, "observed_rows": observed,
                "confirmed": observed == wanted}

    def _verify_bulk(self, request: ChangeRequest, expected_rows: int) -> dict[str, Any]:
        """Read the rows again by their keys and compare every value that was meant to be written."""
        columns = self._columns(request)
        keys = self._bulk_keys(request)
        sql, params = self.builder.select(request.table, columns, ((request.match_column, "in", keys),),
                                          limit=len(keys) + 1)
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
        seen = {row[request.match_column]: row for row in rows}
        good = 0
        for pairs in request.rows:
            wanted = dict(pairs)
            row = seen.get(wanted[request.match_column])
            if row is not None and all(_same(row.get(c), v) for c, v in wanted.items()):
                good += 1
        return {"method": "reread_after_commit", "expected_rows": expected_rows, "observed_rows": good,
                "confirmed": good == expected_rows}

    def _detail_mode(self) -> str:
        mode = self._audit_detail() if callable(self._audit_detail) else self._audit_detail
        return "full" if mode == "full" else "minimal"

    def _data_for_audit(self, request: ChangeRequest, params: list[Any],
                        before: list[dict[str, Any]]) -> dict[str, Any]:
        """The business data an audit record may carry, according to the audit detail setting."""
        if self._detail_mode() == "full":
            return {"before_image": before}
        key = self.table(request.table).entry.key_column
        keys = [{key: row.get(key)} for row in before] if key else []
        # The values written (record["params"]) are withheld too: only how many there were is kept.
        return {"before_image": keys, "before_image_rows": len(before), "params": [], "audit_detail": "minimal"}

    def audit(self, event: str, **fields: Any) -> None:
        """Record a decision that is not an execution (approval, rejection, refusal)."""
        plan = fields.get("plan")
        if self._detail_mode() != "full" and isinstance(plan, Mapping):
            fields["plan"] = self._plan_without_values(plan)
        self._audit({"event": event, **fields})

    @staticmethod
    def _plan_without_values(plan: Mapping[str, Any]) -> dict[str, Any]:
        """A plan as the audit may keep it when it holds no business data: what kind, where, how many, and which
        columns - never the values that would be written or searched for (nor the model's title, which may echo them)."""
        out: dict[str, Any] = {"operation": plan.get("operation"), "table": plan.get("table"),
                               "max_rows": plan.get("max_rows")}
        if isinstance(plan.get("rows"), list):
            out["rows_count"] = len(plan["rows"])
        if plan.get("match_column"):
            out["match_column"] = plan["match_column"]
        if isinstance(plan.get("set"), Mapping):
            out["set_columns"] = sorted(plan["set"])
        if isinstance(plan.get("conditions"), list):
            out["condition_columns"] = [c.get("column") for c in plan["conditions"] if isinstance(c, Mapping)]
        return out

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
    "MAX_IMPORT_ROWS", "MAX_INSERT_ROWS", "OPERATIONS", "WritableTable",
]
