from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Callable, Mapping

from .capability_execution_contracts import (
    CapabilityAdapterError,
    CapabilityRequestValidationError,
)


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# How each SQL family limits rows: LIMIT n (SQLite/PostgreSQL/MySQL), SELECT TOP (n)
# (SQL Server) and FETCH FIRST n ROWS ONLY (Oracle 12c+).
_DIALECTS = frozenset({"limit", "top", "fetch_first"})
_ALLOWED_ARGUMENTS = frozenset({"table", "columns", "filters", "conditions", "order_by", "limit"})

# Structured comparisons. The caller never supplies SQL: only a column (which must be allowlisted
# as filterable), an operator from this closed set, and a value that is always sent as a bound
# parameter.
CONDITION_OPERATORS = frozenset({
    "eq", "ne", "gt", "gte", "lt", "lte", "between", "in",
    "contains", "starts_with", "ends_with", "is_null", "is_not_null",
})
MAX_CONDITIONS = 5
MAX_IN_VALUES = 20
MAX_CONDITION_TEXT = 200
_COMPARISON_SQL = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
# "!" is the LIKE escape character because it means nothing special in any dialect's string
# literals, unlike a backslash (an escape in MySQL, a plain character elsewhere).
_LIKE_ESCAPE = "!"


def _like_pattern(text: str, *, before: bool, after: bool) -> str:
    escaped = (text.replace("!", "!!").replace("%", "!%").replace("_", "!_").replace("[", "!["))
    return f"{'%' if before else ''}{escaped}{'%' if after else ''}"


def _condition_scalar(value: Any, column: str) -> Any:
    if value is None or isinstance(value, (dict, list, tuple, set, bytes)):
        raise CapabilityRequestValidationError(f"Condition value for '{column}' must be a single value.")
    if isinstance(value, float) and not math.isfinite(value):
        raise CapabilityRequestValidationError(f"Condition value for '{column}' must be finite.")
    if isinstance(value, str) and (not value.strip() or len(value) > MAX_CONDITION_TEXT):
        raise CapabilityRequestValidationError(f"Condition text for '{column}' is empty or too long.")
    if not isinstance(value, (str, int, float, bool)):
        raise CapabilityRequestValidationError(f"Condition value for '{column}' has an unsupported type.")
    return value


def _json_safe(value: Any) -> Any:
    """Normalize driver-specific values so results are JSON-serializable.

    SQLite only returns JSON-native types, but other drivers return Decimal,
    date/datetime, UUID and bytes. Decimal becomes float (fine for reporting
    aggregates; exact money arithmetic is out of scope), dates become ISO text
    and binary data is never passed through.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    return str(value)


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid SQL identifier in {label}.")
    return value


@dataclass(frozen=True)
class ReadOnlyTableSpec:
    """Application-owned table and field allowlist for structured SELECTs."""

    table: str
    columns: tuple[str, ...]
    filterable_columns: tuple[str, ...] = ()
    sortable_columns: tuple[str, ...] = ()
    default_limit: int = 100
    max_limit: int = 500

    def __post_init__(self) -> None:
        _validate_identifier(self.table, "table")
        if not self.columns:
            raise ValueError("ReadOnlyTableSpec.columns cannot be empty.")
        for column in self.columns:
            _validate_identifier(column, "columns")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("ReadOnlyTableSpec.columns must be unique.")
        for column in (*self.filterable_columns, *self.sortable_columns):
            _validate_identifier(column, "filter/sort columns")
            if column not in self.columns:
                raise ValueError(
                    f"Column '{column}' is not in the table column allowlist."
                )
        if not (1 <= self.default_limit <= self.max_limit <= 10000):
            raise ValueError(
                "Limits must satisfy 1 <= default_limit <= max_limit <= 10000."
            )


class BusinessDatabaseAdapter:
    """Read-only DB-API adapter using application-defined structured queries.

    No raw SQL, table names, columns, operators, or identifiers are accepted
    from the caller unless they match the static ReadOnlyTableSpec catalog.
    Values are always bound as DB-API parameters. The connection factory must
    return a fresh connection configured with database-level read-only access
    wherever the database supports that mode.
    """

    TARGET = "business_database"
    SELECT_CAPABILITY = "business_database.select"

    def __init__(
        self,
        *,
        connection_factory: Callable[[], Any],
        table_specs: Mapping[str, ReadOnlyTableSpec],
        paramstyle: str = "qmark",
        identifier_quote: str | None = None,
        dialect: str = "limit",
    ) -> None:
        if identifier_quote not in {None, '"', "`", "["}:
            raise ValueError("identifier_quote must be None, '\"', '`' or '['.")
        if dialect not in _DIALECTS:
            raise ValueError(f"dialect must be one of {sorted(_DIALECTS)}.")
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable.")
        if paramstyle not in {"qmark", "format", "numeric"}:
            raise ValueError("paramstyle must be 'qmark', 'format' or 'numeric'.")
        if not isinstance(table_specs, Mapping) or not table_specs:
            raise ValueError("table_specs must be a non-empty mapping.")
        normalized: dict[str, ReadOnlyTableSpec] = {}
        for key, spec in table_specs.items():
            _validate_identifier(key, "table_specs key")
            if not isinstance(spec, ReadOnlyTableSpec):
                raise TypeError(
                    "Every table_specs value must be ReadOnlyTableSpec."
                )
            if key != spec.table:
                raise ValueError("table_specs keys must match each spec.table.")
            normalized[key] = spec
        self._connection_factory = connection_factory
        self._table_specs = dict(normalized)
        self._paramstyle = paramstyle
        self._dialect = dialect
        # Identifiers are already validated against a strict pattern, so quoting
        # them can never be broken out of; it only makes reserved words and
        # mixed-case names usable. None keeps the historical unquoted SQL.
        self._quote = identifier_quote

    def _q(self, identifier: str) -> str:
        if self._quote is None:
            return identifier
        if self._quote == "[":  # SQL Server brackets
            return f"[{identifier}]"
        return f"{self._quote}{identifier}{self._quote}"

    def _placeholder(self, position: int) -> str:
        """Placeholder for the parameter at 1-based ``position`` in the final SQL text."""
        if self._paramstyle == "qmark":
            return "?"
        if self._paramstyle == "format":
            return "%s"
        return f":{position}"  # numeric (Oracle)

    @property
    def target(self) -> str:
        return self.TARGET

    def supports(self, capability_id: str) -> bool:
        return capability_id == self.SELECT_CAPABILITY

    def read(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.supports(capability_id):
            raise CapabilityRequestValidationError(
                f"Unsupported business database capability: {capability_id}"
            )
        query, parameters = self._build_select(arguments)
        connection = None
        cursor = None
        try:
            connection = self._connection_factory()
            if connection is None:
                raise RuntimeError("connection_factory returned None.")
            cursor = connection.cursor()
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            description = cursor.description or ()
            column_names = [item[0] for item in description]
            result: list[dict[str, Any]] = []
            for row in rows:
                if isinstance(row, Mapping):
                    record = dict(row)
                elif hasattr(row, "keys") and callable(row.keys):
                    record = {name: row[name] for name in row.keys()}
                else:
                    record = dict(zip(column_names, row))
                result.append({k: _json_safe(v) for k, v in record.items()})
            return result
        except CapabilityRequestValidationError:
            raise
        except Exception as exc:
            raise CapabilityAdapterError(
                "Read-only business database query failed."
            ) from exc
        finally:
            # Always attempt both closes. A cursor.close() failure must never
            # prevent the connection manager from releasing its lease.
            active_exception = sys.exc_info()[0] is not None
            cleanup_error = None

            if cursor is not None:
                close_cursor = getattr(cursor, "close", None)
                if callable(close_cursor):
                    try:
                        close_cursor()
                    except Exception as exc:
                        cleanup_error = exc

            if connection is not None:
                close_connection = getattr(connection, "close", None)
                if callable(close_connection):
                    try:
                        close_connection()
                    except Exception as exc:
                        if cleanup_error is None:
                            cleanup_error = exc

            # Do not mask an existing query/adapter exception with a secondary
            # cleanup error. If the query succeeded, surface cleanup failure
            # through the adapter's sanitized public error contract.
            if cleanup_error is not None and not active_exception:
                raise CapabilityAdapterError(
                    "Read-only business database resource cleanup failed."
                ) from cleanup_error

    def _condition_sql(self, item: Mapping[str, Any], bind: Callable[[Any], str]) -> str:
        column, operator, value = item["column"], item["operator"], item.get("value")
        quoted = self._q(column)
        if operator in {"is_null", "is_not_null"}:
            if value is not None:
                raise CapabilityRequestValidationError(f"'{operator}' takes no value.")
            return f"{quoted} IS {'NOT ' if operator == 'is_not_null' else ''}NULL"
        if operator in _COMPARISON_SQL:
            return f"{quoted} {_COMPARISON_SQL[operator]} {bind(_condition_scalar(value, column))}"
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise CapabilityRequestValidationError("'between' needs [low, high].")
            low, high = (_condition_scalar(v, column) for v in value)
            return f"{quoted} BETWEEN {bind(low)} AND {bind(high)}"
        if operator == "in":
            if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_IN_VALUES:
                raise CapabilityRequestValidationError(f"'in' needs 1-{MAX_IN_VALUES} values.")
            placeholders = ", ".join(bind(_condition_scalar(v, column)) for v in value)
            return f"{quoted} IN ({placeholders})"
        # contains / starts_with / ends_with
        if not isinstance(value, str):
            raise CapabilityRequestValidationError(f"'{operator}' needs text.")
        _condition_scalar(value, column)
        pattern = _like_pattern(value, before=operator in {"contains", "ends_with"},
                                after=operator in {"contains", "starts_with"})
        return f"{quoted} LIKE {bind(pattern)} ESCAPE '{_LIKE_ESCAPE}'"

    def _build_select(
        self,
        arguments: Mapping[str, Any],
    ) -> tuple[str, tuple[Any, ...]]:
        if not isinstance(arguments, Mapping):
            raise CapabilityRequestValidationError(
                "Database arguments must be a mapping."
            )
        unknown = set(arguments) - _ALLOWED_ARGUMENTS
        if unknown:
            raise CapabilityRequestValidationError(
                "Unsupported database argument(s): "
                + ", ".join(sorted(map(str, unknown)))
            )
        table = arguments.get("table")
        if not isinstance(table, str) or table not in self._table_specs:
            raise CapabilityRequestValidationError(
                "Requested table is not allowlisted."
            )
        spec = self._table_specs[table]

        requested_columns = arguments.get("columns", list(spec.columns))
        if not isinstance(requested_columns, (list, tuple)) or not requested_columns:
            raise CapabilityRequestValidationError(
                "columns must be a non-empty list."
            )
        if any(
            not isinstance(column, str) or column not in spec.columns
            for column in requested_columns
        ):
            raise CapabilityRequestValidationError(
                "Requested column is not allowlisted."
            )
        if len(set(requested_columns)) != len(requested_columns):
            raise CapabilityRequestValidationError(
                "columns must not contain duplicates."
            )

        filters = arguments.get("filters", {})
        if not isinstance(filters, Mapping):
            raise CapabilityRequestValidationError(
                "filters must be a mapping."
            )
        if any(
            not isinstance(key, str) or key not in spec.filterable_columns
            for key in filters
        ):
            raise CapabilityRequestValidationError(
                "A filter column is not allowlisted."
            )

        conditions = arguments.get("conditions", [])
        if not isinstance(conditions, (list, tuple)) or len(conditions) > MAX_CONDITIONS:
            raise CapabilityRequestValidationError(
                f"conditions must be a list of at most {MAX_CONDITIONS} items."
            )
        for item in conditions:
            if not isinstance(item, Mapping) or set(item) - {"column", "operator", "value"}:
                raise CapabilityRequestValidationError(
                    "Each condition must be {column, operator, value}."
                )
            if not isinstance(item.get("column"), str) or item["column"] not in spec.filterable_columns:
                raise CapabilityRequestValidationError("A filter column is not allowlisted.")
            if item.get("operator") not in CONDITION_OPERATORS:
                raise CapabilityRequestValidationError("A condition operator is not supported.")

        order_by = arguments.get("order_by", [])
        if not isinstance(order_by, (list, tuple)):
            raise CapabilityRequestValidationError(
                "order_by must be a list."
            )
        order_fragments: list[str] = []
        for item in order_by:
            if not isinstance(item, str) or not item:
                raise CapabilityRequestValidationError(
                    "Each order_by item must be a column string."
                )
            descending = item.startswith("-")
            column = item[1:] if descending else item
            if column not in spec.sortable_columns:
                raise CapabilityRequestValidationError(
                    "A sort column is not allowlisted."
                )
            order_fragments.append(
                f"{self._q(column)} {'DESC' if descending else 'ASC'}"
            )

        limit = arguments.get("limit", spec.default_limit)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not (1 <= limit <= spec.max_limit)
        ):
            raise CapabilityRequestValidationError(
                f"limit must be an integer between 1 and {spec.max_limit}."
            )

        parameters: list[Any] = []

        def bind(value: Any) -> str:
            # Placeholders are numbered by their order in the final SQL text, so a
            # dialect that puts the row limit first (TOP) still binds correctly.
            parameters.append(value)
            return self._placeholder(len(parameters))

        top = f"TOP ({bind(limit)}) " if self._dialect == "top" else ""
        sql = (f"SELECT {top}{', '.join(self._q(c) for c in requested_columns)} "
               f"FROM {self._q(table)}")
        where_fragments: list[str] = []
        for column, value in filters.items():
            if value is None:
                where_fragments.append(f"{self._q(column)} IS NULL")
            else:
                if isinstance(value, (dict, list, tuple, set)):
                    raise CapabilityRequestValidationError(
                        f"Filter value for '{column}' must be a scalar."
                    )
                where_fragments.append(f"{self._q(column)} = {bind(value)}")
        for item in conditions:
            where_fragments.append(self._condition_sql(item, bind))
        if where_fragments:
            sql += " WHERE " + " AND ".join(where_fragments)
        if order_fragments:
            sql += " ORDER BY " + ", ".join(order_fragments)
        if self._dialect == "limit":
            sql += f" LIMIT {bind(limit)}"
        elif self._dialect == "fetch_first":  # Oracle 12c+, standard SQL
            sql += f" FETCH FIRST {bind(limit)} ROWS ONLY"
        return sql, tuple(parameters)


__all__ = ["ReadOnlyTableSpec", "BusinessDatabaseAdapter"]
