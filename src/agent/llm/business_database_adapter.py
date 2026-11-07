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
_ALLOWED_ARGUMENTS = frozenset({"table", "columns", "filters", "order_by", "limit"})


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
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable.")
        if paramstyle not in {"qmark", "format"}:
            raise ValueError("paramstyle must be 'qmark' or 'format'.")
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
        self._placeholder = "?" if paramstyle == "qmark" else "%s"

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
                f"{column} {'DESC' if descending else 'ASC'}"
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

        sql = f"SELECT {', '.join(requested_columns)} FROM {table}"
        parameters: list[Any] = []
        if filters:
            where_fragments: list[str] = []
            for column, value in filters.items():
                if value is None:
                    where_fragments.append(f"{column} IS NULL")
                else:
                    if isinstance(value, (dict, list, tuple, set)):
                        raise CapabilityRequestValidationError(
                            f"Filter value for '{column}' must be a scalar."
                        )
                    where_fragments.append(
                        f"{column} = {self._placeholder}"
                    )
                    parameters.append(value)
            sql += " WHERE " + " AND ".join(where_fragments)
        if order_fragments:
            sql += " ORDER BY " + ", ".join(order_fragments)
        sql += f" LIMIT {self._placeholder}"
        parameters.append(limit)
        return sql, tuple(parameters)


__all__ = ["ReadOnlyTableSpec", "BusinessDatabaseAdapter"]
