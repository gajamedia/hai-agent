"""Read-only SQLite schema discovery and user-approved table allowlists."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from agent.llm.business_database_adapter import ReadOnlyTableSpec

from .connection_manager import ConnectionManager


class SQLiteSchemaDiscoveryError(RuntimeError):
    """SQLite metadata could not be read through the approved profile."""


class SQLiteAllowlistError(ValueError):
    """A requested table/column allowlist is not part of discovered schema."""


@dataclass(frozen=True)
class SQLiteColumn:
    name: str
    declared_type: str
    nullable: bool
    primary_key: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SQLiteTable:
    name: str
    columns: tuple[SQLiteColumn, ...]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "columns": [item.to_dict() for item in self.columns]}


@dataclass(frozen=True)
class SQLiteSchema:
    profile_name: str
    tables: tuple[SQLiteTable, ...]

    def to_dict(self) -> dict[str, object]:
        return {"profile_name": self.profile_name, "tables": [item.to_dict() for item in self.tables]}


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def discover_sqlite_schema(manager: ConnectionManager, *, profile_name: str) -> SQLiteSchema:
    """Read table and column metadata using an already-approved SQLite profile."""
    if not isinstance(manager, ConnectionManager):
        raise TypeError("manager must be ConnectionManager")
    try:
        profile = manager.get_profile(profile_name)
        if profile.database != "sqlite" or profile.read_only is not True:
            raise SQLiteSchemaDiscoveryError("schema discovery requires an approved SQLite read-only profile")
        with manager.connection(profile_name) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            tables: list[SQLiteTable] = []
            for row in rows:
                name = row[0]
                if not isinstance(name, str) or not name:
                    raise SQLiteSchemaDiscoveryError("SQLite returned an invalid table name")
                columns = connection.execute(
                    f"PRAGMA table_info({_quoted_identifier(name)})"
                ).fetchall()
                parsed = tuple(
                    SQLiteColumn(
                        name=str(column[1]),
                        declared_type=str(column[2] or ""),
                        nullable=not bool(column[3]),
                        primary_key=bool(column[5]),
                    )
                    for column in columns
                )
                if not parsed:
                    raise SQLiteSchemaDiscoveryError("SQLite returned a table without columns")
                tables.append(SQLiteTable(name=name, columns=parsed))
    except SQLiteSchemaDiscoveryError:
        raise
    except Exception as exc:
        raise SQLiteSchemaDiscoveryError("SQLite schema discovery failed") from exc
    return SQLiteSchema(profile_name=profile_name, tables=tuple(tables))


def build_readonly_table_specs(
    schema: SQLiteSchema,
    selections: Mapping[str, tuple[str, ...] | list[str]],
    *,
    default_limit: int = 100,
    max_limit: int = 500,
) -> dict[str, ReadOnlyTableSpec]:
    """Convert explicit user selections into adapter allowlist specifications."""
    if not isinstance(schema, SQLiteSchema):
        raise TypeError("schema must be SQLiteSchema")
    if not isinstance(selections, Mapping) or not selections:
        raise SQLiteAllowlistError("select at least one discovered table and column")
    by_name = {table.name: table for table in schema.tables}
    result: dict[str, ReadOnlyTableSpec] = {}
    for table_name, requested_columns in selections.items():
        table = by_name.get(table_name)
        if table is None:
            raise SQLiteAllowlistError("selected table is not in the discovered schema")
        if not isinstance(requested_columns, (tuple, list)) or not requested_columns:
            raise SQLiteAllowlistError("each selected table needs at least one column")
        available = {column.name for column in table.columns}
        columns = tuple(requested_columns)
        if any(not isinstance(column, str) or column not in available for column in columns):
            raise SQLiteAllowlistError("selected column is not in the discovered schema")
        if len(columns) != len(set(columns)):
            raise SQLiteAllowlistError("selected columns must be unique")
        result[table_name] = ReadOnlyTableSpec(
            table=table_name,
            columns=columns,
            filterable_columns=columns,
            sortable_columns=columns,
            default_limit=default_limit,
            max_limit=max_limit,
        )
    return result


__all__ = [
    "SQLiteAllowlistError", "SQLiteColumn", "SQLiteSchema", "SQLiteSchemaDiscoveryError",
    "SQLiteTable", "build_readonly_table_specs", "discover_sqlite_schema",
]
