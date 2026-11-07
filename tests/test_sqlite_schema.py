import sqlite3

import pytest

from agent.database import (
    ConnectionManager,
    SQLiteAllowlistError,
    build_readonly_table_specs,
    discover_sqlite_schema,
    register_sqlite_read_only_profile,
)


def configured_manager(tmp_path):
    path = tmp_path / "business.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT)")
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total INTEGER NOT NULL)")
    manager = ConnectionManager()
    register_sqlite_read_only_profile(
        manager,
        profile_name="business_ro",
        database_path=path,
        authorization_ref="user-approval:schema-001",
    )
    return manager


def test_schema_discovery_reads_only_user_approved_sqlite_profile(tmp_path):
    manager = configured_manager(tmp_path)
    schema = discover_sqlite_schema(manager, profile_name="business_ro")

    assert [table.name for table in schema.tables] == ["customers", "orders"]
    assert [column.name for column in schema.tables[0].columns] == ["id", "name", "city"]
    assert schema.tables[0].columns[0].primary_key is True
    manager.shutdown()


def test_allowlist_only_contains_user_selected_columns(tmp_path):
    manager = configured_manager(tmp_path)
    schema = discover_sqlite_schema(manager, profile_name="business_ro")
    specs = build_readonly_table_specs(schema, {"orders": ["id", "total"]})

    assert tuple(specs) == ("orders",)
    assert specs["orders"].columns == ("id", "total")
    assert specs["orders"].filterable_columns == ("id", "total")
    manager.shutdown()


def test_allowlist_rejects_undiscovered_table_or_column(tmp_path):
    manager = configured_manager(tmp_path)
    schema = discover_sqlite_schema(manager, profile_name="business_ro")
    with pytest.raises(SQLiteAllowlistError):
        build_readonly_table_specs(schema, {"orders": ["secret"]})
    with pytest.raises(SQLiteAllowlistError):
        build_readonly_table_specs(schema, {"unknown": ["id"]})
    manager.shutdown()
