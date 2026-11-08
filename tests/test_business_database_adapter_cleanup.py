from __future__ import annotations

import pytest

from agent.llm.business_database_adapter import (
    BusinessDatabaseAdapter,
    ReadOnlyTableSpec,
)
from agent.llm.capability_execution_contracts import CapabilityAdapterError


CAPABILITY = "business_database.select"
ARGUMENTS = {
    "table": "orders",
    "columns": ["id"],
    "filters": {},
    "limit": 5,
}


class FakeCursor:
    description = [("id",)]

    def __init__(self, *, close_error=None, execute_error=None):
        self.close_error = close_error
        self.execute_error = execute_error

    def execute(self, _query, _parameters):
        if self.execute_error:
            raise self.execute_error

    def fetchall(self):
        return [(1,)]

    def close(self):
        if self.close_error:
            raise self.close_error


class FakeConnection:
    def __init__(self, cursor, *, close_error=None):
        self._cursor = cursor
        self.close_error = close_error
        self.close_attempted = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.close_attempted = True
        if self.close_error:
            raise self.close_error


def make_adapter(connection):
    return BusinessDatabaseAdapter(
        connection_factory=lambda: connection,
        table_specs={
            "orders": ReadOnlyTableSpec(
                table="orders",
                columns=("id",),
                default_limit=5,
                max_limit=10,
            )
        },
        paramstyle="qmark",
    )


def test_connection_close_is_attempted_when_cursor_close_raises():
    cursor = FakeCursor(close_error=RuntimeError("cursor close failed"))
    connection = FakeConnection(cursor)
    adapter = make_adapter(connection)

    with pytest.raises(
        CapabilityAdapterError,
        match="resource cleanup failed",
    ):
        adapter.read(CAPABILITY, ARGUMENTS)

    assert connection.close_attempted is True


def test_original_query_error_is_not_masked_by_cleanup_errors():
    cursor = FakeCursor(
        close_error=RuntimeError("cursor close secret"),
        execute_error=RuntimeError("query failure secret"),
    )
    connection = FakeConnection(
        cursor,
        close_error=RuntimeError("connection close secret"),
    )
    adapter = make_adapter(connection)

    with pytest.raises(
        CapabilityAdapterError,
        match="Read-only business database query failed",
    ) as captured:
        adapter.read(CAPABILITY, ARGUMENTS)

    assert connection.close_attempted is True
    assert "query failure secret" in str(captured.value.__cause__)
    assert "cursor close secret" not in str(captured.value)
    assert "connection close secret" not in str(captured.value)


def test_connection_close_failure_is_reported_after_successful_read():
    cursor = FakeCursor()
    connection = FakeConnection(
        cursor,
        close_error=RuntimeError("connection close failed"),
    )
    adapter = make_adapter(connection)

    with pytest.raises(
        CapabilityAdapterError,
        match="resource cleanup failed",
    ):
        adapter.read(CAPABILITY, ARGUMENTS)

    assert connection.close_attempted is True
