from __future__ import annotations

import sqlite3

import pytest

from agent.core.action import Action
from agent.database import ConnectionManager, ConnectionPolicyError
from agent.database.capability_gateway import (
    BusinessDatabaseGatewayConfigurationError,
    DEFAULT_CAPABILITY_ID,
    build_business_database_gateway,
)
from agent.llm.authenticated_capability_execution_gateway import (
    AuthenticatedCapabilityReadRequest,
)
from agent.llm.capability_execution_contracts import CapabilityAuthorizationError
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy


PERMISSION = "business:database:read"


def _database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, total INTEGER)"
        )
        connection.executemany(
            "INSERT INTO orders (id, status, total) VALUES (?, ?, ?)",
            [
                (1, "paid", 150000),
                (2, "pending", 90000),
                (3, "paid", 225000),
            ],
        )


def _manager(path, *, guard=None):
    manager = ConnectionManager()

    def default_guard(connection):
        connection.execute("PRAGMA query_only = ON")
        row = connection.execute("PRAGMA query_only").fetchone()
        if not row or row[0] != 1:
            raise PermissionError("SQLite query_only could not be enforced.")

    manager.register(
        name="test_business_ro",
        database="sqlite",
        authorization_ref="test-approval:read-only-profile",
        connection_factory=lambda: sqlite3.connect(path),
        read_only_guard=guard or default_guard,
        read_only=True,
    )
    return manager


def _assembly(tmp_path, *, guard=None, policy=None):
    database_path = tmp_path / "business.sqlite3"
    _database(database_path)
    manager = _manager(database_path, guard=guard)
    assembly = build_business_database_gateway(
        connection_manager=manager,
        profile_name="test_business_ro",
        authorization_policy=policy or AuthorizationPolicy(),
        table_specs={
            "orders": __import__(
                "agent.llm.business_database_adapter",
                fromlist=["ReadOnlyTableSpec"],
            ).ReadOnlyTableSpec(
                table="orders",
                columns=("id", "status", "total"),
                filterable_columns=("status",),
                sortable_columns=("id", "total"),
                default_limit=20,
                max_limit=50,
            )
        },
        trace_path=tmp_path / "trace.jsonl",
        paramstyle="qmark",
        fsync_trace=False,
    )
    return manager, assembly


def _principal():
    return AuthenticatedPrincipal(
        actor_id="test-operator",
        permissions=(PERMISSION,),
    )


def _request():
    return AuthenticatedCapabilityReadRequest(
        target="business_database",
        capability_id=DEFAULT_CAPABILITY_ID,
        arguments={
            "table": "orders",
            "columns": ["id", "status", "total"],
            "filters": {"status": "paid"},
            "order_by": ["-total"],
            "limit": 10,
        },
    )


def test_composed_gateway_reads_through_connection_manager_and_traces(tmp_path):
    manager, assembly = _assembly(tmp_path)

    result = assembly.gateway.read(
        _principal(),
        _request(),
        execution_id="db-read-001",
    )

    assert result.data == [
        {"id": 3, "status": "paid", "total": 225000},
        {"id": 1, "status": "paid", "total": 150000},
    ]
    assert manager.active_connection_count == 0
    assert [event["event_type"] for event in assembly.trace_store.replay("db-read-001")] == [
        "started",
        "authorized",
        "completed",
    ]
    assert all(
        "arguments" not in event["payload"] and "data" not in event["payload"]
        for event in assembly.trace_store.replay("db-read-001")
    )
    manager.shutdown()


def test_connection_guard_enforces_sqlite_query_only(tmp_path):
    manager, _ = _assembly(tmp_path)

    with pytest.raises(sqlite3.OperationalError):
        with manager.connection("test_business_ro") as connection:
            connection.execute("UPDATE orders SET total = 1 WHERE id = 1")

    assert manager.active_connection_count == 0
    manager.shutdown()


def test_failed_read_only_guard_blocks_adapter_and_closes_connection(tmp_path):
    created = []

    def deny(connection):
        created.append(connection)
        raise PermissionError("read-only principal verification failed")

    manager, assembly = _assembly(tmp_path, guard=deny)

    with pytest.raises(Exception):
        assembly.gateway.read(
            _principal(),
            _request(),
            execution_id="db-read-denied-by-guard",
        )

    assert len(created) == 1
    assert manager.active_connection_count == 0
    events = assembly.trace_store.replay("db-read-denied-by-guard")
    assert events[0]["event_type"] == "started"
    assert events[-1]["event_type"] == "failed"
    manager.shutdown()


def test_missing_permission_is_denied_before_database_connection(tmp_path):
    manager, assembly = _assembly(tmp_path)

    principal = AuthenticatedPrincipal(actor_id="no-permission")
    with pytest.raises(CapabilityAuthorizationError):
        assembly.gateway.read(
            principal,
            _request(),
            execution_id="db-read-no-permission",
        )

    assert manager.active_connection_count == 0
    assert assembly.trace_store.replay("db-read-no-permission")[-1]["event_type"] == "denied"
    manager.shutdown()


def test_unknown_connection_profile_fails_during_assembly(tmp_path):
    manager = ConnectionManager()

    with pytest.raises(Exception):
        build_business_database_gateway(
            connection_manager=manager,
            profile_name="missing",
            authorization_policy=AuthorizationPolicy(),
            table_specs={},
            trace_path=tmp_path / "trace.jsonl",
            paramstyle="qmark",
        )

    manager.shutdown()


def test_connection_manager_rejects_non_read_only_profile(tmp_path):
    manager = ConnectionManager()

    with pytest.raises(ConnectionPolicyError):
        manager.register(
            name="not_read_only",
            database="sqlite",
            authorization_ref="test-approval:wrong-profile",
            connection_factory=lambda: sqlite3.connect(":memory:"),
            read_only_guard=lambda connection: None,
            read_only=False,
        )

    manager.shutdown()
