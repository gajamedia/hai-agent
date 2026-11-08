from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.database import ConnectionManager
from agent.database.capability_gateway import (
    DEFAULT_CAPABILITY_ID,
    build_business_database_gateway,
)
from agent.llm.authenticated_capability_execution_gateway import (
    AuthenticatedCapabilityReadRequest,
)
from agent.llm.business_database_adapter import ReadOnlyTableSpec
from agent.llm.capability_execution_contracts import CapabilityAuthorizationError
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy


DB_PERMISSION = "business:database:read"
PROFILE_NAME = "isolated_sqlite_ro"


class TrackingSQLiteConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        return super().close()


def create_fixture_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                total INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO orders (id, status, total) VALUES (?, ?, ?)",
            [
                (1, "paid", 150000),
                (2, "pending", 90000),
                (3, "paid", 225000),
            ],
        )


def readonly_sqlite_factory(path: Path, created: list) :
    # mode=ro is enforced by SQLite's file-opening mode. query_only is an
    # additional per-connection guard, not a replacement for mode=ro.
    uri = f"{path.resolve().as_uri()}?mode=ro"
    def factory():
        connection = sqlite3.connect(
            uri,
            uri=True,
            factory=TrackingSQLiteConnection,
        )
        created.append(connection)
        return connection
    return factory


def sqlite_read_only_guard(connection) -> None:
    connection.execute("PRAGMA query_only = ON")
    query_only = connection.execute("PRAGMA query_only").fetchone()
    database_list = connection.execute("PRAGMA database_list").fetchall()
    if not query_only or query_only[0] != 1:
        raise PermissionError("SQLite query_only guard could not be verified.")
    if not database_list:
        raise PermissionError("SQLite database attachment could not be verified.")


def register_profile(
    manager: ConnectionManager,
    path: Path,
    created: list,
    *,
    guard=sqlite_read_only_guard,
    factory=None,
) -> list:
    guard_calls = []

    def checked_guard(connection):
        guard_calls.append(connection)
        guard(connection)

    manager.register(
        name=PROFILE_NAME,
        database="sqlite",
        authorization_ref="test-only:isolated-read-only-db",
        connection_factory=factory or readonly_sqlite_factory(path, created),
        read_only_guard=checked_guard,
        read_only=True,
        enabled=True,
    )
    return guard_calls


def build_gateway(manager, tmp_path):
    return build_business_database_gateway(
        connection_manager=manager,
        profile_name=PROFILE_NAME,
        authorization_policy=AuthorizationPolicy(),
        table_specs={
            "orders": ReadOnlyTableSpec(
                table="orders",
                columns=("id", "status", "total"),
                filterable_columns=("status",),
                sortable_columns=("id", "total"),
                default_limit=20,
                max_limit=50,
            )
        },
        trace_path=tmp_path / "runtime-verification-trace.jsonl",
        paramstyle="qmark",
        fsync_trace=False,
    )


def principal(*permissions):
    return AuthenticatedPrincipal(
        actor_id="p3565i-test-operator",
        permissions=permissions or (DB_PERMISSION,),
    )


def request(*, filters=None, limit=10):
    return AuthenticatedCapabilityReadRequest(
        target="business_database",
        capability_id=DEFAULT_CAPABILITY_ID,
        arguments={
            "table": "orders",
            "columns": ["id", "status", "total"],
            "filters": filters or {"status": "paid"},
            "order_by": ["-total"],
            "limit": limit,
        },
    )


def test_runtime_read_uses_fresh_guarded_connection_for_every_call(tmp_path):
    database_path = tmp_path / "isolated-business.sqlite3"
    create_fixture_database(database_path)

    manager = ConnectionManager()
    created = []
    guard_calls = register_profile(manager, database_path, created)
    assembly = build_gateway(manager, tmp_path)

    try:
        first = assembly.gateway.read(
            principal(),
            request(),
            execution_id="runtime-read-001",
        )
        second = assembly.gateway.read(
            principal(),
            request(filters={"status": "pending"}),
            execution_id="runtime-read-002",
        )

        assert first.data == [
            {"id": 3, "status": "paid", "total": 225000},
            {"id": 1, "status": "paid", "total": 150000},
        ]
        assert second.data == [
            {"id": 2, "status": "pending", "total": 90000},
        ]
        assert len(created) == 2
        assert len(guard_calls) == 2
        assert created[0] is not created[1]
        assert all(connection.was_closed for connection in created)
        assert manager.active_connection_count == 0
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO orders (id, status, total) VALUES (99, 'paid', 1)",
        "UPDATE orders SET total = 1 WHERE id = 1",
        "DELETE FROM orders WHERE id = 1",
        "CREATE TABLE injected_table (id INTEGER)",
        "DROP TABLE orders",
        "ALTER TABLE orders ADD COLUMN injected TEXT",
    ],
)
def test_sqlite_read_only_profile_rejects_mutation_even_if_query_only_is_disabled(
    tmp_path,
    statement,
):
    database_path = tmp_path / "readonly-enforcement.sqlite3"
    create_fixture_database(database_path)

    manager = ConnectionManager()
    created = []
    register_profile(manager, database_path, created)

    try:
        with pytest.raises(sqlite3.OperationalError):
            with manager.connection(PROFILE_NAME) as connection:
                # Demonstrate that query_only alone is not the relied-upon
                # security boundary: mode=ro must still reject the write.
                connection.execute("PRAGMA query_only = OFF")
                connection.execute(statement)

        assert len(created) == 1
        assert created[0].was_closed
        assert manager.active_connection_count == 0
    finally:
        manager.shutdown()

    with sqlite3.connect(database_path) as verification:
        tables = {
            row[0]
            for row in verification.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert tables == {"orders"}
        rows = verification.execute(
            "SELECT id, status, total FROM orders ORDER BY id"
        ).fetchall()
        assert rows == [
            (1, "paid", 150000),
            (2, "pending", 90000),
            (3, "paid", 225000),
        ]


def test_parameter_values_are_bound_and_cannot_change_query_structure(tmp_path):
    database_path = tmp_path / "parameter-binding.sqlite3"
    create_fixture_database(database_path)

    manager = ConnectionManager()
    created = []
    register_profile(manager, database_path, created)
    assembly = build_gateway(manager, tmp_path)

    malicious_value = "paid' OR 1=1 --"
    try:
        result = assembly.gateway.read(
            principal(),
            request(filters={"status": malicious_value}),
            execution_id="runtime-injection-value-001",
        )
        assert result.data == []
        assert manager.active_connection_count == 0
    finally:
        manager.shutdown()

    with sqlite3.connect(database_path) as verification:
        assert verification.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3


def test_permission_denial_happens_before_opening_database(tmp_path):
    database_path = tmp_path / "permission-check.sqlite3"
    create_fixture_database(database_path)

    manager = ConnectionManager()
    created = []
    register_profile(manager, database_path, created)
    assembly = build_gateway(manager, tmp_path)

    try:
        with pytest.raises(CapabilityAuthorizationError):
            assembly.gateway.read(
                principal("unrelated:permission"),
                request(),
                execution_id="runtime-permission-denied-001",
            )

        assert created == []
        assert manager.active_connection_count == 0
        events = assembly.trace_store.replay("runtime-permission-denied-001")
        assert [event["event_type"] for event in events] == ["started", "denied"]
    finally:
        manager.shutdown()


def test_failed_guard_closes_connection_and_never_yields_it_to_adapter(tmp_path):
    database_path = tmp_path / "guard-failure.sqlite3"
    create_fixture_database(database_path)

    manager = ConnectionManager()
    created = []

    def deny_guard(_connection):
        raise PermissionError("read-only verification deliberately failed")

    register_profile(
        manager,
        database_path,
        created,
        guard=deny_guard,
    )
    assembly = build_gateway(manager, tmp_path)

    try:
        with pytest.raises(Exception):
            assembly.gateway.read(
                principal(),
                request(),
                execution_id="runtime-guard-denied-001",
            )

        assert len(created) == 1
        assert created[0].was_closed
        assert manager.active_connection_count == 0
        events = assembly.trace_store.replay("runtime-guard-denied-001")
        assert events[0]["event_type"] == "started"
        assert events[-1]["event_type"] == "failed"
    finally:
        manager.shutdown()


def test_trace_contains_lifecycle_metadata_but_not_query_or_result_data(tmp_path):
    database_path = tmp_path / "trace-redaction.sqlite3"
    create_fixture_database(database_path)

    manager = ConnectionManager()
    created = []
    register_profile(manager, database_path, created)
    assembly = build_gateway(manager, tmp_path)

    try:
        result = assembly.gateway.read(
            principal(),
            request(),
            execution_id="runtime-trace-redaction-001",
        )
        assert result.data
        records = assembly.trace_store.replay("runtime-trace-redaction-001")
        assert [record["event_type"] for record in records] == [
            "started",
            "authorized",
            "completed",
        ]

        serialized = repr(records)
        assert "225000" not in serialized
        assert "paid" not in serialized
        assert "arguments" not in serialized
        assert "password" not in serialized.lower()
        assert "credential" not in serialized.lower()
    finally:
        manager.shutdown()
