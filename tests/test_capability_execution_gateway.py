from __future__ import annotations

import pytest

from agent.llm.business_database_adapter import (
    BusinessDatabaseAdapter,
    ReadOnlyTableSpec,
)
from agent.llm.capability_execution_contracts import (
    CapabilityAuthorizationError,
    CapabilityReadRequest,
    CapabilityRequestValidationError,
    CapabilityResolutionError,
)
from agent.llm.capability_execution_gateway import CapabilityExecutionGateway
from agent.llm.room_environment_adapter import RoomEnvironmentAdapter


class FakeRoomEnvironment:
    def __init__(self):
        self.state = {"door": "locked", "box": "closed"}
        self.observe_calls = 0

    def observe(self):
        self.observe_calls += 1
        return dict(self.state)


class FakeCursor:
    def __init__(self, rows=(), description=()):
        self.rows = list(rows)
        self.description = description
        self.query = None
        self.parameters = None
        self.closed = False

    def execute(self, query, parameters):
        self.query = query
        self.parameters = parameters

    def fetchall(self):
        return list(self.rows)

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def test_room_adapter_only_observes_state_and_returns_detached_snapshot():
    environment = FakeRoomEnvironment()
    adapter = RoomEnvironmentAdapter(environment)
    snapshot = adapter.read(adapter.OBSERVE_CAPABILITY, {})
    snapshot["door"] = "unlocked"

    assert environment.state["door"] == "locked"
    assert environment.observe_calls == 1
    assert not hasattr(adapter, "execute")
    assert not hasattr(adapter, "reset")
    assert not hasattr(adapter, "set_state")


def test_room_adapter_rejects_unknown_capability_and_arguments():
    adapter = RoomEnvironmentAdapter(FakeRoomEnvironment())
    with pytest.raises(CapabilityRequestValidationError):
        adapter.read("room_environment.execute_action", {"action": "open_box"})
    with pytest.raises(CapabilityRequestValidationError):
        adapter.read(adapter.OBSERVE_CAPABILITY, {"action": "open_box"})


def test_gateway_is_fail_closed_without_authorizer():
    gateway = CapabilityExecutionGateway([RoomEnvironmentAdapter(FakeRoomEnvironment())])
    request = CapabilityReadRequest(
        target="room_environment",
        capability_id=RoomEnvironmentAdapter.OBSERVE_CAPABILITY,
        actor_id="agent-1",
    )
    with pytest.raises(CapabilityAuthorizationError):
        gateway.read(request)


def test_gateway_reads_only_after_explicit_authorization():
    environment = FakeRoomEnvironment()
    seen = []

    def authorize(actor, target, capability):
        seen.append((actor, target, capability))
        return True

    gateway = CapabilityExecutionGateway(
        [RoomEnvironmentAdapter(environment)],
        authorizer=authorize,
    )
    result = gateway.read(CapabilityReadRequest(
        target="room_environment",
        capability_id=RoomEnvironmentAdapter.OBSERVE_CAPABILITY,
        actor_id="operator-7",
    ))

    assert result.data == {"door": "locked", "box": "closed"}
    assert result.metadata["read_only"] is True
    assert seen == [(
        "operator-7",
        "room_environment",
        RoomEnvironmentAdapter.OBSERVE_CAPABILITY,
    )]


def test_gateway_denies_authorizer_exception_and_non_true_values():
    adapter = RoomEnvironmentAdapter(FakeRoomEnvironment())
    request = CapabilityReadRequest(
        target=adapter.target,
        capability_id=adapter.OBSERVE_CAPABILITY,
        actor_id="operator-7",
    )
    for authorizer in (
        lambda *_: False,
        lambda *_: 1,
        lambda *_: (_ for _ in ()).throw(RuntimeError("auth unavailable")),
    ):
        gateway = CapabilityExecutionGateway([adapter], authorizer=authorizer)
        with pytest.raises(CapabilityAuthorizationError):
            gateway.read(request)


def test_gateway_rejects_unknown_target_and_capability():
    adapter = RoomEnvironmentAdapter(FakeRoomEnvironment())
    gateway = CapabilityExecutionGateway([adapter], authorizer=lambda *_: True)
    with pytest.raises(CapabilityResolutionError):
        gateway.read(CapabilityReadRequest(
            target="unknown", capability_id="x.read", actor_id="operator"
        ))
    with pytest.raises(CapabilityResolutionError):
        gateway.read(CapabilityReadRequest(
            target=adapter.target, capability_id="room_environment.execute",
            actor_id="operator"
        ))


def test_business_database_adapter_builds_parameterized_allowlisted_select():
    cursor = FakeCursor(
        rows=[(7, "Bread")],
        description=[("id",), ("name",)],
    )
    connection = FakeConnection(cursor)
    adapter = BusinessDatabaseAdapter(
        connection_factory=lambda: connection,
        table_specs={
            "products": ReadOnlyTableSpec(
                table="products",
                columns=("id", "name", "status"),
                filterable_columns=("status",),
                sortable_columns=("id", "name"),
                default_limit=25,
                max_limit=100,
            )
        },
    )

    rows = adapter.read(adapter.SELECT_CAPABILITY, {
        "table": "products",
        "columns": ["id", "name"],
        "filters": {"status": "active' OR 1=1 --"},
        "order_by": ["-id"],
        "limit": 10,
    })

    assert rows == [{"id": 7, "name": "Bread"}]
    assert cursor.query == (
        "SELECT id, name FROM products WHERE status = ? ORDER BY id DESC LIMIT ?"
    )
    assert cursor.parameters == ("active' OR 1=1 --", 10)
    assert cursor.closed is True
    assert connection.closed is True


def test_business_database_adapter_rejects_unallowlisted_inputs_before_connecting():
    calls = []
    adapter = BusinessDatabaseAdapter(
        connection_factory=lambda: calls.append("connected"),
        table_specs={
            "products": ReadOnlyTableSpec(
                table="products",
                columns=("id", "name"),
                filterable_columns=("id",),
                sortable_columns=("id",),
            )
        },
    )
    invalid_arguments = [
        {"table": "users"},
        {"table": "products; DROP TABLE products"},
        {"table": "products", "columns": ["id; DROP TABLE products"]},
        {"table": "products", "filters": {"name": "Bread"}},
        {"table": "products", "order_by": ["name"]},
        {"table": "products", "limit": 0},
        {"table": "products", "sql": "DELETE FROM products"},
    ]
    for arguments in invalid_arguments:
        with pytest.raises(CapabilityRequestValidationError):
            adapter.read(adapter.SELECT_CAPABILITY, arguments)
    assert calls == []


def test_business_database_adapter_supports_format_paramstyle():
    cursor = FakeCursor(rows=[], description=[])
    connection = FakeConnection(cursor)
    adapter = BusinessDatabaseAdapter(
        connection_factory=lambda: connection,
        table_specs={
            "orders": ReadOnlyTableSpec(
                table="orders",
                columns=("id", "status"),
                filterable_columns=("status",),
            )
        },
        paramstyle="format",
    )
    adapter.read(adapter.SELECT_CAPABILITY, {
        "table": "orders",
        "filters": {"status": "open"},
    })
    assert cursor.query == "SELECT id, status FROM orders WHERE status = %s LIMIT %s"
    assert cursor.parameters == ("open", 100)


def test_gateway_adapter_error_does_not_leak_database_exception_details():
    class BrokenCursor:
        description = []

        def execute(self, *_):
            raise RuntimeError("secret connection string")

        def close(self):
            pass

    connection = FakeConnection(BrokenCursor())
    adapter = BusinessDatabaseAdapter(
        connection_factory=lambda: connection,
        table_specs={
            "products": ReadOnlyTableSpec(table="products", columns=("id",))
        },
    )
    gateway = CapabilityExecutionGateway([adapter], authorizer=lambda *_: True)
    request = CapabilityReadRequest(
        target=adapter.target,
        capability_id=adapter.SELECT_CAPABILITY,
        actor_id="operator",
        arguments={"table": "products"},
    )
    with pytest.raises(Exception) as captured:
        gateway.read(request)
    assert "secret connection string" not in str(captured.value)
