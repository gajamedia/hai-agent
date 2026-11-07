"""Tests for explicit read-only connection lifecycle management."""

import pytest

from agent.database import (
    ConnectionManager,
    ConnectionManagerError,
    ConnectionNotConfiguredError,
    ConnectionPolicyError,
)


class FakeCursor:
    def __init__(self, row=(1,)):
        self.row = row
        self.executed = []
        self.closed = False

    def execute(self, sql):
        self.executed.append(sql)

    def fetchone(self):
        return self.row

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, row=(1,)):
        self.cursor_instance = FakeCursor(row)
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True


def register(manager, connection, guard=None):
    manager.register(
        name="business_ro",
        database="mysql",
        authorization_ref="approval:ticket-123",
        connection_factory=lambda: connection,
        read_only_guard=guard or (lambda conn: None),
        read_only=True,
    )


def test_register_requires_authorization_reference():
    manager = ConnectionManager()
    with pytest.raises(ConnectionPolicyError):
        manager.register(
            name="business_ro",
            database="mysql",
            authorization_ref="",
            connection_factory=FakeConnection,
            read_only_guard=lambda conn: None,
        )


def test_register_requires_read_only_guard():
    manager = ConnectionManager()
    with pytest.raises(ConnectionPolicyError):
        manager.register(
            name="business_ro",
            database="mysql",
            authorization_ref="approval:ticket-123",
            connection_factory=FakeConnection,
            read_only_guard=None,
        )


def test_write_profile_is_rejected():
    manager = ConnectionManager()
    with pytest.raises(ConnectionPolicyError):
        manager.register(
            name="business_rw",
            database="mysql",
            authorization_ref="approval:ticket-123",
            connection_factory=FakeConnection,
            read_only_guard=lambda conn: None,
            read_only=False,
        )


def test_connection_is_closed_after_context_exit():
    manager = ConnectionManager()
    connection = FakeConnection()
    register(manager, connection)

    with manager.connection("business_ro") as yielded:
        assert yielded is connection
        assert manager.active_connection_count == 1

    assert connection.closed is True
    assert manager.active_connection_count == 0


def test_guard_failure_closes_connection_and_blocks_access():
    manager = ConnectionManager()
    connection = FakeConnection()

    def deny(_connection):
        raise PermissionError("read-only principal verification failed")

    register(manager, connection, guard=deny)
    with pytest.raises(PermissionError):
        with manager.connection("business_ro"):
            pytest.fail("connection must not be yielded")

    assert connection.closed is True
    assert manager.active_connection_count == 0


def test_health_check_executes_fixed_select_and_closes_resources():
    manager = ConnectionManager()
    connection = FakeConnection()
    register(manager, connection)

    health = manager.health_check("business_ro")

    assert health.healthy is True
    assert connection.cursor_instance.executed == ["SELECT 1"]
    assert connection.cursor_instance.closed is True
    assert connection.closed is True


def test_health_check_does_not_expose_exception_message():
    manager = ConnectionManager()

    def fail_factory():
        raise RuntimeError("password=do-not-leak")

    manager.register(
        name="broken",
        database="postgresql",
        authorization_ref="approval:ticket-456",
        connection_factory=fail_factory,
        read_only_guard=lambda conn: None,
    )
    health = manager.health_check("broken")

    assert health.healthy is False
    assert health.error_type == "RuntimeError"
    assert not hasattr(health, "error_message")


def test_unknown_profile_is_rejected():
    manager = ConnectionManager()
    with pytest.raises(ConnectionNotConfiguredError):
        with manager.connection("missing"):
            pass


def test_duplicate_profile_is_rejected():
    manager = ConnectionManager()
    register(manager, FakeConnection())
    with pytest.raises(ValueError):
        register(manager, FakeConnection())


def test_shutdown_prevents_new_connections():
    manager = ConnectionManager()
    register(manager, FakeConnection())
    manager.shutdown()

    with pytest.raises(ConnectionManagerError):
        with manager.connection("business_ro"):
            pass


def test_connection_factory_returns_dbapi_handle_and_close_releases_lease():
    manager = ConnectionManager()
    connection = FakeConnection()
    register(manager, connection)

    handle = manager.connection_factory("business_ro")()
    assert handle.cursor() is connection.cursor_instance
    assert manager.active_connection_count == 1

    handle.close()
    handle.close()

    assert connection.closed is True
    assert manager.active_connection_count == 0


def test_connection_factory_opens_a_fresh_connection_per_call():
    manager = ConnectionManager()
    created = []

    def factory():
        connection = FakeConnection()
        created.append(connection)
        return connection

    manager.register(
        name="business_ro",
        database="mysql",
        authorization_ref="approval:ticket-789",
        connection_factory=factory,
        read_only_guard=lambda conn: None,
    )

    connection_factory = manager.connection_factory("business_ro")
    first = connection_factory()
    second = connection_factory()

    assert len(created) == 2
    assert first is not second

    first.close()
    second.close()
    assert all(connection.closed for connection in created)
