from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock

import pytest
from fastapi.testclient import TestClient

from agent.database.connection_manager import ConnectionManager
from agent.llm.authenticated_capability_execution_gateway import (
    AuthenticatedCapabilityReadRequest,
)
from agent.llm.business_database_adapter import ReadOnlyTableSpec
from api.database_gateway_factory import (
    APIDatabaseGatewayConfigurationError,
    create_api_business_database_gateway_factory,
)
from api.main import create_app
from api.resource_lifecycle import CapabilityExecutionResourceBundle
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy

DB_PERMISSION = "business:database:read"
API_PERMISSION = "capability:read:execute"
CAPABILITY_ID = "business_database.select"


class TrackingConnectionManager(ConnectionManager):
    def __init__(self):
        super().__init__()
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1
        super().shutdown()


class StaticProvider:
    def __init__(self, principal):
        self.principal = principal

    async def authenticate(self, request):
        return self.principal


class FakeEnvironment:
    def observe(self):
        return {"room": "isolated"}


class FakeExecutionRuntime:
    def __init__(self, environment):
        self.environment = environment
        self.lock = RLock()


class FakeRepository:
    def close(self):
        pass


def _create_database(path: Path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, total INTEGER)"
        )
        connection.executemany(
            "INSERT INTO orders (id, status, total) VALUES (?, ?, ?)",
            [(1, "paid", 150000), (2, "pending", 90000), (3, "paid", 225000)],
        )


def _table_specs():
    return {
        "orders": ReadOnlyTableSpec(
            table="orders",
            columns=("id", "status", "total"),
            filterable_columns=("status",),
            sortable_columns=("id", "total"),
            default_limit=20,
            max_limit=50,
        )
    }


def _configure_sqlite(manager, app, database_path):
    def read_only_guard(connection):
        connection.execute("PRAGMA query_only = ON")
        row = connection.execute("PRAGMA query_only").fetchone()
        if not row or row[0] != 1:
            raise PermissionError("SQLite read-only mode was not enabled.")

    manager.register(
        name="test_business_ro",
        database="sqlite",
        authorization_ref="test-approval:business-read-only",
        connection_factory=lambda: sqlite3.connect(database_path),
        read_only_guard=read_only_guard,
        read_only=True,
        enabled=True,
    )


def _make_factory(tmp_path, manager_holder):
    database_path = tmp_path / "business.sqlite3"
    _create_database(database_path)

    def manager_factory():
        manager = TrackingConnectionManager()
        manager_holder.append(manager)
        return manager

    return create_api_business_database_gateway_factory(
        profile_name="test_business_ro",
        profile_configurator=lambda manager, app: _configure_sqlite(
            manager, app, database_path
        ),
        table_specs=_table_specs(),
        trace_path=tmp_path / "business-capability-trace.jsonl",
        paramstyle="qmark",
        manager_factory=manager_factory,
        permission=DB_PERMISSION,
        fsync_trace=False,
    )


def _principal():
    return AuthenticatedPrincipal(
        actor_id="api-db-test-operator",
        permissions=(API_PERMISSION, DB_PERMISSION),
    )


def _create_test_app(tmp_path, monkeypatch, gateway_factory, principal):
    monkeypatch.setenv("HAI_AUTH_ENABLED", "true")
    monkeypatch.setenv("HAI_AUTH_REQUIRED", "false")
    monkeypatch.setenv("HAI_ENV", "development")
    environment = FakeEnvironment()
    app = create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: FakeExecutionRuntime(environment),
        capability_store_factory=lambda: object(),
        capability_execution_gateway_factory=gateway_factory,
        authentication_provider_factory=lambda app: StaticProvider(principal),
        authorization_policy_factory=lambda app: AuthorizationPolicy(),
        autonomous_testing_repository_factory=lambda path: FakeRepository(),
        autonomous_testing_manager_factory=lambda repository: object(),
    )
    return app


def test_factory_returns_api_owned_resource_bundle_without_opening_connection(tmp_path):
    database_path = tmp_path / "business.sqlite3"
    _create_database(database_path)
    managers = []

    def manager_factory():
        manager = TrackingConnectionManager()
        managers.append(manager)
        return manager

    factory = create_api_business_database_gateway_factory(
        profile_name="test_business_ro",
        profile_configurator=lambda manager, app: _configure_sqlite(
            manager, app, database_path
        ),
        table_specs=_table_specs(),
        trace_path=tmp_path / "trace.jsonl",
        paramstyle="qmark",
        manager_factory=manager_factory,
        permission=DB_PERMISSION,
        fsync_trace=False,
    )

    class App:
        state = type("State", (), {"authorization_policy": AuthorizationPolicy()})()

    bundle = factory(App())
    assert isinstance(bundle, CapabilityExecutionResourceBundle)
    assert bundle.gateway is not None
    assert bundle.owned_resources == (managers[0],)
    assert managers[0].active_connection_count == 0
    assert managers[0].shutdown_calls == 0
    managers[0].shutdown()


def test_api_http_read_uses_connection_manager_and_lifespan_closes_it(
    tmp_path, monkeypatch
):
    managers = []
    factory = _make_factory(tmp_path, managers)
    app = _create_test_app(tmp_path, monkeypatch, factory, _principal())

    with TestClient(app) as client:
        manager = managers[0]
        assert manager.active_connection_count == 0
        response = client.post(
            "/api/v1/capability-reads",
            json={
                "target": "business_database",
                "capability_id": CAPABILITY_ID,
                "arguments": {
                    "table": "orders",
                    "columns": ["id", "status", "total"],
                    "filters": {"status": "paid"},
                    "order_by": ["-total"],
                    "limit": 10,
                },
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"] == [
            {"id": 3, "status": "paid", "total": 225000},
            {"id": 1, "status": "paid", "total": 150000},
        ]
        assert response.json()["read_only"] is True
        assert manager.active_connection_count == 0
        trace_store = app.state.execution_trace_store
        assert trace_store is not None

    assert managers[0].shutdown_calls == 1
    assert managers[0].registered_profile_names == ()


def test_factory_shuts_manager_down_if_profile_configuration_fails(tmp_path):
    manager = TrackingConnectionManager()

    def broken_configurator(manager, app):
        raise RuntimeError("profile setup failed")

    factory = create_api_business_database_gateway_factory(
        profile_name="test_business_ro",
        profile_configurator=broken_configurator,
        table_specs=_table_specs(),
        trace_path=tmp_path / "trace.jsonl",
        paramstyle="qmark",
        manager_factory=lambda: manager,
        permission=DB_PERMISSION,
        fsync_trace=False,
    )

    class App:
        state = type("State", (), {"authorization_policy": AuthorizationPolicy()})()

    with pytest.raises(RuntimeError, match="profile setup failed"):
        factory(App())
    assert manager.shutdown_calls == 1
    assert manager.registered_profile_names == ()


def test_factory_fails_closed_without_application_authorization_policy(tmp_path):
    manager = TrackingConnectionManager()
    factory = create_api_business_database_gateway_factory(
        profile_name="test_business_ro",
        profile_configurator=lambda manager, app: None,
        table_specs=_table_specs(),
        trace_path=tmp_path / "trace.jsonl",
        paramstyle="qmark",
        manager_factory=lambda: manager,
        permission=DB_PERMISSION,
        fsync_trace=False,
    )

    class App:
        state = type("State", (), {"authorization_policy": None})()

    with pytest.raises(APIDatabaseGatewayConfigurationError):
        factory(App())
    assert manager.shutdown_calls == 1


def test_factory_rejects_empty_table_allowlist():
    with pytest.raises(APIDatabaseGatewayConfigurationError):
        create_api_business_database_gateway_factory(
            profile_name="test_business_ro",
            profile_configurator=lambda manager, app: None,
            table_specs={},
            trace_path="unused-trace.jsonl",
            paramstyle="qmark",
        )
