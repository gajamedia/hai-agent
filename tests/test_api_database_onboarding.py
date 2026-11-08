import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.database import ConnectionManager, register_sqlite_read_only_profile
from api.routes.database_onboarding import router


def configured_app(tmp_path):
    path = tmp_path / "reporting.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, amount INTEGER)")
    manager = ConnectionManager()
    register_sqlite_read_only_profile(
        manager,
        profile_name="reporting_ro",
        database_path=path,
        authorization_ref="user-approval:onboarding-001",
    )
    app = FastAPI()
    app.state.business_database_connection_manager = manager
    app.include_router(router, prefix="/api/v1")
    return app, manager


def test_schema_endpoint_only_uses_configured_profile(tmp_path):
    app, manager = configured_app(tmp_path)
    try:
        response = TestClient(app).get(
            "/api/v1/database-onboarding/schema", params={"profile_name": "reporting_ro"}
        )
        assert response.status_code == 200
        assert response.json()["profile_name"] == "reporting_ro"
        assert response.json()["tables"][0]["name"] == "sales"
    finally:
        manager.shutdown()


def test_schema_endpoint_rejects_unknown_profile(tmp_path):
    app, manager = configured_app(tmp_path)
    try:
        response = TestClient(app).get(
            "/api/v1/database-onboarding/schema", params={"profile_name": "not-configured"}
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "sqlite_schema_discovery_failed"
    finally:
        manager.shutdown()


def test_allowlist_endpoint_only_persists_discovered_tables_and_columns(tmp_path):
    app, manager = configured_app(tmp_path)
    try:
        response = TestClient(app).post(
            "/api/v1/database-onboarding/allowlists",
            json={"profile_name": "reporting_ro", "tables": {"sales": ["id", "amount"]}},
        )
        assert response.status_code == 200, response.text
        assert response.json()["tables"] == {"sales": ["id", "amount"]}
        assert response.json()["activation"] == "pending_controlled_restart"
        assert app.state.database_onboarding_allowlists["reporting_ro"]["tables"] == {
            "sales": ["id", "amount"]
        }
    finally:
        manager.shutdown()


def test_allowlist_endpoint_rejects_undiscovered_column(tmp_path):
    app, manager = configured_app(tmp_path)
    try:
        response = TestClient(app).post(
            "/api/v1/database-onboarding/allowlists",
            json={"profile_name": "reporting_ro", "tables": {"sales": ["secret"]}},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_database_allowlist"
    finally:
        manager.shutdown()
