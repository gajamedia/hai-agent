from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.autonomous_testing.batch_manager import AutonomousTestBatchManager
from agent.autonomous_testing.result_repository import (
    ResultRepositoryError,
    SQLiteTestResultRepository,
)
from api.routes.autonomous_testing import router
from api.security.authentication_middleware import required_permission_for_route


def _setup(tmp_path, name="reconcile.sqlite3"):
    repository = SQLiteTestResultRepository(tmp_path / name)
    manager = AutonomousTestBatchManager(repository=repository)
    batch = manager.create_batch(count=2, seed=77, batch_id="reconcile-batch")
    repository.create_batch(batch)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.autonomous_test_result_repository = repository
    app.state.autonomous_test_batch_manager = manager
    app.state.autonomous_test_active_batches = set()
    app.state.autonomous_test_active_lock = Lock()
    return TestClient(app), repository, batch


def test_repository_reconciles_stale_batch_once_and_persists_audit(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "repository.sqlite3")
    try:
        batch = AutonomousTestBatchManager(repository=repository).create_batch(
            count=1, batch_id="repository-reconcile"
        )
        repository.create_batch(batch)
        stale = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        assert repository.claim_batch_running(batch.batch_id, stale, "dead-worker")
        result = repository.reconcile_stale_batch(
            batch.batch_id,
            stale_before=(datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
            reconciled_at=now,
            reconciled_by="operator-1",
            reason="Worker process terminated and heartbeat expired.",
        )
        assert result["status"] == "failed"
        assert result["replay_performed"] is False
        saved = repository.get_batch(batch.batch_id)
        assert saved["status"] == "failed"
        assert saved["worker_id"] == "dead-worker"
        events = repository.list_reconciliation_events(batch.batch_id)
        assert len(events) == 1
        assert events[0]["reconciled_by"] == "operator-1"
        assert events[0]["previous_worker_id"] == "dead-worker"
        with pytest.raises(ResultRepositoryError, match="not running"):
            repository.reconcile_stale_batch(
                batch.batch_id,
                stale_before=(datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
                reconciled_at=now,
                reconciled_by="operator-2",
                reason="Attempt duplicate reconciliation safely.",
            )
    finally:
        repository.close()


def test_repository_refuses_fresh_heartbeat(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "fresh.sqlite3")
    try:
        batch = AutonomousTestBatchManager(repository=repository).create_batch(
            count=1, batch_id="fresh-reconcile"
        )
        repository.create_batch(batch)
        now = datetime.now(timezone.utc).isoformat()
        assert repository.claim_batch_running(batch.batch_id, now, "live-worker")
        with pytest.raises(ResultRepositoryError, match="still fresh"):
            repository.reconcile_stale_batch(
                batch.batch_id,
                stale_before=(datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
                reconciled_at=now,
                reconciled_by="operator-1",
                reason="Trying to reconcile a live worker batch.",
            )
        assert repository.get_batch(batch.batch_id)["status"] == "running"
        assert repository.list_reconciliation_events(batch.batch_id) == []
    finally:
        repository.close()


def test_api_reconciliation_requires_explicit_no_replay_confirmation(tmp_path):
    client, repository, batch = _setup(tmp_path)
    try:
        stale = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        assert repository.claim_batch_running(batch.batch_id, stale, "dead-worker")
        response = client.post(
            f"/api/v1/autonomous-testing/batches/{batch.batch_id}/reconcile",
            json={"reason": "Worker process terminated unexpectedly."},
        )
        assert response.status_code == 422
        assert repository.get_batch(batch.batch_id)["status"] == "running"
    finally:
        repository.close()


def test_api_reconciliation_closes_stale_batch_without_replay(tmp_path):
    client, repository, batch = _setup(tmp_path)
    try:
        stale = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        assert repository.claim_batch_running(batch.batch_id, stale, "dead-worker")
        response = client.post(
            f"/api/v1/autonomous-testing/batches/{batch.batch_id}/reconcile",
            json={
                "reason": "Worker process terminated and heartbeat expired.",
                "confirm_no_replay": True,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "reconciled"
        assert body["replay_performed"] is False
        assert body["batch"]["status"] == "failed"
        assert body["batch"]["reconciled_by"] == "local-development"
        assert len(body["reconciliation_events"]) == 1
        detail = client.get(
            f"/api/v1/autonomous-testing/batches/{batch.batch_id}"
        ).json()
        assert detail["status"] == "failed"
        assert "worker_id" not in detail
        history = client.get(
            f"/api/v1/autonomous-testing/batches/{batch.batch_id}/reconciliation"
        )
        assert history.status_code == 200
        assert history.json()["count"] == 1
    finally:
        repository.close()


def test_api_reconciliation_refuses_fresh_or_already_finalized_batch(tmp_path):
    client, repository, batch = _setup(tmp_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        assert repository.claim_batch_running(batch.batch_id, now, "live-worker")
        response = client.post(
            f"/api/v1/autonomous-testing/batches/{batch.batch_id}/reconcile",
            json={
                "reason": "Attempt reconciliation while worker is active.",
                "confirm_no_replay": True,
            },
        )
        assert response.status_code == 409
        assert repository.get_batch(batch.batch_id)["status"] == "running"
    finally:
        repository.close()


def test_reconciliation_routes_have_explicit_security_permissions():
    assert required_permission_for_route(
        "POST", "/api/v1/autonomous-testing/batches/b-1/reconcile"
    ) == "autonomous-testing:batch:reconcile"
    assert required_permission_for_route(
        "GET", "/api/v1/autonomous-testing/batches/b-1/reconciliation"
    ) == "autonomous-testing:batch:read"


def test_keycloak_reconciler_role_grants_only_reconciliation_permission():
    from api.security.authorization_config import _role_map_for_keycloak

    mapping = _role_map_for_keycloak("hai-agent-api")
    permissions = mapping["client:hai-agent-api:autonomous-testing-reconciler"]
    assert permissions == ("autonomous-testing:batch:reconcile",)


def test_schema_v2_migrates_to_reconciliation_schema_v3(tmp_path):
    import sqlite3

    database = tmp_path / "legacy-v2.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_info(version INTEGER NOT NULL);
        INSERT INTO schema_info(version) VALUES (2);
        CREATE TABLE test_batches (
            batch_id TEXT PRIMARY KEY,
            seed INTEGER NOT NULL,
            difficulty TEXT NOT NULL,
            requested_cases INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            finished_at TEXT,
            worker_id TEXT,
            heartbeat_at TEXT,
            batch_json TEXT NOT NULL,
            result_json TEXT
        );
        CREATE TABLE test_case_results (
            batch_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            status TEXT NOT NULL,
            saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            result_json TEXT NOT NULL,
            PRIMARY KEY (batch_id, case_id)
        );
        """
    )
    connection.close()

    repository = SQLiteTestResultRepository(database)
    try:
        with repository._connect() as migrated:
            version = migrated.execute(
                "SELECT version FROM schema_info"
            ).fetchone()["version"]
            tables = {
                row["name"]
                for row in migrated.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert version == 3
        assert "batch_reconciliation_events" in tables
    finally:
        repository.close()
