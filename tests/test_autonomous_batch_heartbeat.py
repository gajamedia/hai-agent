from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.autonomous_testing.batch_manager import AutonomousTestBatchManager
from agent.autonomous_testing.result_repository import SQLiteTestResultRepository
from api.routes.autonomous_testing import router


def _create_batch(repository, batch_id="heartbeat-batch"):
    manager = AutonomousTestBatchManager(repository=repository)
    batch = manager.create_batch(count=1, seed=11, batch_id=batch_id)
    repository.create_batch(batch)
    return batch


def _client(repository, manager):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.autonomous_test_result_repository = repository
    app.state.autonomous_test_batch_manager = manager
    app.state.autonomous_test_active_batches = set()
    app.state.autonomous_test_active_lock = Lock()
    return TestClient(app)


def test_schema_v1_is_migrated_to_durable_heartbeat_columns(tmp_path):
    import sqlite3

    database = tmp_path / "legacy-v1.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_info(version INTEGER NOT NULL);
        INSERT INTO schema_info(version) VALUES (1);
        CREATE TABLE test_batches (
            batch_id TEXT PRIMARY KEY,
            seed INTEGER NOT NULL,
            difficulty TEXT NOT NULL,
            requested_cases INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            finished_at TEXT,
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
            columns = {
                row["name"]
                for row in migrated.execute("PRAGMA table_info(test_batches)")
            }
            version = migrated.execute(
                "SELECT version FROM schema_info"
            ).fetchone()["version"]
        assert {"worker_id", "heartbeat_at"}.issubset(columns)
        assert version == 3
    finally:
        repository.close()


def test_heartbeat_is_accepted_only_for_current_owner(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "owner.sqlite3")
    try:
        batch = _create_batch(repository)
        now = datetime.now(timezone.utc).isoformat()
        assert repository.claim_batch_running(batch.batch_id, now, "worker-a")
        assert repository.heartbeat_batch(batch.batch_id, "worker-a", now)
        assert not repository.heartbeat_batch(batch.batch_id, "worker-b", now)
        saved = repository.get_batch(batch.batch_id)
        assert saved["worker_id"] == "worker-a"
        assert saved["heartbeat_at"] == now
    finally:
        repository.close()


def test_running_batch_with_fresh_heartbeat_is_not_reported_interrupted(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "fresh.sqlite3")
    try:
        batch = _create_batch(repository)
        now = datetime.now(timezone.utc).isoformat()
        assert repository.claim_batch_running(batch.batch_id, now, "worker-other-process")
        client = _client(repository, AutonomousTestBatchManager(repository=repository))
        response = client.get(f"/api/v1/autonomous-testing/batches/{batch.batch_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["worker_active"] is True
        assert data["interrupted_requires_reconciliation"] is False
        assert data["active_in_process"] is False
    finally:
        repository.close()


def test_running_batch_with_stale_heartbeat_requires_reconciliation(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "stale.sqlite3")
    try:
        batch = _create_batch(repository)
        old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        assert repository.claim_batch_running(batch.batch_id, old, "dead-worker")
        client = _client(repository, AutonomousTestBatchManager(repository=repository))
        response = client.get(f"/api/v1/autonomous-testing/batches/{batch.batch_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["worker_active"] is False
        assert data["interrupted_requires_reconciliation"] is True
        assert data["status"] == "running"
    finally:
        repository.close()


def test_running_batch_without_heartbeat_is_conservatively_stale(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "missing-heartbeat.sqlite3")
    try:
        batch = _create_batch(repository)
        with repository._connect() as connection:
            connection.execute(
                "UPDATE test_batches SET status = 'running' WHERE batch_id = ?",
                (batch.batch_id,),
            )
        client = _client(repository, AutonomousTestBatchManager(repository=repository))
        data = client.get(
            f"/api/v1/autonomous-testing/batches/{batch.batch_id}"
        ).json()
        assert data["worker_active"] is False
        assert data["interrupted_requires_reconciliation"] is True
    finally:
        repository.close()


def test_manager_does_not_finalize_batch_after_heartbeat_ownership_is_lost(tmp_path):
    from agent.autonomous_testing.batch_manager import BatchManagerError

    class LostHeartbeatRepository(SQLiteTestResultRepository):
        def heartbeat_batch(self, batch_id, worker_id, heartbeat_at):
            return False

    repository = LostHeartbeatRepository(tmp_path / "lost-owner.sqlite3")
    try:
        batch = _create_batch(repository, "lost-owner-batch")
        manager = AutonomousTestBatchManager(repository=repository)
        import pytest

        with pytest.raises(BatchManagerError, match="ownership was lost"):
            manager.run_batch(batch, persist=True, already_persisted=True)

        # The outcome is deliberately left unresolved; it is not rewritten to
        # FAILED because the current worker could not prove ownership.
        assert repository.get_batch(batch.batch_id)["status"] == "running"
    finally:
        repository.close()
