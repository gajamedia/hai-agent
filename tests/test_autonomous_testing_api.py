from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.autonomous_testing.batch_manager import AutonomousTestBatchManager
from agent.autonomous_testing.contracts import (
    CaseStatus,
    GoalEvaluation,
    TestCaseResult,
)
from agent.autonomous_testing.result_repository import SQLiteTestResultRepository
from api.routes.autonomous_testing import router


class PassingRunner:
    def run_case(self, test_case, *, recovery_policy=None, environment_config=None):
        now = datetime.now(timezone.utc)
        scenario = test_case.scenario
        evaluation = GoalEvaluation(
            goal=dict(scenario.goal),
            initial_state=dict(scenario.initial_state),
            final_state=dict(scenario.goal),
            goal_satisfied=True,
            evidence=("fake runner independently supplied final state",),
            reason="API test fixture",
        )
        return TestCaseResult(
            case_id=test_case.case_id,
            status=CaseStatus.PASSED,
            started_at=now,
            finished_at=now,
            evaluation=evaluation,
        )


def make_client(tmp_path):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    repository = SQLiteTestResultRepository(tmp_path / "api-results.sqlite3")
    manager = AutonomousTestBatchManager(runner=PassingRunner(), repository=repository)
    app.state.autonomous_test_result_repository = repository
    app.state.autonomous_test_batch_manager = manager
    app.state.autonomous_test_active_batches = set()
    app.state.autonomous_test_active_lock = Lock()
    return TestClient(app), repository


def test_create_batch_persists_definition_and_lists_it(tmp_path):
    client, repository = make_client(tmp_path)
    try:
        response = client.post("/api/v1/autonomous-testing/batches", json={"count": 3, "seed": 42})
        assert response.status_code == 201
        batch_id = response.json()["batch_id"]
        detail = client.get(f"/api/v1/autonomous-testing/batches/{batch_id}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "pending"
        assert detail.json()["batch"]["requested_cases"] == 3
        listed = client.get("/api/v1/autonomous-testing/batches")
        assert listed.status_code == 200
        assert listed.json()["count"] == 1
    finally:
        repository.close()


def test_run_batch_is_background_accepted_and_results_are_queryable(tmp_path):
    client, repository = make_client(tmp_path)
    try:
        created = client.post("/api/v1/autonomous-testing/batches", json={"count": 2, "seed": 9})
        batch_id = created.json()["batch_id"]
        response = client.post(f"/api/v1/autonomous-testing/batches/{batch_id}/run")
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"

        detail = client.get(f"/api/v1/autonomous-testing/batches/{batch_id}").json()
        assert detail["status"] == "completed"
        assert detail["interrupted_requires_reconciliation"] is False
        results = client.get(f"/api/v1/autonomous-testing/batches/{batch_id}/results")
        assert results.status_code == 200
        assert results.json()["count"] == 2
        assert all(item["status"] == "passed" for item in results.json()["items"])

        repeated = client.post(f"/api/v1/autonomous-testing/batches/{batch_id}/run")
        assert repeated.status_code == 409
    finally:
        repository.close()


def test_run_can_reconstruct_pending_batch_after_api_process_restart(tmp_path):
    database = tmp_path / "restart.sqlite3"
    app1 = FastAPI()
    app1.include_router(router, prefix="/api/v1")
    repo1 = SQLiteTestResultRepository(database)
    manager1 = AutonomousTestBatchManager(runner=PassingRunner(), repository=repo1)
    app1.state.autonomous_test_result_repository = repo1
    app1.state.autonomous_test_batch_manager = manager1
    app1.state.autonomous_test_active_batches = set()
    app1.state.autonomous_test_active_lock = Lock()
    with TestClient(app1) as client1:
        created = client1.post("/api/v1/autonomous-testing/batches", json={"count": 1, "batch_id": "restart-batch"})
        assert created.status_code == 201
    repo1.close()

    app2 = FastAPI()
    app2.include_router(router, prefix="/api/v1")
    repo2 = SQLiteTestResultRepository(database)
    manager2 = AutonomousTestBatchManager(runner=PassingRunner(), repository=repo2)
    app2.state.autonomous_test_result_repository = repo2
    app2.state.autonomous_test_batch_manager = manager2
    app2.state.autonomous_test_active_batches = set()
    app2.state.autonomous_test_active_lock = Lock()
    client2 = TestClient(app2)
    try:
        response = client2.post("/api/v1/autonomous-testing/batches/restart-batch/run")
        assert response.status_code == 202
        detail = client2.get("/api/v1/autonomous-testing/batches/restart-batch")
        assert detail.json()["status"] == "completed"
    finally:
        repo2.close()


def test_create_rejects_parallel_or_invalid_requests(tmp_path):
    client, repository = make_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/autonomous-testing/batches",
            json={"count": 2, "environment_config": {"sequential_only": False}},
        )
        assert response.status_code == 422
        response = client.post("/api/v1/autonomous-testing/batches", json={"count": 1001})
        assert response.status_code == 422
        response = client.post("/api/v1/autonomous-testing/batches", json={"unexpected": True})
        assert response.status_code == 422
    finally:
        repository.close()


def test_results_for_unknown_batch_returns_404(tmp_path):
    client, repository = make_client(tmp_path)
    try:
        response = client.get("/api/v1/autonomous-testing/batches/missing/results")
        assert response.status_code == 404
    finally:
        repository.close()
