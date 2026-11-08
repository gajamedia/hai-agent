from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agent.autonomous_testing.batch_manager import (
    AutonomousTestBatchManager,
    BatchManagerError,
)
from agent.autonomous_testing.result_repository import SQLiteTestResultRepository


def _create_pending_batch(repository: SQLiteTestResultRepository, batch_id: str):
    manager = AutonomousTestBatchManager(repository=repository)
    batch = manager.create_batch(count=2, seed=17, batch_id=batch_id)
    repository.create_batch(batch)
    return batch


def test_only_one_repository_instance_can_claim_a_pending_batch(tmp_path):
    database = tmp_path / "atomic-claim.sqlite3"
    creator = SQLiteTestResultRepository(database)
    worker_repositories = [SQLiteTestResultRepository(database) for _ in range(8)]
    try:
        _create_pending_batch(creator, "atomic-batch")
        barrier = Barrier(len(worker_repositories))

        def claim(repository):
            barrier.wait(timeout=5)
            return repository.claim_batch_running(
                "atomic-batch", "2026-11-03T00:00:00+00:00"
            )

        with ThreadPoolExecutor(max_workers=len(worker_repositories)) as pool:
            outcomes = list(pool.map(claim, worker_repositories))

        assert outcomes.count(True) == 1
        assert outcomes.count(False) == len(worker_repositories) - 1
        assert creator.get_batch("atomic-batch")["status"] == "running"
    finally:
        for repository in worker_repositories:
            repository.close()
        creator.close()


def test_claim_returns_false_for_non_pending_batch(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "non-pending.sqlite3")
    try:
        _create_pending_batch(repository, "once-only")
        assert repository.claim_batch_running(
            "once-only", "2026-11-03T00:00:00+00:00"
        ) is True
        assert repository.claim_batch_running(
            "once-only", "2026-11-03T00:00:01+00:00"
        ) is False
    finally:
        repository.close()


def test_claim_returns_false_for_missing_batch(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "missing.sqlite3")
    try:
        assert repository.claim_batch_running(
            "does-not-exist", "2026-11-03T00:00:00+00:00"
        ) is False
    finally:
        repository.close()


def test_manager_does_not_overwrite_another_workers_claim(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "manager-claim.sqlite3")
    try:
        batch = _create_pending_batch(repository, "claimed-elsewhere")
        assert repository.claim_batch_running(
            batch.batch_id, "2026-11-03T00:00:00+00:00"
        ) is True

        manager = AutonomousTestBatchManager(repository=repository)
        with pytest.raises(BatchManagerError, match="already been claimed"):
            manager.run_batch(batch, persist=True, already_persisted=True)

        # A losing worker must not mark the winning worker's batch as failed.
        assert repository.get_batch(batch.batch_id)["status"] == "running"
    finally:
        repository.close()


def test_manager_accepts_a_claim_already_acquired_by_api(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "preclaimed.sqlite3")
    try:
        batch = _create_pending_batch(repository, "preclaimed-batch")
        owner_id = "api-owner-token"
        assert repository.claim_batch_running(
            batch.batch_id, "2026-11-03T00:00:00+00:00", owner_id
        ) is True

        class PassingRunner:
            def run_case(self, test_case, *, recovery_policy=None, environment_config=None):
                from datetime import datetime, timezone
                from agent.autonomous_testing.contracts import (
                    CaseStatus,
                    GoalEvaluation,
                    TestCaseResult,
                )

                now = datetime.now(timezone.utc)
                scenario = test_case.scenario
                return TestCaseResult(
                    case_id=test_case.case_id,
                    status=CaseStatus.PASSED,
                    started_at=now,
                    finished_at=now,
                    evaluation=GoalEvaluation(
                        goal=dict(scenario.goal),
                        initial_state=dict(scenario.initial_state),
                        final_state=dict(scenario.goal),
                        goal_satisfied=True,
                        evidence=("test runner",),
                        reason="test",
                    ),
                )

        manager = AutonomousTestBatchManager(runner=PassingRunner(), repository=repository)
        result = manager.run_batch(
            batch,
            persist=True,
            already_persisted=True,
            already_claimed=True,
            execution_owner_id=owner_id,
        )
        assert result.status.value == "completed"
        assert repository.get_batch(batch.batch_id)["status"] == "completed"
    finally:
        repository.close()
