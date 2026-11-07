from __future__ import annotations

from datetime import datetime, timezone
from threading import Event, Thread
import time
import pytest as __pytest

from agent.autonomous_testing.batch_manager import (
    AutonomousTestBatchManager,
    BatchShutdownRequested,
)
from agent.autonomous_testing.contracts import CaseStatus, TestCaseResult
from agent.autonomous_testing.result_repository import SQLiteTestResultRepository


class RecordingRunner:
    def __init__(self, *, started: Event | None = None, release: Event | None = None):
        self.started = started
        self.release = release
        self.calls: list[str] = []

    def run_case(self, test_case, *, recovery_policy, environment_config):
        self.calls.append(test_case.case_id)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=5), "test did not release active case"
        now = datetime.now(timezone.utc)
        return TestCaseResult(
            case_id=test_case.case_id,
            status=CaseStatus.PASSED,
            started_at=now,
            finished_at=now,
            metadata={"test": "cooperative-shutdown"},
        )


def _stored(repository, batch_id):
    return repository.get_batch(batch_id)


def test_shutdown_requested_before_first_case_leaves_batch_running(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "shutdown-before.sqlite3")
    runner = RecordingRunner()
    manager = AutonomousTestBatchManager(
        runner=runner, repository=repository, heartbeat_interval_seconds=0.02
    )
    batch = manager.create_batch(count=3, seed=11, batch_id="shutdown-before")
    manager.request_shutdown()
    try:
        result = manager.run_batch(batch)
        assert result.status.value == "running"
        assert result.metadata["shutdown_stopped"] is True
        assert result.metadata["requires_reconciliation"] is True
        assert runner.calls == []
        stored = _stored(repository, batch.batch_id)
        assert stored["status"] == "running"
        assert repository.list_case_results(batch.batch_id) == []
    finally:
        repository.close()


def test_shutdown_during_case_finishes_and_persists_only_active_case(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "shutdown-during.sqlite3")
    started = Event()
    release = Event()
    runner = RecordingRunner(started=started, release=release)
    manager = AutonomousTestBatchManager(
        runner=runner, repository=repository, heartbeat_interval_seconds=0.02
    )
    batch = manager.create_batch(count=3, seed=12, batch_id="shutdown-during")
    outcomes = []
    errors = []

    def execute():
        try:
            outcomes.append(manager.run_batch(batch))
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    worker = Thread(target=execute, name="test-batch-worker")
    worker.start()
    try:
        assert started.wait(timeout=5), "first case did not start"
        manager.request_shutdown()
        release.set()
        worker.join(timeout=5)
        assert not worker.is_alive(), "batch manager did not stop at case boundary"
        assert errors == []
        assert len(outcomes) == 1
        result = outcomes[0]
        assert result.status.value == "running"
        assert result.metadata["shutdown_stopped"] is True
        assert len(result.results) == 1
        assert runner.calls == [batch.cases[0].case_id]

        stored = _stored(repository, batch.batch_id)
        assert stored["status"] == "running"
        saved_results = repository.list_case_results(batch.batch_id)
        assert len(saved_results) == 1
        assert saved_results[0]["case_id"] == batch.cases[0].case_id

        heartbeat_at_stop = stored["heartbeat_at"]
        time.sleep(0.08)
        stored_after_wait = _stored(repository, batch.batch_id)
        assert stored_after_wait["heartbeat_at"] == heartbeat_at_stop
    finally:
        release.set()
        worker.join(timeout=5)
        repository.close()


def test_shutdown_is_sticky_and_does_not_allow_manager_reuse(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "shutdown-sticky.sqlite3")
    runner = RecordingRunner()
    manager = AutonomousTestBatchManager(
        runner=runner, repository=repository, heartbeat_interval_seconds=0.02
    )
    manager.request_shutdown()
    first = manager.create_batch(count=1, seed=13, batch_id="shutdown-sticky-1")
    second = manager.create_batch(count=1, seed=14, batch_id="shutdown-sticky-2")
    try:
        first_result = manager.run_batch(first)
        second_result = manager.run_batch(second)
        assert first_result.status.value == "running"
        assert second_result.status.value == "running"
        assert runner.calls == []
        assert _stored(repository, first.batch_id)["status"] == "running"
        assert _stored(repository, second.batch_id)["status"] == "running"
    finally:
        repository.close()


class ShutdownAndRaiseRunner:
    def __init__(self):
        self.manager = None
        self.calls = []

    def run_case(self, test_case, *, recovery_policy, environment_config):
        self.calls.append(test_case.case_id)
        self.manager.request_shutdown()
        raise RuntimeError("isolated runner error during shutdown")


def test_non_persisted_shutdown_raises_legacy_exception_and_releases_lock():
    runner = ShutdownAndRaiseRunner()
    manager = AutonomousTestBatchManager(runner=runner)
    runner.manager = manager
    batch = manager.create_batch(count=4, seed=17)

    with __pytest.raises(BatchShutdownRequested, match="stopped between cases"):
        manager.run_batch(batch, persist=False)

    assert len(runner.calls) == 1
    assert manager.shutdown_requested is True
    assert manager._running is False
    assert manager._lock.acquire(blocking=False) is True
    manager._lock.release()


def test_non_persisted_shutdown_before_first_case_raises():
    runner = RecordingRunner()
    manager = AutonomousTestBatchManager(runner=runner)
    manager.request_shutdown()
    batch = manager.create_batch(count=2, seed=19)

    with __pytest.raises(BatchShutdownRequested):
        manager.run_batch(batch, persist=False)

    assert runner.calls == []
    assert manager._running is False
