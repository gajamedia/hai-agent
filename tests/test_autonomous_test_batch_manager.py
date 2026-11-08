from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.autonomous_testing.batch_manager import (
    AutonomousTestBatchManager,
    BatchShutdownRequested,
)
from api.resource_lifecycle import BackgroundTaskRegistry


class ShutdownOnFirstCaseRunner:
    def __init__(self):
        self.manager = None
        self.calls = []

    def run_case(self, test_case, *, recovery_policy=None, environment_config=None):
        self.calls.append(test_case.case_id)
        self.manager.request_shutdown()
        raise RuntimeError("case completed with an isolated runner error")


def test_batch_shutdown_stops_between_cases_and_releases_manager_lock():
    runner = ShutdownOnFirstCaseRunner()
    manager = AutonomousTestBatchManager(runner=runner)
    runner.manager = manager
    batch = manager.create_batch(count=4, seed=17)

    with pytest.raises(BatchShutdownRequested, match="stopped between cases"):
        manager.run_batch(batch, persist=False)

    assert len(runner.calls) == 1
    assert manager.shutdown_requested is True
    assert manager._running is False
    assert manager._lock.acquire(blocking=False) is True
    manager._lock.release()


def test_shutdown_requested_before_batch_prevents_first_case():
    runner = ShutdownOnFirstCaseRunner()
    manager = AutonomousTestBatchManager(runner=runner)
    runner.manager = manager
    manager.request_shutdown()
    batch = manager.create_batch(count=2, seed=19)

    with pytest.raises(BatchShutdownRequested):
        manager.run_batch(batch, persist=False)

    assert runner.calls == []
    assert manager._running is False


def test_background_registry_rejects_registration_after_shutdown():
    registry = BackgroundTaskRegistry()
    registry.register("batch-1")
    assert registry.active_task_ids == ("batch-1",)
    registry.stop_accepting()
    assert registry.accepting is False

    with pytest.raises(RuntimeError, match="shutting down"):
        registry.register("batch-2")

    registry.complete("batch-1")
    assert registry.wait_for_idle(timeout=0.1) is True


def test_background_registry_wait_timeout_and_completion():
    registry = BackgroundTaskRegistry()
    registry.register("batch-1")
    assert registry.wait_for_idle(timeout=0) is False
    registry.complete("batch-1")
    assert registry.wait_for_idle(timeout=0.1) is True


def test_background_registry_duplicate_registration_is_rejected():
    registry = BackgroundTaskRegistry()
    registry.register("batch-1")
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register("batch-1")
    registry.complete("batch-1")
