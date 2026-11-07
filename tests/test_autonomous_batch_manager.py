from datetime import datetime, timezone

import pytest

from agent.autonomous_testing.batch_manager import (
    AutonomousTestBatchManager,
    BatchManagerError,
)
from agent.autonomous_testing.contracts import (
    BatchStatus,
    CaseStatus,
    Difficulty,
    TestCaseResult,
)
from agent.autonomous_testing.result_repository import SQLiteTestResultRepository


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run_case(self, case, *, recovery_policy, environment_config):
        self.calls.append(case.case_id)
        now = datetime.now(timezone.utc)
        if case.case_id.endswith("0002"):
            return TestCaseResult(
                case_id=case.case_id,
                status=CaseStatus.BLOCKED,
                started_at=now,
                finished_at=now,
                error="fixture blocked",
            )
        from agent.autonomous_testing.contracts import GoalEvaluation
        initial = dict(case.scenario.initial_state)
        final = dict(initial)
        final.update(case.scenario.goal)
        evaluation = GoalEvaluation(
            goal=case.scenario.goal,
            initial_state=initial,
            final_state=final,
            goal_satisfied=True,
        )
        return TestCaseResult(
            case_id=case.case_id,
            status=CaseStatus.PASSED,
            started_at=now,
            finished_at=now,
            evaluation=evaluation,
        )


class RaisingRunner:
    def run_case(self, case, **kwargs):
        raise RuntimeError("simulated runner crash")


def test_create_batch_is_deterministic_and_sequential():
    manager = AutonomousTestBatchManager(runner=FakeRunner())
    first = manager.create_batch(count=3, seed=17, batch_id="batch-a")
    second = manager.create_batch(count=3, seed=17, batch_id="batch-b")
    assert [c.scenario.to_dict() for c in first.cases] == [
        c.scenario.to_dict() for c in second.cases
    ]
    assert first.metadata["execution_mode"] == "sequential"
    assert first.metadata["learning_integration"] is False


def test_create_batch_rejects_parallel_config():
    from agent.autonomous_testing.contracts import TestEnvironmentConfig
    manager = AutonomousTestBatchManager(runner=FakeRunner())
    with pytest.raises(ValueError):
        manager.create_batch(
            count=1,
            environment_config=TestEnvironmentConfig(sequential_only=False),
        )


def test_run_batch_calls_runner_in_case_order_and_persists(tmp_path):
    runner = FakeRunner()
    repository = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    manager = AutonomousTestBatchManager(runner=runner, repository=repository)
    batch = manager.create_batch(count=3, seed=5, batch_id="batch-run")
    summary = manager.run_batch(batch)
    assert runner.calls == [case.case_id for case in batch.cases]
    assert summary.status == BatchStatus.COMPLETED
    assert summary.total == 3
    assert summary.counts["passed"] == 2
    assert summary.counts["blocked"] == 1
    stored = repository.get_batch("batch-run")
    assert stored["status"] == "completed"
    assert len(repository.list_batches()) == 1
    for case in batch.cases:
        repository.get_case_result("batch-run", case.case_id)


def test_run_without_persistence_is_explicit():
    runner = FakeRunner()
    manager = AutonomousTestBatchManager(runner=runner)
    batch = manager.create_batch(count=2, seed=4)
    summary = manager.run_batch(batch, persist=False)
    assert summary.total == 2
    assert len(runner.calls) == 2


def test_persist_requires_repository():
    manager = AutonomousTestBatchManager(runner=FakeRunner())
    batch = manager.create_batch(count=1)
    with pytest.raises(BatchManagerError):
        manager.run_batch(batch, persist=True)


def test_runner_exception_is_recorded_as_blocked_and_batch_continues(tmp_path):
    repository = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    manager = AutonomousTestBatchManager(
        runner=RaisingRunner(),
        repository=repository,
    )
    batch = manager.create_batch(count=3, batch_id="batch-errors")
    summary = manager.run_batch(batch)
    assert summary.total == 3
    assert summary.counts["blocked"] == 3
    assert summary.status == BatchStatus.COMPLETED


def test_duplicate_batch_id_does_not_run_cases(tmp_path):
    runner = FakeRunner()
    repository = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    manager = AutonomousTestBatchManager(runner=runner, repository=repository)
    batch = manager.create_batch(count=1, batch_id="duplicate")
    repository.create_batch(batch)
    from agent.autonomous_testing.result_repository import DuplicateBatchError
    with pytest.raises(DuplicateBatchError):
        manager.run_batch(batch)
    assert runner.calls == []


def test_fault_injection_cadence_is_passed_to_generator():
    manager = AutonomousTestBatchManager(runner=FakeRunner())
    batch = manager.create_batch(count=4, seed=2, inject_fault_every=2)
    assert batch.cases[1].scenario.fault_policy.enabled is True
    assert batch.cases[3].scenario.fault_policy.enabled is True
    assert batch.cases[0].scenario.fault_policy is None


def test_batch_summary_uses_independent_case_statuses():
    manager = AutonomousTestBatchManager(runner=FakeRunner())
    batch = manager.create_batch(count=3)
    summary = manager.run_batch(batch, persist=False)
    assert summary.counts["passed"] + summary.counts["blocked"] == 3
