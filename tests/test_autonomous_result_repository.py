from datetime import datetime, timezone

import pytest

from agent.autonomous_testing.contracts import (
    AutonomousTestBatch,
    AutonomousTestCase,
    BatchStatus,
    CaseStatus,
    Difficulty,
    GoalEvaluation,
    TestBatchResult,
    TestCaseResult,
)
from agent.autonomous_testing.result_repository import (
    DuplicateBatchError,
    ResultNotFoundError,
    SQLiteTestResultRepository,
)

INITIAL = {
    "door": "locked",
    "box": "closed",
    "key": "inside_box",
    "spare_key": "unavailable",
    "agent_location": "room",
}


def make_case(case_id="case-1"):
    from agent.autonomous_testing.contracts import ScenarioDefinition
    scenario = ScenarioDefinition(
        scenario_id=f"scenario-{case_id}",
        name="test scenario",
        difficulty=Difficulty.MEDIUM,
        seed=1,
        initial_state=INITIAL,
        goal={"door": "unlocked"},
        goal_description="Unlock door",
        metadata={"solution_path": ["open_box", "take_key", "unlock_door"]},
    )
    return AutonomousTestCase(case_id, scenario)


def make_batch(batch_id="batch-1"):
    case = make_case()
    return AutonomousTestBatch(
        batch_id=batch_id,
        seed=1,
        requested_cases=1,
        difficulty=Difficulty.MEDIUM,
        cases=(case,),
    )


def passed_result(case_id="case-1"):
    final_state = {**INITIAL, "door": "unlocked"}
    evaluation = GoalEvaluation(
        goal={"door": "unlocked"},
        initial_state=INITIAL,
        final_state=final_state,
        goal_satisfied=True,
    )
    now = datetime.now(timezone.utc)
    return TestCaseResult(
        case_id=case_id,
        status=CaseStatus.PASSED,
        started_at=now,
        finished_at=now,
        evaluation=evaluation,
    )


def test_create_batch_and_retrieve_definition(tmp_path):
    repo = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    batch = make_batch()
    repo.create_batch(batch)
    stored = repo.get_batch(batch.batch_id)
    assert stored["status"] == "pending"
    assert stored["batch"]["batch_id"] == batch.batch_id
    assert stored["batch"]["cases"][0]["case_id"] == "case-1"


def test_duplicate_batch_id_is_rejected(tmp_path):
    repo = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    batch = make_batch()
    repo.create_batch(batch)
    with pytest.raises(DuplicateBatchError):
        repo.create_batch(batch)


def test_save_and_read_case_result(tmp_path):
    repo = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    repo.create_batch(make_batch())
    repo.save_case_result("batch-1", passed_result())
    stored = repo.get_case_result("batch-1", "case-1")
    assert stored["status"] == "passed"
    assert stored["result"]["evaluation"]["goal_satisfied"] is True


def test_case_result_requires_existing_batch(tmp_path):
    repo = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    with pytest.raises(ResultNotFoundError):
        repo.save_case_result("missing", passed_result())


def test_complete_batch_persists_summary(tmp_path):
    repo = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    repo.create_batch(make_batch())
    result = TestBatchResult(
        batch_id="batch-1",
        status=BatchStatus.COMPLETED,
        results=(passed_result(),),
        seed=1,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    repo.complete_batch(result)
    stored = repo.get_batch("batch-1")
    assert stored["status"] == "completed"
    assert stored["result"]["counts"]["passed"] == 1


def test_list_batches_is_newest_first_and_limited(tmp_path):
    repo = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    repo.create_batch(make_batch("batch-a"))
    repo.create_batch(make_batch("batch-b"))
    rows = repo.list_batches(limit=1)
    assert len(rows) == 1
    assert rows[0]["batch_id"] in {"batch-a", "batch-b"}


def test_missing_batch_and_case_raise_clear_errors(tmp_path):
    repo = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    with pytest.raises(ResultNotFoundError):
        repo.get_batch("missing")
    repo.create_batch(make_batch())
    with pytest.raises(ResultNotFoundError):
        repo.get_case_result("batch-1", "missing")


def test_invalid_limit_rejected(tmp_path):
    repo = SQLiteTestResultRepository(tmp_path / "results.sqlite3")
    with pytest.raises(ValueError):
        repo.list_batches(0)


def test_repository_reopens_persisted_database(tmp_path):
    path = tmp_path / "results.sqlite3"
    SQLiteTestResultRepository(path).create_batch(make_batch())
    reopened = SQLiteTestResultRepository(path)
    assert reopened.get_batch("batch-1")["batch"]["seed"] == 1


def test_in_memory_repository_keeps_data_across_connections():
    repo = SQLiteTestResultRepository(":memory:")
    repo.create_batch(make_batch("memory-batch"))
    assert repo.get_batch("memory-batch")["batch"]["batch_id"] == "memory-batch"
