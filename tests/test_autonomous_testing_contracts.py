from datetime import datetime, timedelta, timezone
import pytest
from agent.autonomous_testing.contracts import *

INITIAL = {"door":"locked","box":"closed","key":"inside_box","spare_key":"unavailable","agent_location":"room"}

def scenario(**kw):
    data = dict(scenario_id="s1", name="unlock", difficulty=Difficulty.MEDIUM, seed=7,
                initial_state=INITIAL, goal={"door":"unlocked"}, goal_description="Unlock door")
    data.update(kw)
    return ScenarioDefinition(**data)

def test_valid_scenario_and_immutable_nested_data():
    source = dict(INITIAL)
    s = scenario(initial_state=source, metadata={"labels":["x"]})
    source["door"] = "unlocked"
    assert s.initial_state["door"] == "locked"
    assert s.metadata["labels"] == ("x",)
    with pytest.raises(TypeError): s.initial_state["door"] = "unlocked"

@pytest.mark.parametrize("state", [{"door":"locked"}, {**INITIAL,"x":"y"}, {**INITIAL,"door":"ajar"}])
def test_invalid_initial_state_rejected(state):
    with pytest.raises((ValueError, TypeError)): scenario(initial_state=state)

@pytest.mark.parametrize("goal", [{}, {"x":"y"}, {"door":"ajar"}])
def test_invalid_goal_rejected(goal):
    with pytest.raises((ValueError, TypeError)): scenario(goal=goal)

def test_invalid_seed_and_difficulty_rejected():
    with pytest.raises(ValueError): scenario(seed=-1)
    with pytest.raises(ValueError): scenario(difficulty="expert")

def test_fault_policy_requires_fault_type_when_enabled():
    with pytest.raises(ValueError): FaultInjectionPolicy(enabled=True)
    p = FaultInjectionPolicy(enabled=True, fault_type="action_failure", trigger_step=2)
    assert p.to_dict()["fault_type"] == "action_failure"

def test_recovery_policy_and_environment_defaults():
    assert RecoveryPolicy(enabled=False).max_attempts == 0
    cfg = TestEnvironmentConfig()
    assert cfg.cognitive_runtime_mode == CognitiveRuntimeMode.ISOLATED
    assert cfg.sequential_only and cfg.simulation_memory_mode == "in_memory"

def test_batch_validates_count_and_unique_case_ids():
    c = AutonomousTestCase("c1", scenario())
    with pytest.raises(ValueError): AutonomousTestBatch("b1", 7, 2, Difficulty.MEDIUM, (c,))
    with pytest.raises(ValueError): AutonomousTestBatch("b1", 7, 2, Difficulty.MEDIUM, (c,c))

def test_batch_serialization_is_deterministic():
    cases = (AutonomousTestCase("c1",scenario()), AutonomousTestCase("c2",scenario(scenario_id="s2",seed=8)))
    b = AutonomousTestBatch("b1",7,2,Difficulty.MEDIUM,cases)
    assert canonical_json(b) == canonical_json(b.to_dict())
    assert b.to_dict()["status"] == "pending"

def test_goal_evaluation_uses_observed_state_not_claimed_success():
    good = GoalEvaluation({"door":"unlocked"}, INITIAL, {**INITIAL,"door":"unlocked"}, True)
    assert good.goal_satisfied is True
    with pytest.raises(ValueError):
        GoalEvaluation({"door":"unlocked"}, INITIAL, INITIAL, True)
    bad = GoalEvaluation({"door":"unlocked"}, INITIAL, INITIAL, False)
    assert bad.goal_satisfied is False

def test_goal_evaluation_supports_incomplete_observation_as_unknown():
    ev = GoalEvaluation({"door":"unlocked"}, INITIAL, {}, None)
    assert ev.goal_satisfied is None

def test_pass_requires_verified_goal_success():
    with pytest.raises(ValueError): TestCaseResult("c1",CaseStatus.PASSED)
    ev = GoalEvaluation({"door":"unlocked"},INITIAL,{**INITIAL,"door":"unlocked"},True)
    assert TestCaseResult("c1",CaseStatus.PASSED,evaluation=ev).status == CaseStatus.PASSED

def test_inconclusive_cannot_hide_known_failure():
    ev = GoalEvaluation({"door":"unlocked"},INITIAL,INITIAL,False)
    with pytest.raises(ValueError): TestCaseResult("c1",CaseStatus.INCONCLUSIVE,evaluation=ev)

def test_timestamps_must_be_aware_and_ordered():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError): TestCaseResult("c1",CaseStatus.FAILED,now,now-timedelta(seconds=1))
    with pytest.raises(ValueError): TestCaseResult("c1",CaseStatus.FAILED,datetime.now())

def test_batch_result_counts_are_derived():
    ev = GoalEvaluation({"door":"unlocked"},INITIAL,{**INITIAL,"door":"unlocked"},True)
    results = (TestCaseResult("c1",CaseStatus.PASSED,evaluation=ev),
               TestCaseResult("c2",CaseStatus.BLOCKED,error="Policy blocked"),
               TestCaseResult("c3",CaseStatus.INCONCLUSIVE))
    summary = TestBatchResult("b1",BatchStatus.COMPLETED,results,7)
    assert summary.total == 3 and summary.counts["passed"] == 1
    assert summary.counts["blocked"] == 1 and summary.counts["inconclusive"] == 1

def test_batch_result_rejects_duplicate_ids():
    r = TestCaseResult("c1",CaseStatus.FAILED)
    with pytest.raises(ValueError): TestBatchResult("b1",BatchStatus.FAILED,(r,r),7)
