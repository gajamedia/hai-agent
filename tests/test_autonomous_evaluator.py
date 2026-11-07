from datetime import datetime, timezone

import pytest

from agent.autonomous_testing.contracts import CaseStatus, Difficulty
from agent.autonomous_testing.evaluator import (
    EvaluationInputError,
    IndependentEvaluator,
    evaluate_test_case,
)
from agent.autonomous_testing.scenario_generator import AutonomousScenarioGenerator


@pytest.fixture
def scenario():
    return AutonomousScenarioGenerator(seed=17).generate_scenario(0, Difficulty.MEDIUM)


def test_passed_requires_all_goal_fields_to_match(scenario):
    result = IndependentEvaluator().evaluate(
        scenario, {"door": "unlocked", "box": "open"}
    )
    assert result.status == CaseStatus.PASSED
    assert result.evaluation.goal_satisfied is True
    assert "observed_final_state_only" == result.metadata["decision_basis"]


def test_failed_when_observed_goal_field_mismatches(scenario):
    result = IndependentEvaluator().evaluate(scenario, {"door": "locked"})
    assert result.status == CaseStatus.FAILED
    assert result.evaluation.goal_satisfied is False
    assert "door" in result.evaluation.reason


def test_inconclusive_when_goal_field_is_missing(scenario):
    result = IndependentEvaluator().evaluate(scenario, {"box": "open"})
    assert result.status == CaseStatus.INCONCLUSIVE
    assert result.evaluation.goal_satisfied is None


def test_none_final_state_is_inconclusive(scenario):
    result = IndependentEvaluator().evaluate(scenario, None)
    assert result.status == CaseStatus.INCONCLUSIVE
    assert result.evaluation.final_state == {}


def test_partial_state_with_observed_mismatch_is_failed(scenario):
    result = IndependentEvaluator().evaluate(
        scenario, {"door": "locked", "box": "closed"}
    )
    assert result.status == CaseStatus.FAILED


def test_execution_error_alone_does_not_determine_goal_result(scenario):
    passed = IndependentEvaluator().evaluate(
        scenario,
        {"door": "unlocked"},
        error="transport warning after state observation",
    )
    assert passed.status == CaseStatus.PASSED

    inconclusive = IndependentEvaluator().evaluate(
        scenario,
        None,
        error="connection lost before final observation",
    )
    assert inconclusive.status == CaseStatus.INCONCLUSIVE


def test_uses_case_id_and_timestamps(scenario):
    now = datetime.now(timezone.utc)
    result = IndependentEvaluator().evaluate(
        scenario,
        {"door": "unlocked"},
        case_id="case-custom",
        started_at=now,
        finished_at=now,
        recovery_attempts=1,
    )
    assert result.case_id == "case-custom"
    assert result.started_at == now
    assert result.finished_at == now
    assert result.recovery_attempts == 1


@pytest.mark.parametrize("bad_state", [
    "not-a-mapping",
    {"unknown": "value"},
    {"door": "ajar"},
    {"door": None},
])
def test_rejects_malformed_observation(scenario, bad_state):
    with pytest.raises(EvaluationInputError):
        IndependentEvaluator().evaluate(scenario, bad_state)


def test_rejects_non_scenario_input():
    with pytest.raises(TypeError):
        IndependentEvaluator().evaluate(None, {"door": "unlocked"})


def test_functional_wrapper(scenario):
    result = evaluate_test_case(scenario, {"door": "unlocked"})
    assert result.status == CaseStatus.PASSED


def test_evaluator_does_not_mutate_input_state(scenario):
    observed = {"door": "unlocked"}
    before = dict(observed)
    IndependentEvaluator().evaluate(scenario, observed)
    assert observed == before


def test_result_does_not_depend_on_execution_success_claim(scenario):
    # Evaluator deliberately has no execution_success parameter.
    result = IndependentEvaluator().evaluate(scenario, {"door": "unlocked"})
    assert result.status == CaseStatus.PASSED
