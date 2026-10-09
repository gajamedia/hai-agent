from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from agent.llm.goal_utility_evaluation import (
    GoalUtilityEvaluationConfigurationError,
    GoalUtilityEvaluationInputError,
    GoalUtilityEvaluator,
    GoalUtilityInput,
    GoalUtilityWeights,
)


def make_goal(**overrides):
    data = {
        "goal_id": "g1",
        "description": "Improve target",
        "value": 0.8,
        "urgency": 0.6,
        "importance": 0.7,
        "progress": 0.2,
        "estimated_effort": 0.2,
        "risk": 0.1,
        "opportunity_cost": 0.1,
        "confidence": 0.9,
        "evidence_quality": 0.8,
        "metadata": {"source": "test"},
    }
    data.update(overrides)
    return GoalUtilityInput(**data)


def test_default_weights_total_is_one():
    assert GoalUtilityWeights().total == pytest.approx(1.0)


def test_weights_accept_nonnegative_bounded_values():
    weights = GoalUtilityWeights(value=1, urgency=0, importance=0, progress_need=0, effort_cost=0, risk_cost=0, opportunity_cost=0)
    assert weights.total == 1


@pytest.mark.parametrize("field", [
    "value", "urgency", "importance", "progress", "estimated_effort",
    "risk", "opportunity_cost", "confidence", "evidence_quality"
])
def test_goal_input_rejects_out_of_range(field):
    values = make_goal().__dict__.copy()
    values[field] = 1.1
    with pytest.raises(GoalUtilityEvaluationInputError):
        GoalUtilityInput(**values)


def test_goal_input_rejects_empty_id():
    with pytest.raises(GoalUtilityEvaluationInputError):
        make_goal(goal_id="   ")


def test_goal_input_rejects_non_mapping_metadata():
    with pytest.raises(GoalUtilityEvaluationInputError):
        make_goal(metadata=[])


def test_weights_reject_negative():
    with pytest.raises(GoalUtilityEvaluationConfigurationError):
        GoalUtilityWeights(value=-0.1)


def test_weights_reject_above_one():
    with pytest.raises(GoalUtilityEvaluationConfigurationError):
        GoalUtilityWeights(value=1.1)


def test_weights_require_positive_total():
    with pytest.raises(GoalUtilityEvaluationConfigurationError):
        GoalUtilityWeights(value=0, urgency=0, importance=0, progress_need=0, effort_cost=0, risk_cost=0, opportunity_cost=0)


@pytest.mark.parametrize("field", ["recommendation_threshold", "minimum_confidence"])
def test_evaluator_rejects_bad_config(field):
    kwargs = {field: 1.1}
    with pytest.raises(GoalUtilityEvaluationConfigurationError):
        GoalUtilityEvaluator(**kwargs)


def test_progress_need_is_inverse_progress():
    assert make_goal(progress=0.25).progress_need == pytest.approx(0.75)


def test_evaluator_returns_expected_fields():
    result = GoalUtilityEvaluator().evaluate(make_goal())
    assert result.goal_id == "g1"
    assert result.description == "Improve target"
    assert 0 <= result.benefit_score <= 1
    assert 0 <= result.cost_score <= 1
    assert -1 <= result.expected_utility <= 1
    assert 0 <= result.utility_score <= 1


def test_evaluator_preserves_metadata_copy():
    goal = make_goal()
    result = GoalUtilityEvaluator().evaluate(goal)
    assert result.metadata == {"source": "test"}
    assert result.metadata is not goal.metadata


def test_evaluator_is_side_effect_free():
    goal = make_goal()
    before = goal.__dict__.copy()
    GoalUtilityEvaluator().evaluate(goal)
    assert goal.__dict__ == before


def test_high_value_low_cost_is_recommended():
    result = GoalUtilityEvaluator().evaluate(
        make_goal(value=1, urgency=1, importance=1, progress=0, estimated_effort=0, risk=0, opportunity_cost=0, confidence=1, evidence_quality=1)
    )
    assert result.recommended is True
    assert result.recommendation == "prioritize"


def test_low_value_high_cost_is_deferred():
    result = GoalUtilityEvaluator().evaluate(
        make_goal(value=0, urgency=0, importance=0, progress=1, estimated_effort=1, risk=1, opportunity_cost=1)
    )
    assert result.recommended is False
    assert result.recommendation == "defer"


def test_confidence_gate_can_force_insufficient_confidence():
    result = GoalUtilityEvaluator(minimum_confidence=0.8).evaluate(make_goal(confidence=0.7))
    assert result.recommendation == "insufficient_confidence"
    assert result.recommended is False


def test_confidence_adjustment_reduces_positive_utility():
    high = GoalUtilityEvaluator().evaluate(make_goal(confidence=1, evidence_quality=1))
    low = GoalUtilityEvaluator().evaluate(make_goal(confidence=0.5, evidence_quality=0.5))
    assert low.confidence_adjusted_utility <= high.confidence_adjusted_utility


def test_components_include_all_major_dimensions():
    result = GoalUtilityEvaluator().evaluate(make_goal())
    expected = {
        "value", "urgency", "importance", "progress_need", "estimated_effort",
        "risk", "opportunity_cost", "confidence", "evidence_quality"
    }
    assert expected == set(result.components)


def test_custom_weights_change_evaluation():
    default = GoalUtilityEvaluator().evaluate(make_goal(value=1, estimated_effort=1))
    custom = GoalUtilityEvaluator(
        weights=GoalUtilityWeights(
            value=0.0, urgency=0.0, importance=0.0, progress_need=0.0,
            effort_cost=1.0, risk_cost=0.0, opportunity_cost=0.0,
        )
    ).evaluate(make_goal(value=1, estimated_effort=1))
    assert default.expected_utility != custom.expected_utility


def test_zero_benefit_weight_is_supported():
    result = GoalUtilityEvaluator(
        weights=GoalUtilityWeights(value=0, urgency=0, importance=0, progress_need=0, effort_cost=1, risk_cost=0, opportunity_cost=0)
    ).evaluate(make_goal(estimated_effort=0))
    assert result.benefit_score == 0
    assert result.cost_score == 0


def test_zero_cost_weight_is_supported():
    result = GoalUtilityEvaluator(
        weights=GoalUtilityWeights(value=1, urgency=0, importance=0, progress_need=0, effort_cost=0, risk_cost=0, opportunity_cost=0)
    ).evaluate(make_goal(value=1))
    assert result.cost_score == 0


def test_mapping_input_is_supported():
    result = GoalUtilityEvaluator().evaluate(make_goal().__dict__)
    assert result.goal_id == "g1"


def test_object_input_is_supported():
    obj = SimpleNamespace(**make_goal().__dict__)
    result = GoalUtilityEvaluator().evaluate(obj)
    assert result.goal_id == "g1"


def test_id_alias_is_supported():
    result = GoalUtilityEvaluator().evaluate({"id": "g9", "description": "x"})
    assert result.goal_id == "g9"


def test_goal_alias_is_supported():
    result = GoalUtilityEvaluator().evaluate({"id": "g9", "goal": "x"})
    assert result.description == "x"


def test_expected_value_alias_is_supported():
    result = GoalUtilityEvaluator().evaluate({"id": "g9", "expected_value": 1})
    assert result.components["value"] == 1


def test_priority_score_alias_is_supported():
    result = GoalUtilityEvaluator().evaluate({"id": "g9", "priority_score": 1})
    assert result.components["importance"] == 1


def test_measured_progress_alias_is_supported():
    result = GoalUtilityEvaluator().evaluate({"id": "g9", "measured_progress": 0.4})
    assert result.components["progress_need"] == pytest.approx(0.6)


def test_effort_alias_is_supported():
    result = GoalUtilityEvaluator().evaluate({"id": "g9", "effort": 0.4})
    assert result.components["estimated_effort"] == 0.4


def test_risk_score_alias_is_supported():
    result = GoalUtilityEvaluator().evaluate({"id": "g9", "risk_score": 0.4})
    assert result.components["risk"] == 0.4


def test_many_sorts_by_confidence_adjusted_utility():
    goals = [
        make_goal(goal_id="a", value=0.2, risk=0.8, confidence=0.5),
        make_goal(goal_id="b", value=1.0, risk=0.0, confidence=1.0),
        make_goal(goal_id="c", value=0.5, risk=0.1, confidence=0.9),
    ]
    results = GoalUtilityEvaluator().evaluate_many(goals)
    assert [item.goal_id for item in results] == ["b", "c", "a"]


def test_many_accepts_empty_list():
    assert GoalUtilityEvaluator().evaluate_many([]) == []


def test_recommendation_threshold_is_respected():
    evaluator = GoalUtilityEvaluator(recommendation_threshold=0.99)
    result = evaluator.evaluate(make_goal(value=1, urgency=1, importance=1, progress=0, estimated_effort=0, risk=0, opportunity_cost=0, confidence=1, evidence_quality=1))
    assert result.recommended is True


def test_neutral_utility_can_be_deferred():
    result = GoalUtilityEvaluator(recommendation_threshold=0.6).evaluate(
        make_goal(value=0.5, urgency=0.5, importance=0.5, progress=0.5, estimated_effort=0.5, risk=0.5, opportunity_cost=0.5)
    )
    assert result.expected_utility == pytest.approx(0, abs=1e-12)
    assert result.recommended is False


def test_priority_property_matches_recommendation():
    result = GoalUtilityEvaluator().evaluate(make_goal())
    assert result.should_prioritize == result.recommended


def test_net_utility_alias_matches_expected_utility():
    result = GoalUtilityEvaluator().evaluate(make_goal())
    assert result.net_utility == result.expected_utility


def test_evaluation_is_frozen():
    result = GoalUtilityEvaluator().evaluate(make_goal())
    with pytest.raises(FrozenInstanceError):
        result.goal_id = "x"


def test_input_is_frozen():
    goal = make_goal()
    with pytest.raises(FrozenInstanceError):
        goal.goal_id = "x"


def test_weights_are_frozen():
    weights = GoalUtilityWeights()
    with pytest.raises(FrozenInstanceError):
        weights.value = 1


def test_boolean_numeric_inputs_are_rejected():
    with pytest.raises(GoalUtilityEvaluationInputError):
        make_goal(value=True)


def test_boolean_weight_is_rejected():
    with pytest.raises(GoalUtilityEvaluationConfigurationError):
        GoalUtilityWeights(value=True)


def test_non_numeric_goal_field_is_rejected():
    with pytest.raises(GoalUtilityEvaluationInputError):
        GoalUtilityEvaluator().evaluate({"goal_id": "g", "value": "high"})


def test_description_defaults_empty():
    result = GoalUtilityEvaluator().evaluate({"goal_id": "g"})
    assert result.description == ""


def test_missing_numeric_inputs_default_to_zero_or_one():
    result = GoalUtilityEvaluator().evaluate({"goal_id": "g"})
    assert result.components["value"] == 0
    assert result.components["confidence"] == 1


def test_progress_is_clamped_by_input_contract():
    with pytest.raises(GoalUtilityEvaluationInputError):
        make_goal(progress=-0.1)


def test_confidence_one_and_evidence_one_preserve_utility():
    result = GoalUtilityEvaluator().evaluate(make_goal(confidence=1, evidence_quality=1))
    assert result.confidence_adjusted_utility == result.expected_utility


def test_zero_confidence_zeroes_adjusted_utility():
    result = GoalUtilityEvaluator().evaluate(make_goal(confidence=0))
    assert result.confidence_adjusted_utility == 0


def test_zero_evidence_zeroes_adjusted_utility():
    result = GoalUtilityEvaluator().evaluate(make_goal(evidence_quality=0))
    assert result.confidence_adjusted_utility == 0


def test_negative_expected_utility_maps_to_lower_half_score():
    result = GoalUtilityEvaluator().evaluate(
        make_goal(value=0, urgency=0, importance=0, progress=1, estimated_effort=1, risk=1, opportunity_cost=1)
    )
    assert result.utility_score < 0.5


def test_positive_expected_utility_maps_to_upper_half_score():
    result = GoalUtilityEvaluator().evaluate(
        make_goal(value=1, urgency=1, importance=1, progress=0, estimated_effort=0, risk=0, opportunity_cost=0)
    )
    assert result.utility_score > 0.5


def test_rationale_is_human_readable():
    result = GoalUtilityEvaluator().evaluate(make_goal())
    assert "benefit=" in result.rationale
    assert "cost=" in result.rationale
    assert "expected_utility=" in result.rationale


def test_metadata_is_not_shared():
    goal = make_goal()
    result = GoalUtilityEvaluator().evaluate(goal)
    result.metadata["x"] = 1
    assert "x" not in goal.metadata


def test_evaluate_many_does_not_mutate_inputs():
    goals = [make_goal(goal_id="a"), make_goal(goal_id="b")]
    before = [goal.__dict__.copy() for goal in goals]
    GoalUtilityEvaluator().evaluate_many(goals)
    assert [goal.__dict__ for goal in goals] == before


def test_custom_threshold_zero_recommends_nonnegative_adjusted_utility():
    result = GoalUtilityEvaluator(recommendation_threshold=0).evaluate(
        make_goal(value=0, urgency=0, importance=0, progress=1, estimated_effort=0, risk=0, opportunity_cost=0)
    )
    assert result.recommended is True


def test_custom_threshold_one_requires_maximum_utility():
    result = GoalUtilityEvaluator(recommendation_threshold=1).evaluate(make_goal())
    assert result.recommended is False


def test_minimum_confidence_zero_does_not_block():
    result = GoalUtilityEvaluator(minimum_confidence=0).evaluate(make_goal(confidence=0))
    assert result.recommendation != "insufficient_confidence"


def test_cost_score_increases_with_effort():
    low = GoalUtilityEvaluator().evaluate(make_goal(estimated_effort=0))
    high = GoalUtilityEvaluator().evaluate(make_goal(estimated_effort=1))
    assert high.cost_score > low.cost_score


def test_benefit_score_increases_with_value():
    low = GoalUtilityEvaluator().evaluate(make_goal(value=0))
    high = GoalUtilityEvaluator().evaluate(make_goal(value=1))
    assert high.benefit_score > low.benefit_score


def test_benefit_score_increases_with_urgency():
    low = GoalUtilityEvaluator().evaluate(make_goal(urgency=0))
    high = GoalUtilityEvaluator().evaluate(make_goal(urgency=1))
    assert high.benefit_score > low.benefit_score


def test_cost_score_increases_with_risk():
    low = GoalUtilityEvaluator().evaluate(make_goal(risk=0))
    high = GoalUtilityEvaluator().evaluate(make_goal(risk=1))
    assert high.cost_score > low.cost_score


def test_benefit_score_increases_as_progress_decreases():
    high_progress = GoalUtilityEvaluator().evaluate(make_goal(progress=1))
    low_progress = GoalUtilityEvaluator().evaluate(make_goal(progress=0))
    assert low_progress.benefit_score > high_progress.benefit_score


def test_opportunity_cost_reduces_expected_utility():
    low = GoalUtilityEvaluator().evaluate(make_goal(opportunity_cost=0))
    high = GoalUtilityEvaluator().evaluate(make_goal(opportunity_cost=1))
    assert high.expected_utility < low.expected_utility


def test_unknown_mapping_fields_are_ignored():
    result = GoalUtilityEvaluator().evaluate({"goal_id": "g", "value": 0.5, "unknown": 123})
    assert result.goal_id == "g"


def test_plain_string_goal_id_is_normalized():
    result = GoalUtilityEvaluator().evaluate({"goal_id": 123})
    assert result.goal_id == "123"


def test_evaluator_can_be_reused():
    evaluator = GoalUtilityEvaluator()
    first = evaluator.evaluate(make_goal(goal_id="a"))
    second = evaluator.evaluate(make_goal(goal_id="b"))
    assert first.goal_id == "a"
    assert second.goal_id == "b"


def test_result_metadata_is_mapping():
    result = GoalUtilityEvaluator().evaluate(make_goal())
    assert result.metadata["source"] == "test"


def test_result_components_are_mapping():
    result = GoalUtilityEvaluator().evaluate(make_goal())
    assert result.components["value"] == 0.8
