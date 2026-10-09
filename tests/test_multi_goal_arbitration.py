from __future__ import annotations

import pytest

from agent.llm.goal_management import GoalLifecycleManager, GoalStatus, ManagedGoal
from agent.llm.goal_progress_evaluation import GoalAchievementEvaluation
from agent.llm.multi_goal_arbitration import (
    MultiGoalArbitrationApplier,
    MultiGoalArbitrationCandidate,
    MultiGoalArbitrationConfigurationError,
    MultiGoalArbitrationDecision,
    MultiGoalArbitrationInputError,
    MultiGoalArbitrationThresholds,
    MultiGoalArbitrationWeights,
    MultiGoalArbiter,
)


def goal(goal_id, priority=0.5, progress=0.0, status=GoalStatus.PENDING, sequence=1):
    return ManagedGoal(goal_id, goal_id, priority=priority, progress=progress, status=status, sequence=sequence)


def evaluation(goal_id, score):
    return GoalAchievementEvaluation(
        goal_id=goal_id,
        goal_description=goal_id,
        previous_progress=0.0,
        measured_progress=score,
        progress_delta=score,
        evidence_count=1,
        evidence_quality=1.0,
        success_evidence=score,
        achievement_evidence=score,
        achievement_score=score,
        achieved=False,
        progress_measured=True,
        confidence=1.0,
        recommended_status="active",
        reason="test",
        evidence=(),
        metadata={},
    )


def test_default_weights_normalize():
    weights = MultiGoalArbitrationWeights()
    assert sum((weights.priority, weights.urgency, weights.progress_need, weights.achievement_gap)) == pytest.approx(1.0)


def test_zero_weights_rejected():
    with pytest.raises(MultiGoalArbitrationConfigurationError):
        MultiGoalArbitrationWeights(0, 0, 0, 0)


def test_negative_weight_rejected():
    with pytest.raises(MultiGoalArbitrationConfigurationError):
        MultiGoalArbitrationWeights(priority=-1)


def test_invalid_threshold_rejected():
    with pytest.raises(MultiGoalArbitrationConfigurationError):
        MultiGoalArbitrationThresholds(preempt_score_margin=1.1)


def test_invalid_minimum_urgency_rejected():
    with pytest.raises(MultiGoalArbitrationConfigurationError):
        MultiGoalArbitrationThresholds(minimum_urgency=-0.1)


def test_empty_goals():
    result = MultiGoalArbiter().arbitrate([])
    assert result.selected_goal_id is None
    assert result.reason == "no_eligible_goals"


def test_highest_score_selected_without_current():
    arbiter = MultiGoalArbiter()
    result = arbiter.arbitrate([goal("a", .9, sequence=1), goal("b", .2, sequence=2)])
    assert result.selected_goal_id == "a"
    assert result.switched is False
    assert result.preempted is False


def test_priority_dominates_default_score():
    arbiter = MultiGoalArbiter()
    result = arbiter.arbitrate([goal("a", .9, progress=.9), goal("b", .5, progress=0.0)])
    assert result.selected_goal_id == "a"


def test_urgency_can_change_winner():
    arbiter = MultiGoalArbiter()
    result = arbiter.arbitrate([goal("a", .5), goal("b", .5, sequence=2)], urgency={"b": 1.0})
    assert result.selected_goal_id == "b"


def test_progress_need_can_break_tie():
    arbiter = MultiGoalArbiter()
    result = arbiter.arbitrate([goal("a", .5, progress=.9), goal("b", .5, progress=0.0, sequence=2)])
    assert result.selected_goal_id == "b"


def test_evaluation_gap_can_influence_winner():
    arbiter = MultiGoalArbiter()
    result = arbiter.arbitrate(
        [goal("a", .5), goal("b", .5, sequence=2)],
        evaluations={"a": evaluation("a", 1.0), "b": evaluation("b", 0.0)},
    )
    assert result.selected_goal_id == "b"


def test_current_goal_retained_when_margin_not_exceeded():
    arbiter = MultiGoalArbiter()
    result = arbiter.arbitrate([goal("a", .6, sequence=1), goal("b", .62, sequence=2)], current_goal_id="a")
    assert result.selected_goal_id == "a"
    assert result.switched is False
    assert result.preempted is False
    assert "retained" in result.reason


def test_current_goal_preempted_when_margin_exceeded():
    arbiter = MultiGoalArbiter()
    result = arbiter.arbitrate([goal("a", .4, sequence=1), goal("b", .9, sequence=2)], current_goal_id="a")
    assert result.selected_goal_id == "b"
    assert result.switched is True
    assert result.preempted is True


def test_current_missing_gets_replaced():
    result = MultiGoalArbiter().arbitrate([goal("a", .8)], current_goal_id="missing")
    assert result.selected_goal_id == "a"
    assert result.switched is True
    assert result.preempted is False
    assert result.reason == "current_goal_not_eligible"


def test_terminal_current_gets_replaced():
    result = MultiGoalArbiter().arbitrate(
        [goal("a", .5, status=GoalStatus.COMPLETED, sequence=1), goal("b", .6, sequence=2)],
        current_goal_id="a",
    )
    assert result.selected_goal_id == "b"


def test_terminal_candidate_excluded():
    result = MultiGoalArbiter().arbitrate([goal("a", .9, status=GoalStatus.COMPLETED)])
    assert result.selected_goal_id is None
    assert result.candidates[0].exclusion_reason == "terminal_goal"


def test_blocked_candidate_excluded():
    result = MultiGoalArbiter().arbitrate([goal("a", .9), goal("b", .2, sequence=2)], blocked_goal_ids={"a"})
    assert result.selected_goal_id == "b"
    assert result.candidates[0].eligible is False
    assert result.candidates[0].exclusion_reason == "blocked_goal"


def test_minimum_urgency_filters_candidate():
    arbiter = MultiGoalArbiter(thresholds=MultiGoalArbitrationThresholds(minimum_urgency=.5))
    result = arbiter.arbitrate([goal("a", .9), goal("b", .5, sequence=2)], urgency={"a": .2, "b": .8})
    assert result.selected_goal_id == "b"


def test_eligibility_order_is_ranked():
    result = MultiGoalArbiter().arbitrate([goal("a", .2, sequence=1), goal("b", .8, sequence=2), goal("c", .5, sequence=3)])
    assert result.ranked_goal_ids[0] == "b"
    assert set(result.ranked_goal_ids) == {"a", "b", "c"}


def test_candidate_order_preserves_sequence():
    result = MultiGoalArbiter().arbitrate([goal("b", .8, sequence=2), goal("a", .8, sequence=1)])
    assert [item.goal_id for item in result.candidates] == ["a", "b"]


def test_tie_prefers_higher_priority_after_score():
    weights = MultiGoalArbitrationWeights(priority=1, urgency=0, progress_need=0, achievement_gap=0)
    result = MultiGoalArbiter(weights=weights).arbitrate([goal("a", .6), goal("b", .5, sequence=2)])
    assert result.selected_goal_id == "a"


def test_tie_prefers_higher_urgency_after_score():
    weights = MultiGoalArbitrationWeights(priority=0, urgency=1, progress_need=0, achievement_gap=0)
    result = MultiGoalArbiter(weights=weights).arbitrate([goal("a", .5), goal("b", .5, sequence=2)], urgency={"a": .3, "b": .9})
    assert result.selected_goal_id == "b"


def test_sequence_breaks_complete_tie():
    weights = MultiGoalArbitrationWeights(priority=1, urgency=0, progress_need=0, achievement_gap=0)
    result = MultiGoalArbiter(weights=weights).arbitrate([goal("a", .5, sequence=2), goal("b", .5, sequence=1)])
    assert result.selected_goal_id == "b"


def test_scores_are_bounded():
    result = MultiGoalArbiter().arbitrate([goal("a", 1.0)])
    assert 0.0 <= result.selected_score <= 1.0


def test_decision_has_selection_property():
    result = MultiGoalArbiter().arbitrate([goal("a")])
    assert result.has_selection is True
    assert result.should_switch is False


def test_empty_decision_has_no_selection():
    result = MultiGoalArbiter().arbitrate([])
    assert result.has_selection is False
    assert result.should_switch is False


def test_metadata_contains_threshold():
    result = MultiGoalArbiter().arbitrate([goal("a")])
    assert "score_margin_threshold" in result.metadata


def test_metadata_contains_weights():
    result = MultiGoalArbiter().arbitrate([goal("a")])
    assert set(result.metadata["weights"]) == {"priority", "urgency", "progress_need", "achievement_gap"}


def test_urgency_defaults_to_zero():
    result = MultiGoalArbiter().arbitrate([goal("a")])
    assert result.candidates[0].urgency == 0.0


def test_urgency_invalid_type_rejected():
    with pytest.raises(MultiGoalArbitrationInputError):
        MultiGoalArbiter().arbitrate([goal("a")], urgency={"a": "x"})


def test_urgency_out_of_bounds_rejected():
    with pytest.raises(MultiGoalArbitrationInputError):
        MultiGoalArbiter().arbitrate([goal("a")], urgency={"a": 2})


def test_invalid_goal_type_rejected():
    with pytest.raises(MultiGoalArbitrationInputError):
        MultiGoalArbiter().arbitrate([object()])


def test_invalid_goal_mapping_rejected():
    with pytest.raises(MultiGoalArbitrationInputError):
        MultiGoalArbiter().arbitrate([{"description": "missing id"}])


def test_invalid_evaluation_score_rejected():
    with pytest.raises(MultiGoalArbitrationInputError):
        MultiGoalArbiter().arbitrate([goal("a")], evaluations={"a": {"achievement_score": "bad"}})


def test_mapping_evaluation_supported():
    result = MultiGoalArbiter().arbitrate([goal("a")], evaluations={"a": {"achievement_score": .4}})
    assert result.candidates[0].achievement_score == .4


def test_object_evaluation_supported():
    result = MultiGoalArbiter().arbitrate([goal("a")], evaluations={"a": evaluation("a", .4)})
    assert result.candidates[0].achievement_score == .4


def test_manager_arbitration_uses_current_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A", priority=.4)
    manager.add_goal("b", "B", priority=.9)
    manager.set_current_goal("a")
    result = MultiGoalArbiter().arbitrate_manager(manager)
    assert result.previous_goal_id == "a"
    assert result.selected_goal_id == "b"


def test_manager_wrong_type_rejected():
    with pytest.raises(MultiGoalArbitrationInputError):
        MultiGoalArbiter().arbitrate_manager(object())


def test_applier_switches_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A", priority=.2)
    manager.add_goal("b", "B", priority=.9)
    manager.set_current_goal("a")
    decision = MultiGoalArbiter().arbitrate_manager(manager)
    selected = MultiGoalArbitrationApplier().apply(manager, decision)
    assert selected.goal_id == "b"
    assert manager.current_goal_id == "b"


def test_applier_no_selection_clears_current_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A", priority=.5)
    manager.set_current_goal("a")
    manager.complete("a")
    decision = MultiGoalArbiter().arbitrate_manager(manager)
    assert decision.selected_goal_id is None
    assert MultiGoalArbitrationApplier().apply(manager, decision) is None
    assert manager.current_goal_id is None


def test_applier_wrong_manager_rejected():
    with pytest.raises(Exception):
        MultiGoalArbitrationApplier().apply(object(), MultiGoalArbitrationDecision(None, None, 0, 0, 0, False, False, "x", (), (), (), {}))


def test_applier_wrong_decision_rejected():
    manager = GoalLifecycleManager()
    with pytest.raises(Exception):
        MultiGoalArbitrationApplier().apply(manager, object())


def test_candidate_is_immutable_dataclass():
    candidate = MultiGoalArbiter().arbitrate([goal("a")]).candidates[0]
    assert isinstance(candidate, MultiGoalArbitrationCandidate)
    with pytest.raises(Exception):
        candidate.score = .2


def test_decision_score_margin_is_current_relative():
    result = MultiGoalArbiter(thresholds=MultiGoalArbitrationThresholds(preempt_score_margin=0)).arbitrate(
        [goal("a", .1), goal("b", .9, sequence=2)], current_goal_id="a"
    )
    assert result.score_margin > 0


def test_blocked_current_is_replaced_even_with_small_margin():
    result = MultiGoalArbiter().arbitrate(
        [goal("a", .6), goal("b", .61, sequence=2)],
        current_goal_id="a",
        blocked_goal_ids={"a"},
    )
    assert result.selected_goal_id == "b"
    assert result.switched is True


def test_terminal_goals_never_ranked():
    result = MultiGoalArbiter().arbitrate([
        goal("done", 1.0, status=GoalStatus.COMPLETED),
        goal("open", .1, sequence=2),
    ])
    assert result.ranked_goal_ids == ("open",)


def test_candidate_metadata_copies_goal_metadata():
    g = ManagedGoal("a", "A", metadata={"source": "test"})
    result = MultiGoalArbiter().arbitrate([g])
    assert result.candidates[0].metadata == {"source": "test"}


def test_default_reason_for_first_selection():
    result = MultiGoalArbiter().arbitrate([goal("a")])
    assert result.reason == "highest_arbitration_score"


def test_continue_current_goal_reason():
    result = MultiGoalArbiter().arbitrate([goal("a")], current_goal_id="a")
    assert result.reason == "continue_current_goal"


def test_score_rounding_is_stable():
    result = MultiGoalArbiter().arbitrate([goal("a", .3333333333333)])
    assert result.selected_score == round(result.selected_score, 12)
