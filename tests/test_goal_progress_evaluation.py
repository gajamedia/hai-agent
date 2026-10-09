from __future__ import annotations

import pytest

from agent.llm.goal_management import GoalLifecycleManager, GoalStatus, ManagedGoal
from agent.llm.goal_progress_evaluation import (
    GoalAchievementEvidence,
    GoalEvaluationConfigurationError,
    GoalEvaluationError,
    GoalEvaluationEvidenceError,
    GoalEvaluationInputError,
    GoalEvaluationThresholds,
    GoalProgressAchievementEvaluator,
)


def goal(progress=0.0, **kwargs):
    defaults = dict(goal_id="g1", description="finish task", priority=0.5, sequence=1)
    defaults.update(progress=progress)
    defaults.update(kwargs)
    return ManagedGoal(**defaults)


def evidence(**kwargs):
    defaults = dict(source="test", success=True, progress_delta=0.2, confidence=1.0, quality=1.0)
    defaults.update(kwargs)
    return GoalAchievementEvidence(**defaults)


def test_evidence_defaults_are_bounded():
    item = GoalAchievementEvidence()
    assert item.source == "unknown"
    assert item.confidence == 0.5
    assert item.quality == 0.5


def test_evidence_metadata_is_copied():
    meta = {"a": 1}
    item = GoalAchievementEvidence(metadata=meta)
    meta["a"] = 2
    assert item.metadata == {"a": 1}


def test_evidence_invalid_confidence_rejected():
    with pytest.raises((GoalEvaluationConfigurationError, ValueError)):
        GoalAchievementEvidence(confidence=2.0)


def test_evidence_invalid_quality_rejected():
    with pytest.raises((GoalEvaluationConfigurationError, ValueError)):
        GoalAchievementEvidence(quality=-0.1)


def test_evidence_invalid_progress_rejected():
    with pytest.raises((GoalEvaluationConfigurationError, ValueError)):
        GoalAchievementEvidence(progress=1.1)


def test_large_progress_delta_is_allowed_for_normalization():
    item = GoalAchievementEvidence(progress_delta=2.0)
    assert item.progress_delta == 2.0


def test_thresholds_normalize_weights():
    thresholds = GoalEvaluationThresholds(success_weight=1, progress_weight=1, explicit_achievement_weight=2)
    assert thresholds.success_weight + thresholds.progress_weight + thresholds.explicit_achievement_weight == pytest.approx(1.0)
    assert thresholds.explicit_achievement_weight == pytest.approx(0.5)


def test_thresholds_reject_zero_weights():
    with pytest.raises(GoalEvaluationConfigurationError):
        GoalEvaluationThresholds(success_weight=0, progress_weight=0, explicit_achievement_weight=0)


def test_thresholds_reject_negative_tolerance():
    with pytest.raises(GoalEvaluationConfigurationError):
        GoalEvaluationThresholds(measurement_tolerance=-1)


def test_evaluator_defaults():
    evaluator = GoalProgressAchievementEvaluator()
    assert evaluator.thresholds.achievement_threshold == 0.8


def test_goal_mapping_is_supported():
    result = GoalProgressAchievementEvaluator().evaluate(
        {"goal_id": "g1", "description": "x", "progress": 0.1}
    )
    assert result.goal_id == "g1"
    assert result.previous_progress == 0.1


def test_invalid_goal_type_rejected():
    with pytest.raises(GoalEvaluationInputError):
        GoalProgressAchievementEvaluator().evaluate("not a goal")


def test_invalid_goal_mapping_rejected():
    with pytest.raises(GoalEvaluationInputError):
        GoalProgressAchievementEvaluator().evaluate({"description": "x"})


def test_empty_evidence_is_unchanged():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.2))
    assert result.evidence_count == 0
    assert result.progress_delta == 0.0
    assert result.achieved is False
    assert result.reason == "no_achievement_evidence"


def test_mapping_evidence_supported():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(0.1), [{"success": True, "progress_delta": 0.2, "quality": 1.0, "confidence": 1.0}]
    )
    assert result.evidence_count == 1
    assert result.measured_progress == 0.3


def test_object_evidence_supported():
    class Evidence:
        success = True
        progress_delta = 0.1
        confidence = 0.8
        quality = 0.9

    result = GoalProgressAchievementEvaluator().evaluate(goal(0.2), [Evidence()])
    assert result.evidence_count == 1


def test_blank_object_evidence_rejected():
    with pytest.raises(GoalEvaluationEvidenceError):
        GoalProgressAchievementEvaluator().evaluate(goal(), [object()])


def test_negative_progress_delta_rejected():
    with pytest.raises(GoalEvaluationEvidenceError):
        GoalProgressAchievementEvaluator().evaluate(goal(), [{"progress_delta": -0.1}])


def test_large_progress_delta_is_clamped():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.7), [{"progress_delta": 2.0, "quality": 1.0, "confidence": 1.0}])
    assert result.measured_progress == 1.0


def test_explicit_progress_takes_precedence_over_delta():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(0.2), [{"progress": 0.6, "progress_delta": 0.1, "quality": 1.0, "confidence": 1.0}]
    )
    assert result.measured_progress == 0.6
    assert result.progress_delta == 0.4


def test_multiple_explicit_progress_values_choose_max():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(0.1), [evidence(progress=0.4), evidence(progress=0.7)]
    )
    assert result.measured_progress == 0.7


def test_multiple_deltas_are_accumulated():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(0.1), [evidence(progress_delta=0.2), evidence(progress_delta=0.3)]
    )
    assert result.measured_progress == 0.6


def test_progress_signal_without_success_can_drive_achievement():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(0.0), [evidence(success=None, progress=1.0)]
    )
    assert result.achieved is True


def test_explicit_achievement_can_drive_achievement():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(0.1), [evidence(success=None, achieved=True, progress=None)]
    )
    assert result.achieved is False


def test_explicit_achievement_requires_score_threshold():
    thresholds = GoalEvaluationThresholds(
        achievement_threshold=0.5,
        success_weight=0.0,
        progress_weight=0.0,
        explicit_achievement_weight=1.0,
    )
    result = GoalProgressAchievementEvaluator(thresholds=thresholds).evaluate(
        goal(0.1), [evidence(success=None, achieved=True, progress=None)]
    )
    assert result.achieved is True


def test_success_only_is_not_automatic_achievement():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.0), [evidence(progress=None)])
    assert result.achieved is False
    assert result.recommended_status == "active"


def test_quality_filters_low_quality_evidence():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(0.0), [evidence(progress=1.0, quality=0.1)]
    )
    assert result.reason == "insufficient_evidence_quality"


def test_evidence_quality_is_confidence_weighted():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(), [evidence(quality=1.0, confidence=1.0), evidence(quality=0.0, confidence=1.0)]
    )
    assert result.evidence_quality == pytest.approx(0.5)


def test_success_signal_is_bounded():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(success=True)])
    assert 0.0 <= result.success_evidence <= 1.0


def test_failure_signal_does_not_count_as_success():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(success=False)])
    assert result.success_evidence == 0.0


def test_achievement_signal_is_bounded():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(achieved=True)])
    assert 0.0 <= result.achievement_evidence <= 1.0


def test_already_high_goal_is_achieved_without_new_evidence():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.9))
    assert result.achieved is True
    assert result.recommended_status == "completed"


def test_threshold_achievement_marks_completed():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.2), [evidence(progress=0.95)])
    assert result.achieved is True
    assert result.should_complete is True
    assert result.recommended_status == "completed"


def test_progress_but_not_achievement_recommends_active():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.2), [evidence(progress=0.4)])
    assert result.achieved is False
    assert result.recommended_status == "active"


def test_failure_without_achievement_is_active():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.3), [evidence(success=False, progress=0.3)])
    assert result.recommended_status == "active"
    assert "failure" in result.reason


def test_progress_delta_is_rounded():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.1), [evidence(progress_delta=0.2)])
    assert result.progress_delta == 0.2


def test_should_update_progress_when_measurement_changes():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.2), [evidence(progress=0.4)])
    assert result.should_update_progress is True


def test_should_update_progress_false_when_no_measurement():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(success=True, progress=None, progress_delta=None)])
    assert result.should_update_progress is False


def test_metadata_marks_evidence_sufficiency():
    result = GoalProgressAchievementEvaluator().evaluate(goal())
    assert result.metadata["evidence_sufficient"] is False


def test_evidence_is_returned_normalized():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [{"source": "obs", "success": True}])
    assert isinstance(result.evidence[0], GoalAchievementEvidence)
    assert result.evidence[0].source == "obs"


def test_result_is_immutable():
    result = GoalProgressAchievementEvaluator().evaluate(goal())
    with pytest.raises(AttributeError):
        result.achieved = True


def test_goal_manager_not_mutated():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "finish")
    before = manager.snapshot()
    result = GoalProgressAchievementEvaluator().evaluate(manager.get_goal("g1"), [evidence(progress=0.9)])
    after = manager.snapshot()
    assert result.achieved is True
    assert before == after


def test_terminal_goal_can_be_evaluated():
    result = GoalProgressAchievementEvaluator().evaluate(goal(1.0, status=GoalStatus.COMPLETED))
    assert result.achieved is True


def test_custom_threshold_can_be_used():
    thresholds = GoalEvaluationThresholds(achievement_threshold=0.6)
    result = GoalProgressAchievementEvaluator(thresholds=thresholds).evaluate(goal(), [evidence(progress=0.7)])
    assert result.achieved is True


def test_custom_tolerance_can_zero_small_delta():
    thresholds = GoalEvaluationThresholds(measurement_tolerance=0.1)
    result = GoalProgressAchievementEvaluator(thresholds=thresholds).evaluate(goal(0.2), [evidence(progress=0.25)])
    assert result.progress_delta == 0.0


def test_confidence_is_bounded_without_evidence():
    result = GoalProgressAchievementEvaluator().evaluate(goal())
    assert 0.0 <= result.confidence <= 1.0


def test_confidence_increases_with_high_quality_evidence():
    low = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(progress=None, quality=0.4, confidence=0.4)])
    high = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(progress=None, quality=1.0, confidence=1.0)])
    assert high.confidence > low.confidence


def test_reason_for_stagnant_evidence():
    result = GoalProgressAchievementEvaluator().evaluate(goal(0.3), [evidence(success=True, progress=0.3)])
    assert result.reason == "insufficient_achievement_signal"


def test_evidence_count_includes_low_quality_items():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(quality=0.1), evidence(quality=0.1)])
    assert result.evidence_count == 2


def test_multiple_success_evidence_is_aggregated():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(), [evidence(success=True, progress=None), evidence(success=True, progress=None)]
    )
    assert result.success_evidence == pytest.approx(1.0)


def test_mixed_success_evidence_is_aggregated():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(), [evidence(success=True, progress=None), evidence(success=False, progress=None)]
    )
    assert 0.0 < result.success_evidence < 1.0


def test_explicit_achievement_false_does_not_produce_positive_signal():
    result = GoalProgressAchievementEvaluator().evaluate(
        goal(), [evidence(success=None, achieved=False, progress=None)]
    )
    assert result.achievement_evidence == 0.0


def test_progress_measurement_is_true_for_explicit_progress():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(progress=0.2)])
    assert result.progress_measured is True


def test_progress_measurement_is_true_for_delta():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(progress_delta=0.2)])
    assert result.progress_measured is True


def test_recommended_status_needs_more_evidence_for_filtered_inputs():
    result = GoalProgressAchievementEvaluator().evaluate(goal(), [evidence(progress=1.0, quality=0.1)])
    assert result.recommended_status == "needs_more_evidence"


def test_base_error_is_exported():
    assert issubclass(GoalEvaluationConfigurationError, GoalEvaluationError)
