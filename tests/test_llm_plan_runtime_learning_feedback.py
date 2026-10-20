from types import SimpleNamespace

import pytest

from agent.llm.llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedback,
    LLMPlanRuntimeLearningFeedbackBridge,
    LLMPlanRuntimeLearningFeedbackConfigurationError,
    LLMPlanRuntimeLearningFeedbackEngine,
    LLMPlanRuntimeLearningFeedbackInputError,
)


def experience(step_index=1, action="open_box", success=True):
    return SimpleNamespace(
        step_index=step_index,
        action=action,
        success=success,
    )


def reflection(
    *,
    success=True,
    progress_delta=0.5,
    confidence=0.9,
    evidence_quality=1.0,
    diagnosis="successful_outcome_with_progress",
    lesson="The action produced progress.",
    next_step="Continue.",
    change_required=False,
):
    return SimpleNamespace(
        success=success,
        progress_delta=progress_delta,
        confidence=confidence,
        evidence_quality=evidence_quality,
        diagnosis=diagnosis,
        lesson=lesson,
        next_step=next_step,
        change_required=change_required,
    )


def test_engine_builds_feedback():
    result = LLMPlanRuntimeLearningFeedbackEngine().build(
        experience(),
        reflection(),
    )
    assert isinstance(result, LLMPlanRuntimeLearningFeedback)
    assert result.step_index == 1
    assert result.action == "open_box"
    assert result.success is True
    assert result.progress_delta == 0.5
    assert result.information_gain == 0.5
    assert result.risk_level == "low"
    assert result.continue_recommended is True
    assert result.exploration_recommended is False


def test_feedback_is_immutable():
    result = LLMPlanRuntimeLearningFeedbackEngine().build(
        experience(), reflection()
    )
    with pytest.raises(Exception):
        result.risk_score = 1.0


def test_failure_recommends_exploration_and_stop():
    result = LLMPlanRuntimeLearningFeedbackEngine().build(
        experience(success=False),
        reflection(
            success=False,
            progress_delta=0.0,
            diagnosis="failure_with_reported_error",
            change_required=True,
        ),
    )
    assert result.risk_level == "high"
    assert result.exploration_recommended is True
    assert result.continue_recommended is False
    assert result.recovery_recommended is True


def test_non_positive_progress_is_medium_risk():
    result = LLMPlanRuntimeLearningFeedbackEngine().build(
        experience(),
        reflection(success=True, progress_delta=0.0),
    )
    assert result.risk_level == "medium"
    assert result.exploration_recommended is True


def test_change_required_escalates_success_to_behavior_change():
    result = LLMPlanRuntimeLearningFeedbackEngine().build(
        experience(),
        reflection(
            success=True,
            progress_delta=0.2,
            change_required=True,
        ),
    )
    assert result.change_required is True
    assert result.continue_recommended is False
    assert result.should_change_behavior is True


def test_unknown_success_defaults_to_medium_risk():
    exp = experience()
    exp.success = None
    ref = reflection(success=None)
    result = LLMPlanRuntimeLearningFeedbackEngine().build(exp, ref)
    assert result.success is None
    assert result.risk_level == "medium"


def test_negative_progress_produces_zero_information_gain():
    result = LLMPlanRuntimeLearningFeedbackEngine().build(
        experience(),
        reflection(progress_delta=-0.4),
    )
    assert result.progress_delta == -0.4
    assert result.information_gain == 0.0


def test_none_progress_produces_zero_information_gain():
    result = LLMPlanRuntimeLearningFeedbackEngine().build(
        experience(),
        reflection(progress_delta=None),
    )
    assert result.information_gain == 0.0


@pytest.mark.parametrize("value", [-1, -0.1, 0, 0.25, 1, 2])
def test_evidence_quality_is_bounded(value):
    ref = reflection(evidence_quality=value)
    result = LLMPlanRuntimeLearningFeedbackEngine().build(experience(), ref)
    assert 0.0 <= result.evidence_quality <= 1.0


@pytest.mark.parametrize("value", [-1, -0.1, 0, 0.25, 1, 2])
def test_confidence_is_bounded(value):
    ref = reflection(confidence=value)
    result = LLMPlanRuntimeLearningFeedbackEngine().build(experience(), ref)
    assert 0.0 <= result.confidence <= 1.0


def test_step_index_must_be_integer():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackInputError):
        LLMPlanRuntimeLearningFeedbackEngine().build(
            SimpleNamespace(step_index="x", action="inspect"),
            reflection(),
        )


def test_negative_step_index_is_rejected():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackInputError):
        LLMPlanRuntimeLearningFeedbackEngine().build(
            experience(step_index=-1), reflection()
        )


def test_missing_action_is_rejected():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackInputError):
        LLMPlanRuntimeLearningFeedbackEngine().build(
            experience(action=""), reflection()
        )


def test_missing_experience_is_rejected():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackInputError):
        LLMPlanRuntimeLearningFeedbackEngine().build(None, reflection())


def test_missing_reflection_is_rejected():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackInputError):
        LLMPlanRuntimeLearningFeedbackEngine().build(experience(), None)


def test_bridge_requires_reflection_provider_or_embedded_reflection():
    bridge = LLMPlanRuntimeLearningFeedbackBridge()
    with pytest.raises(LLMPlanRuntimeLearningFeedbackInputError):
        bridge.observe_experience(
            experience(), 1, "open_box", {}
        )


def test_bridge_uses_reflection_provider():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    result = bridge.observe_experience(
        experience(), 1, "open_box", {"source": "test"}
    )
    assert result.step_index == 1
    assert bridge.last_feedback is result
    assert len(bridge.feedback_history) == 1


def test_bridge_reads_embedded_reflection():
    episode = experience()
    episode.reflection = reflection()
    bridge = LLMPlanRuntimeLearningFeedbackBridge()
    result = bridge.observe_experience(episode, 1, "open_box", {})
    assert result.reflection is episode.reflection


def test_bridge_tracks_feedback_by_step():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    bridge.observe_experience(experience(step_index=3), 3, "a", {})
    assert bridge.get_feedback(3) is not None
    assert 3 in bridge.feedback_by_step


def test_bridge_deduplicates_same_step():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    first = bridge.observe_experience(experience(step_index=2), 2, "a", {})
    second = bridge.observe_experience(experience(step_index=2), 2, "a", {})
    assert second is first
    assert len(bridge.feedback_history) == 1


def test_bridge_preserves_metadata():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    result = bridge.observe_experience(
        experience(), 1, "open_box", {"trace_id": "abc"}
    )
    assert result.metadata["trace_id"] == "abc"
    assert result.metadata["observer_step_index"] == 1
    assert result.metadata["observer_action"] == "open_box"


def test_bridge_observer_receives_feedback():
    seen = []
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
        feedback_observer=seen.append,
    )
    result = bridge.observe_experience(experience(), 1, "a", {})
    assert seen == [result]


def test_runtime_evaluation_provider_is_called():
    calls = []

    def evaluator(feedback, episode):
        calls.append((feedback.step_index, episode))
        return {"mode": "observe"}

    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
        runtime_evaluation_provider=evaluator,
    )
    result = bridge.observe_experience(experience(), 1, "a", {})
    assert calls == [(1, result.episode)]
    assert result.runtime_evaluation == {"mode": "observe"}


def test_runtime_evaluation_is_attached_to_feedback():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
        runtime_evaluation_provider=lambda feedback, episode: {"risk": feedback.risk_score},
    )
    result = bridge.observe_experience(experience(), 1, "a", {})
    assert result.runtime_evaluation["risk"] == 0.25


def test_reset_clears_feedback_history():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    bridge.observe_experience(experience(), 1, "a", {})
    bridge.reset()
    assert bridge.last_feedback is None
    assert bridge.feedback_history == ()
    assert bridge.feedback_by_step == {}


def test_latest_runtime_guidance_is_none_without_feedback():
    bridge = LLMPlanRuntimeLearningFeedbackBridge()
    assert bridge.latest_runtime_guidance() is None


def test_latest_runtime_guidance_contains_bounded_signals():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    bridge.observe_experience(experience(), 1, "a", {})
    guidance = bridge.latest_runtime_guidance()
    assert guidance["continue_recommended"] is True
    assert guidance["exploration_recommended"] is False
    assert guidance["risk_score"] == 0.25
    assert guidance["information_gain"] == 0.5


def test_configuration_rejects_non_callable_reflection_provider():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackConfigurationError):
        LLMPlanRuntimeLearningFeedbackBridge(reflection_provider=1)


def test_configuration_rejects_non_callable_observer():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackConfigurationError):
        LLMPlanRuntimeLearningFeedbackBridge(feedback_observer=1)


def test_configuration_rejects_non_callable_runtime_provider():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackConfigurationError):
        LLMPlanRuntimeLearningFeedbackBridge(runtime_evaluation_provider=1)


def test_configuration_rejects_invalid_engine():
    with pytest.raises(LLMPlanRuntimeLearningFeedbackConfigurationError):
        LLMPlanRuntimeLearningFeedbackBridge(feedback_engine=object())


def test_successful_positive_feedback_reports_positive_progress():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(progress_delta=0.8),
    )
    result = bridge.observe_experience(experience(), 1, "a", {})
    assert result.is_positive_progress is True
    assert result.information_gain == 0.8


def test_recovery_recommendation_for_failure():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(
            success=False,
            progress_delta=0,
            change_required=True,
        ),
    )
    result = bridge.observe_experience(experience(success=False), 1, "a", {})
    assert result.recovery_recommended is True
    assert result.should_change_behavior is True


def test_metadata_is_copied_not_shared():
    source = {"x": {"nested": True}}
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    result = bridge.observe_experience(experience(), 1, "a", source)
    assert result.metadata is not source


def test_feedback_bridge_can_be_used_as_experience_observer():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    result = bridge.observe_experience(
        episode=experience(step_index=4),
        step_index=4,
        action="inspect",
        metadata={"runtime": "live"},
    )
    assert result.step_index == 4


def test_success_without_progress_is_not_positive():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(progress_delta=0),
    )
    result = bridge.observe_experience(experience(), 1, "a", {})
    assert result.is_positive_progress is False


def test_feedback_history_preserves_execution_order():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    for index in (1, 2, 3):
        bridge.observe_experience(experience(step_index=index), index, str(index), {})
    assert [item.step_index for item in bridge.feedback_history] == [1, 2, 3]


def test_runtime_evaluation_is_optional():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(),
    )
    result = bridge.observe_experience(experience(), 1, "a", {})
    assert result.runtime_evaluation is None


def test_feedback_diagnosis_is_preserved():
    bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: reflection(diagnosis="failure_with_partial_progress"),
    )
    result = bridge.observe_experience(experience(success=False), 1, "a", {})
    assert result.diagnosis == "failure_with_partial_progress"

def test_success_without_target_progress_preserves_action_success():
    result = LLMPlanRuntimeLearningFeedbackEngine().build(
        experience(action="open_box", success=True),
        reflection(
            success=True,
            progress_delta=0.0,
            diagnosis="no_observed_target_progress",
            lesson="The observed state did not advance the target.",
            next_step="Reassess whether another action is appropriate.",
        ),
    )

    assert result.success is True
    assert result.progress_delta == 0.0
    assert result.diagnosis == "no_observed_target_progress"
    assert result.risk_level == "medium"
    assert result.change_required is False
