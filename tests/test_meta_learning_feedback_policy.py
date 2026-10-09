import pytest

from agent.learning.meta_learning_feedback import (
    MetaLearningFeedback,
)
from agent.learning.meta_learning_feedback_policy import (
    FeedbackDecision,
    MetaLearningFeedbackPolicy,
)


def test_policy_rejects_none_feedback():
    policy = MetaLearningFeedbackPolicy()

    with pytest.raises(
        ValueError,
        match="feedback cannot be None",
    ):
        policy.evaluate(None)


def test_policy_rejects_low_confidence():
    policy = MetaLearningFeedbackPolicy(
        minimum_confidence=0.50,
    )

    feedback = MetaLearningFeedback(
        preferred_mode="exploit",
        exploitation_pressure=0.80,
        confidence=0.40,
    )

    decision = policy.evaluate(
        feedback
    )

    assert isinstance(
        decision,
        FeedbackDecision,
    )

    assert not decision.apply
    assert decision.mode_bias is None
    assert decision.bias_strength == 0.0


def test_policy_applies_strong_exploitation_feedback():
    policy = MetaLearningFeedbackPolicy()

    feedback = MetaLearningFeedback(
        preferred_mode="exploit",
        exploitation_pressure=0.80,
        confidence=0.90,
    )

    decision = policy.evaluate(
        feedback
    )

    assert decision.apply
    assert decision.should_exploit
    assert not decision.should_explore

    assert decision.mode_bias == "exploit"
    assert decision.bias_strength > 0.0
    assert decision.bias_strength <= 0.75


def test_policy_applies_strong_exploration_feedback():
    policy = MetaLearningFeedbackPolicy()

    feedback = MetaLearningFeedback(
        preferred_mode="explore",
        exploration_pressure=0.80,
        confidence=0.90,
    )

    decision = policy.evaluate(
        feedback
    )

    assert decision.apply
    assert decision.should_explore
    assert not decision.should_exploit

    assert decision.mode_bias == "explore"
    assert decision.bias_strength > 0.0


def test_policy_rejects_weak_mode_pressure():
    policy = MetaLearningFeedbackPolicy(
        minimum_pressure=0.20,
    )

    feedback = MetaLearningFeedback(
        preferred_mode="exploit",
        exploitation_pressure=0.10,
        confidence=0.90,
    )

    decision = policy.evaluate(
        feedback
    )

    assert not decision.apply
    assert decision.mode_bias is None
    assert decision.bias_strength == 0.0


def test_policy_detects_adaptation_request_without_mode_bias():
    policy = MetaLearningFeedbackPolicy(
        minimum_adaptation_pressure=0.50,
    )

    feedback = MetaLearningFeedback(
        preferred_mode=None,
        adaptation_pressure=0.80,
        confidence=0.90,
    )

    decision = policy.evaluate(
        feedback
    )

    assert decision.apply
    assert decision.request_adaptation
    assert decision.mode_bias is None
    assert decision.is_neutral is False


def test_policy_ignores_weak_adaptation_pressure():
    policy = MetaLearningFeedbackPolicy(
        minimum_adaptation_pressure=0.50,
    )

    feedback = MetaLearningFeedback(
        preferred_mode=None,
        adaptation_pressure=0.20,
        confidence=0.90,
    )

    decision = policy.evaluate(
        feedback
    )

    assert not decision.apply
    assert not decision.request_adaptation


def test_policy_can_apply_mode_and_adaptation_together():
    policy = MetaLearningFeedbackPolicy(
        minimum_adaptation_pressure=0.50,
    )

    feedback = MetaLearningFeedback(
        preferred_mode="exploit",
        exploitation_pressure=0.70,
        adaptation_pressure=0.80,
        confidence=0.90,
    )

    decision = policy.evaluate(
        feedback
    )

    assert decision.apply
    assert decision.should_exploit
    assert decision.request_adaptation
    assert decision.bias_strength > 0.0


def test_policy_limits_bias_strength():
    policy = MetaLearningFeedbackPolicy(
        maximum_bias=0.40,
    )

    feedback = MetaLearningFeedback(
        preferred_mode="explore",
        exploration_pressure=1.0,
        confidence=1.0,
    )

    decision = policy.evaluate(
        feedback
    )

    assert decision.apply
    assert decision.bias_strength == 0.40


def test_policy_unsupported_mode_is_rejected():
    policy = MetaLearningFeedbackPolicy()

    feedback = MetaLearningFeedback(
        preferred_mode="invalid",
        confidence=0.90,
    )

    decision = policy.evaluate(
        feedback
    )

    assert not decision.apply
    assert decision.mode_bias is None


def test_feedback_decision_helpers():
    explore = FeedbackDecision(
        mode_bias="explore",
        bias_strength=0.5,
        apply=True,
    )

    assert explore.has_mode_bias
    assert explore.should_explore
    assert not explore.should_exploit
    assert not explore.is_neutral

    exploit = FeedbackDecision(
        mode_bias="exploit",
        bias_strength=0.5,
        apply=True,
    )

    assert exploit.has_mode_bias
    assert exploit.should_exploit
    assert not exploit.should_explore

    neutral = FeedbackDecision()

    assert not neutral.has_mode_bias
    assert neutral.is_neutral


def test_policy_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        MetaLearningFeedbackPolicy(
            minimum_confidence=-0.1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackPolicy(
            minimum_confidence=1.1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackPolicy(
            minimum_pressure=-0.1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackPolicy(
            maximum_bias=1.1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackPolicy(
            minimum_adaptation_pressure=-0.1
        )


def test_policy_neutral_feedback_with_sufficient_confidence():
    policy = MetaLearningFeedbackPolicy()

    feedback = MetaLearningFeedback(
        preferred_mode=None,
        confidence=0.80,
        adaptation_pressure=0.20,
    )

    decision = policy.evaluate(
        feedback
    )

    assert not decision.apply
    assert not decision.request_adaptation
    assert decision.mode_bias is None
    assert decision.bias_strength == 0.0
