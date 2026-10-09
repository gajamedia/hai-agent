import pytest

from agent.learning.meta_learning_feedback_policy import (
    FeedbackDecision,
)
from agent.learning.meta_learning_decision_integration import (
    IntegratedDecision,
    MetaLearningDecisionIntegrator,
)


def test_integrator_requires_valid_base_mode():
    integrator = MetaLearningDecisionIntegrator()

    feedback = FeedbackDecision()

    with pytest.raises(ValueError):
        integrator.integrate(
            base_mode="",
            base_reason="test",
            feedback_decision=feedback,
        )

    with pytest.raises(ValueError):
        integrator.integrate(
            base_mode="invalid",
            base_reason="test",
            feedback_decision=feedback,
        )


def test_integrator_rejects_none_feedback():
    integrator = MetaLearningDecisionIntegrator()

    with pytest.raises(
        ValueError,
        match="feedback_decision cannot be None",
    ):
        integrator.integrate(
            base_mode="exploit",
            base_reason="base decision",
            feedback_decision=None,
        )


def test_neutral_feedback_keeps_base_decision():
    integrator = MetaLearningDecisionIntegrator()

    decision = integrator.integrate(
        base_mode="exploit",
        base_reason="low uncertainty",
        feedback_decision=FeedbackDecision(),
    )

    assert isinstance(
        decision,
        IntegratedDecision,
    )

    assert decision.base_mode == "exploit"
    assert decision.final_mode == "exploit"
    assert not decision.mode_changed
    assert not decision.feedback_applied


def test_explore_feedback_does_not_override_exploit():
    integrator = MetaLearningDecisionIntegrator()

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=0.70,
        apply=True,
    )

    decision = integrator.integrate(
        base_mode="exploit",
        base_reason="risk too high for exploration",
        feedback_decision=feedback,
    )

    assert decision.base_mode == "exploit"
    assert decision.final_mode == "exploit"

    assert decision.feedback_mode == "explore"
    assert decision.feedback_bias == 0.70

    assert decision.feedback_applied
    assert not decision.mode_changed


def test_exploit_feedback_does_not_override_explore():
    integrator = MetaLearningDecisionIntegrator()

    feedback = FeedbackDecision(
        mode_bias="exploit",
        bias_strength=0.60,
        apply=True,
    )

    decision = integrator.integrate(
        base_mode="explore",
        base_reason="uncertainty and information gain are high",
        feedback_decision=feedback,
    )

    assert decision.base_mode == "explore"
    assert decision.final_mode == "explore"

    assert decision.feedback_mode == "exploit"
    assert decision.feedback_bias == 0.60

    assert decision.feedback_applied
    assert not decision.mode_changed


def test_feedback_without_mode_can_request_adaptation():
    integrator = MetaLearningDecisionIntegrator()

    feedback = FeedbackDecision(
        apply=True,
        request_adaptation=True,
    )

    decision = integrator.integrate(
        base_mode="exploit",
        base_reason="stable policy",
        feedback_decision=feedback,
    )

    assert decision.base_mode == "exploit"
    assert decision.final_mode == "exploit"

    assert decision.feedback_applied
    assert decision.adaptation_requested
    assert not decision.mode_changed
