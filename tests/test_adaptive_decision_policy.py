import pytest

from agent.learning.adaptive_decision_policy import (
    AdaptiveDecision,
    AdaptiveDecisionPolicy,
    AdaptiveModeScore,
)
from agent.learning.meta_learning_feedback_policy import (
    FeedbackDecision,
)


def test_exploration_is_selected_when_information_gain_is_high():
    policy = AdaptiveDecisionPolicy()

    decision = policy.evaluate(
        uncertainty=0.90,
        information_gain=0.95,
        risk=0.10,
    )

    assert isinstance(
        decision,
        AdaptiveDecision,
    )

    assert decision.mode == "explore"
    assert decision.is_explore
    assert not decision.is_exploit


def test_exploitation_is_selected_when_uncertainty_is_low():
    policy = AdaptiveDecisionPolicy()

    decision = policy.evaluate(
        uncertainty=0.10,
        information_gain=0.10,
        risk=0.10,
    )

    assert decision.mode == "exploit"
    assert decision.is_exploit
    assert not decision.is_explore


def test_exploration_feedback_can_change_borderline_decision():
    policy = AdaptiveDecisionPolicy()

    base = policy.evaluate(
        uncertainty=0.50,
        information_gain=0.50,
        risk=0.50,
    )

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=0.25,
        apply=True,
    )

    adaptive = policy.evaluate(
        uncertainty=0.50,
        information_gain=0.50,
        risk=0.50,
        feedback_decision=feedback,
    )

    assert base.mode == "exploit"
    assert adaptive.mode == "explore"
    assert adaptive.feedback_applied


def test_feedback_cannot_exceed_maximum_bonus():
    policy = AdaptiveDecisionPolicy(
        maximum_feedback_bonus=0.10,
    )

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=1.0,
        apply=True,
    )

    explore, exploit = policy.score_modes(
        uncertainty=0.5,
        information_gain=0.5,
        risk=0.5,
        feedback_decision=feedback,
    )

    assert isinstance(
        explore,
        AdaptiveModeScore,
    )

    assert explore.feedback_bonus == 0.10
    assert exploit.feedback_bonus == 0.0


def test_feedback_does_not_override_forbidden_exploration():
    policy = AdaptiveDecisionPolicy()

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=0.75,
        apply=True,
    )

    decision = policy.evaluate(
        uncertainty=1.0,
        information_gain=1.0,
        risk=0.0,
        feedback_decision=feedback,
        explore_allowed=False,
        exploit_allowed=True,
    )

    assert decision.mode == "exploit"
    assert decision.explore_score == float("-inf")


def test_feedback_does_not_override_forbidden_exploitation():
    policy = AdaptiveDecisionPolicy()

    feedback = FeedbackDecision(
        mode_bias="exploit",
        bias_strength=0.75,
        apply=True,
    )

    decision = policy.evaluate(
        uncertainty=0.0,
        information_gain=0.0,
        risk=0.0,
        feedback_decision=feedback,
        explore_allowed=True,
        exploit_allowed=False,
    )

    assert decision.mode == "explore"
    assert decision.exploit_score == float("-inf")


def test_both_modes_cannot_be_forbidden():
    policy = AdaptiveDecisionPolicy()

    with pytest.raises(
        ValueError,
        match="at least one decision mode",
    ):
        policy.evaluate(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.5,
            explore_allowed=False,
            exploit_allowed=False,
        )


def test_invalid_uncertainty_is_rejected():
    policy = AdaptiveDecisionPolicy()

    with pytest.raises(ValueError):
        policy.evaluate(
            uncertainty=1.1,
            information_gain=0.5,
            risk=0.5,
        )


def test_invalid_information_gain_is_rejected():
    policy = AdaptiveDecisionPolicy()

    with pytest.raises(ValueError):
        policy.evaluate(
            uncertainty=0.5,
            information_gain=-0.1,
            risk=0.5,
        )


def test_invalid_risk_is_rejected():
    policy = AdaptiveDecisionPolicy()

    with pytest.raises(ValueError):
        policy.evaluate(
            uncertainty=0.5,
            information_gain=0.5,
            risk=1.1,
        )


def test_tie_prefers_exploitation():
    policy = AdaptiveDecisionPolicy(
        exploration_weight=0.0,
        uncertainty_weight=0.0,
        risk_weight=0.0,
        exploitation_weight=0.0,
        exploitation_stability_weight=0.0,
    )

    decision = policy.evaluate(
        uncertainty=0.5,
        information_gain=0.5,
        risk=0.5,
    )

    assert decision.explore_score == 0.0
    assert decision.exploit_score == 0.0
    assert decision.mode == "exploit"


def test_adaptation_request_is_preserved():
    policy = AdaptiveDecisionPolicy()

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=0.20,
        apply=True,
        request_adaptation=True,
    )

    decision = policy.evaluate(
        uncertainty=0.7,
        information_gain=0.7,
        risk=0.2,
        feedback_decision=feedback,
    )

    assert decision.feedback_applied
    assert decision.adaptation_requested


def test_weak_feedback_has_no_effect_when_not_applied():
    policy = AdaptiveDecisionPolicy()

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=0.80,
        apply=False,
    )

    base = policy.evaluate(
        uncertainty=0.4,
        information_gain=0.4,
        risk=0.4,
    )

    adaptive = policy.evaluate(
        uncertainty=0.4,
        information_gain=0.4,
        risk=0.4,
        feedback_decision=feedback,
    )

    assert adaptive.mode == base.mode
    assert adaptive.explore_score == base.explore_score
    assert adaptive.exploit_score == base.exploit_score
    assert not adaptive.feedback_applied


def test_score_modes_exposes_base_and_feedback_components():
    policy = AdaptiveDecisionPolicy()

    feedback = FeedbackDecision(
        mode_bias="exploit",
        bias_strength=0.20,
        apply=True,
    )

    explore, exploit = policy.score_modes(
        uncertainty=0.2,
        information_gain=0.2,
        risk=0.2,
        feedback_decision=feedback,
    )

    assert exploit.feedback_bonus == 0.20
    assert explore.feedback_bonus == 0.0

    assert exploit.final_score > exploit.base_score
    assert explore.final_score == explore.base_score


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        AdaptiveDecisionPolicy(
            exploration_weight=-0.1,
        )

    with pytest.raises(ValueError):
        AdaptiveDecisionPolicy(
            uncertainty_weight=-0.1,
        )

    with pytest.raises(ValueError):
        AdaptiveDecisionPolicy(
            risk_weight=-0.1,
        )

    with pytest.raises(ValueError):
        AdaptiveDecisionPolicy(
            exploitation_weight=-0.1,
        )

    with pytest.raises(ValueError):
        AdaptiveDecisionPolicy(
            exploitation_stability_weight=-0.1,
        )

    with pytest.raises(ValueError):
        AdaptiveDecisionPolicy(
            maximum_feedback_bonus=1.1,
        )
