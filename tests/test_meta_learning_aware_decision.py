import pytest

from agent.learning.adaptive_decision_policy import (
    AdaptiveDecisionPolicy,
)
from agent.learning.meta_learning_aware_decision import (
    MetaLearningAwareDecision,
    MetaLearningAwareDecisionMaker,
)
from agent.learning.meta_learning_feedback_policy import (
    FeedbackDecision,
)


class FakeDecision:
    def __init__(
        self,
        mode,
        reason="base decision",
    ):
        self.mode = mode
        self.reason = reason


class FakeDecisionMaker:
    def __init__(
        self,
        mode="exploit",
        reason="base decision",
    ):
        self.mode = mode
        self.reason = reason
        self.calls = []

    def decide(
        self,
        uncertainty,
        information_gain,
        risk,
    ):
        self.calls.append(
            (
                uncertainty,
                information_gain,
                risk,
            )
        )

        return FakeDecision(
            mode=self.mode,
            reason=self.reason,
        )


def test_requires_base_decision_maker():
    with pytest.raises(
        ValueError,
        match="base_decision_maker cannot be None",
    ):
        MetaLearningAwareDecisionMaker(
            base_decision_maker=None
        )


def test_base_decision_maker_is_called():
    base = FakeDecisionMaker()

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    maker.decide(
        uncertainty=0.2,
        information_gain=0.3,
        risk=0.1,
    )

    assert base.calls == [
        (0.2, 0.3, 0.1)
    ]


def test_returns_meta_learning_aware_decision():
    base = FakeDecisionMaker(
        mode="exploit",
        reason="stable environment",
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    result = maker.decide(
        uncertainty=0.1,
        information_gain=0.1,
        risk=0.1,
    )

    assert isinstance(
        result,
        MetaLearningAwareDecision,
    )

    assert result.mode == "exploit"
    assert result.base_mode == "exploit"

    assert result.uncertainty == 0.1
    assert result.information_gain == 0.1
    assert result.risk == 0.1

    assert result.base_reason == (
        "stable environment"
    )


def test_adaptive_feedback_can_change_final_mode():
    base = FakeDecisionMaker(
        mode="exploit",
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=0.25,
        apply=True,
    )

    result = maker.decide(
        uncertainty=0.5,
        information_gain=0.5,
        risk=0.5,
        feedback_decision=feedback,
    )

    assert result.base_mode == "exploit"
    assert result.mode == "explore"
    assert result.mode_changed
    assert result.feedback_applied


def test_without_feedback_normal_policy_is_used():
    base = FakeDecisionMaker(
        mode="exploit",
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    result = maker.decide(
        uncertainty=0.9,
        information_gain=0.9,
        risk=0.1,
    )

    assert result.mode == "explore"
    assert result.base_mode == "exploit"
    assert not result.feedback_applied


def test_adaptation_request_is_preserved():
    base = FakeDecisionMaker(
        mode="exploit",
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    feedback = FeedbackDecision(
        mode_bias="exploit",
        bias_strength=0.2,
        apply=True,
        request_adaptation=True,
    )

    result = maker.decide(
        uncertainty=0.2,
        information_gain=0.2,
        risk=0.2,
        feedback_decision=feedback,
    )

    assert result.adaptation_requested


def test_exploration_constraint_is_respected():
    base = FakeDecisionMaker(
        mode="exploit",
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=0.75,
        apply=True,
    )

    result = maker.decide(
        uncertainty=1.0,
        information_gain=1.0,
        risk=0.0,
        feedback_decision=feedback,
        explore_allowed=False,
    )

    assert result.mode == "exploit"


def test_exploitation_constraint_is_respected():
    base = FakeDecisionMaker(
        mode="exploit",
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    feedback = FeedbackDecision(
        mode_bias="exploit",
        bias_strength=0.75,
        apply=True,
    )

    result = maker.decide(
        uncertainty=0.0,
        information_gain=0.0,
        risk=0.0,
        feedback_decision=feedback,
        exploit_allowed=False,
    )

    assert result.mode == "explore"


def test_custom_adaptive_policy_is_used():
    base = FakeDecisionMaker()

    policy = AdaptiveDecisionPolicy(
        maximum_feedback_bonus=0.10,
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
        adaptive_policy=policy,
    )

    feedback = FeedbackDecision(
        mode_bias="explore",
        bias_strength=1.0,
        apply=True,
    )

    result = maker.decide(
        uncertainty=0.5,
        information_gain=0.5,
        risk=0.5,
        feedback_decision=feedback,
    )

    assert result.mode in {
        "explore",
        "exploit",
    }


def test_invalid_base_mode_is_rejected():
    base = FakeDecisionMaker(
        mode="invalid",
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    with pytest.raises(
        ValueError,
        match="base decision mode",
    ):
        maker.decide(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.5,
        )


def test_base_decision_maker_without_decide_is_rejected():
    class InvalidDecisionMaker:
        pass

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=InvalidDecisionMaker(),
    )

    with pytest.raises(
        AttributeError,
        match="'decide'",
    ):
        maker.decide(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.5,
        )


def test_helper_properties():
    explore = MetaLearningAwareDecision(
        mode="explore",
        base_mode="exploit",
        uncertainty=0.8,
        information_gain=0.8,
        risk=0.2,
        base_reason="test",
        reason="adaptive",
    )

    assert explore.is_explore
    assert not explore.is_exploit
    assert explore.mode_changed

    exploit = MetaLearningAwareDecision(
        mode="exploit",
        base_mode="exploit",
        uncertainty=0.1,
        information_gain=0.1,
        risk=0.1,
        base_reason="test",
        reason="adaptive",
    )

    assert exploit.is_exploit
    assert not exploit.is_explore
    assert not exploit.mode_changed

def test_base_decision_is_preserved():
    base = FakeDecisionMaker(
        mode="exploit",
        reason="base decision",
    )

    maker = MetaLearningAwareDecisionMaker(
        base_decision_maker=base,
    )

    result = maker.decide(
        uncertainty=0.2,
        information_gain=0.2,
        risk=0.2,
    )

    assert result.base_decision is not None
    assert result.base_decision.mode == "exploit"
    assert result.base_decision.reason == "base decision"
