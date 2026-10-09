import pytest

from agent.learning.meta_learning import (
    MetaLearningProfile,
)
from agent.learning.meta_learning_feedback import (
    MetaLearningFeedbackEngine,
)
from agent.learning.meta_learning_feedback_policy import (
    MetaLearningFeedbackPolicy,
)
from agent.learning.meta_learning_runtime import (
    MetaLearningRuntime,
    MetaLearningRuntimeResult,
    RuntimeDecisionContext,
)


class FakeDecision:
    def __init__(
        self,
        mode="exploit",
        reason="base",
    ):
        self.mode = mode
        self.reason = reason


class FakeDecisionMaker:
    def __init__(
        self,
        mode="exploit",
    ):
        self.mode = mode
        self.calls = []

    def decide(
        self,
        uncertainty,
        information_gain,
        risk,
    ):
        self.calls.append(
            {
                "uncertainty": uncertainty,
                "information_gain": information_gain,
                "risk": risk,
            }
        )

        return FakeDecision(
            mode=self.mode,
            reason="base",
        )


def create_profile_with_decisions(
    exploration_successes=0,
    exploration_failures=0,
    exploitation_successes=0,
    exploitation_failures=0,
):
    profile = MetaLearningProfile()

    for index in range(
        exploration_successes
    ):
        profile.record_exploration(
            success=True,
            event_id=f"explore-success-{index}",
        )

    for index in range(
        exploration_failures
    ):
        profile.record_exploration(
            success=False,
            event_id=f"explore-failure-{index}",
        )

    for index in range(
        exploitation_successes
    ):
        profile.record_exploitation(
            success=True,
            event_id=f"exploit-success-{index}",
        )

    for index in range(
        exploitation_failures
    ):
        profile.record_exploitation(
            success=False,
            event_id=f"exploit-failure-{index}",
        )

    return profile


def test_runtime_requires_profile():
    base = FakeDecisionMaker()

    with pytest.raises(
        ValueError,
        match="profile cannot be None",
    ):
        MetaLearningRuntime(
            profile=None,
            base_decision_maker=base,
        )


def test_runtime_requires_base_decision_maker():
    profile = MetaLearningProfile()

    with pytest.raises(
        ValueError,
        match="base_decision_maker cannot be None",
    ):
        MetaLearningRuntime(
            profile=profile,
            base_decision_maker=None,
        )


def test_runtime_evaluates_context():
    profile = MetaLearningProfile()

    base = FakeDecisionMaker()

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
    )

    context = RuntimeDecisionContext(
        uncertainty=0.2,
        information_gain=0.2,
        risk=0.2,
    )

    result = runtime.evaluate(
        context
    )

    assert isinstance(
        result,
        MetaLearningRuntimeResult,
    )

    assert result.feedback is not None
    assert result.feedback_decision is not None
    assert result.decision is not None

    assert len(base.calls) == 1

    assert base.calls[0] == {
        "uncertainty": 0.2,
        "information_gain": 0.2,
        "risk": 0.2,
    }


def test_runtime_starts_with_neutral_feedback():
    profile = MetaLearningProfile()

    base = FakeDecisionMaker(
        mode="exploit"
    )

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
    )

    result = runtime.evaluate_from_values(
        uncertainty=0.2,
        information_gain=0.2,
        risk=0.2,
    )

    assert not result.feedback_applied
    assert result.mode == "exploit"


def test_runtime_can_use_strong_exploitation_feedback():
    profile = create_profile_with_decisions(
        exploitation_successes=10,
        exploration_failures=3,
        exploration_successes=1,
    )

    base = FakeDecisionMaker(
        mode="exploit"
    )

    engine = MetaLearningFeedbackEngine(
        min_decisions=3,
    )

    policy = MetaLearningFeedbackPolicy(
        minimum_confidence=0.0,
    )

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
        feedback_engine=engine,
        feedback_policy=policy,
    )

    result = runtime.evaluate_from_values(
        uncertainty=0.5,
        information_gain=0.5,
        risk=0.5,
    )

    assert result.feedback is not None
    assert result.feedback_decision is not None
    assert result.decision is not None

    assert result.decision.feedback_applied
    assert (
        result.feedback_decision.mode_bias
        == "exploit"
    )


def test_runtime_can_use_strong_exploration_feedback():
    profile = create_profile_with_decisions(
        exploration_successes=10,
        exploitation_failures=3,
        exploitation_successes=1,
    )

    base = FakeDecisionMaker(
        mode="exploit"
    )

    engine = MetaLearningFeedbackEngine(
        min_decisions=3,
    )

    policy = MetaLearningFeedbackPolicy(
        minimum_confidence=0.0,
    )

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
        feedback_engine=engine,
        feedback_policy=policy,
    )

    result = runtime.evaluate_from_values(
        uncertainty=0.5,
        information_gain=0.5,
        risk=0.5,
    )

    assert result.feedback_decision.mode_bias == (
        "explore"
    )

    assert result.feedback_applied


def test_runtime_preserves_forbidden_mode_constraint():
    profile = create_profile_with_decisions(
        exploration_successes=10,
        exploitation_failures=3,
        exploitation_successes=1,
    )

    base = FakeDecisionMaker(
        mode="exploit"
    )

    engine = MetaLearningFeedbackEngine(
        min_decisions=3,
    )

    policy = MetaLearningFeedbackPolicy(
        minimum_confidence=0.0,
    )

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
        feedback_engine=engine,
        feedback_policy=policy,
    )

    result = runtime.evaluate_from_values(
        uncertainty=1.0,
        information_gain=1.0,
        risk=0.0,
        explore_allowed=False,
        exploit_allowed=True,
    )

    assert result.mode == "exploit"


def test_runtime_preserves_adaptation_request():
    profile = create_profile_with_decisions(
        exploration_successes=1,
        exploration_failures=5,
        exploitation_successes=1,
        exploitation_failures=5,
    )

    # Add successful adaptation events so the
    # profile has sufficient adaptation evidence.
    for index in range(4):
        profile.record_evolution(
            success=False,
            event_id=f"evolution-failure-{index}",
        )

    base = FakeDecisionMaker()

    engine = MetaLearningFeedbackEngine(
        min_decisions=3,
        min_adaptation_events=2,
    )

    policy = MetaLearningFeedbackPolicy(
        minimum_confidence=0.0,
        minimum_adaptation_pressure=0.10,
    )

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
        feedback_engine=engine,
        feedback_policy=policy,
    )

    result = runtime.evaluate_from_values(
        uncertainty=0.5,
        information_gain=0.5,
        risk=0.5,
    )

    assert result.adaptation_requested


def test_runtime_can_use_custom_components():
    profile = MetaLearningProfile()

    base = FakeDecisionMaker(
        mode="exploit"
    )

    engine = MetaLearningFeedbackEngine(
        min_decisions=10,
    )

    policy = MetaLearningFeedbackPolicy(
        minimum_confidence=0.1,
    )

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
        feedback_engine=engine,
        feedback_policy=policy,
    )

    result = runtime.evaluate_from_values(
        uncertainty=0.3,
        information_gain=0.4,
        risk=0.2,
    )

    assert result.mode == "exploit"
    assert not result.feedback_applied


def test_runtime_context_none_is_rejected():
    profile = MetaLearningProfile()
    base = FakeDecisionMaker()

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
    )

    with pytest.raises(
        ValueError,
        match="context cannot be None",
    ):
        runtime.evaluate(None)


def test_evaluate_from_values_delegates_to_context():
    profile = MetaLearningProfile()
    base = FakeDecisionMaker()

    runtime = MetaLearningRuntime(
        profile=profile,
        base_decision_maker=base,
    )

    result = runtime.evaluate_from_values(
        uncertainty=0.6,
        information_gain=0.7,
        risk=0.2,
    )

    assert result.decision is not None

    assert base.calls[-1] == {
        "uncertainty": 0.6,
        "information_gain": 0.7,
        "risk": 0.2,
    }
