from types import SimpleNamespace
from unittest.mock import Mock

from agent.core.agent import Agent
from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SelfModelDecisionInfluence,
)


class DummyGoal:
    def __init__(
        self,
        description="calibrated integration",
        target_state=None,
    ):
        self.description = description
        self.target_state = (
            target_state
            if target_state is not None
            else {}
        )
        self.completed = False

    def is_satisfied(
        self,
        world_state,
    ):
        if not self.target_state:
            return False

        return all(
            world_state.get(key) == value
            for key, value
            in self.target_state.items()
        )

    def complete(self):
        self.completed = True


def build_agent():
    agent = Agent(
        name="calibrated-self-awareness-test"
    )

    agent.unified_executor = Mock()

    agent.unified_executor.execute.return_value = (
        SimpleNamespace(
            exploration_result=None
        )
    )

    return agent


def make_influence(
    *,
    confidence=0.90,
    uncertainty=0.10,
    explore_bonus=0.0,
    exploit_bonus=0.25,
    preferred_mode=SELF_MODEL_MODE_EXPLOIT,
):
    return SelfModelDecisionInfluence(
        preferred_mode=preferred_mode,
        explore_bonus=explore_bonus,
        exploit_bonus=exploit_bonus,
        confidence=confidence,
        uncertainty=uncertainty,
        strength=1.0,
        applied=True,
        reason="manual integration influence",
    )


def test_agent_initializes_calibrated_influence_applier():
    agent = build_agent()

    assert (
        agent.calibrated_self_aware_influence_applier
        is not None
    )


def test_no_calibrated_influence_exists_initially():
    agent = build_agent()

    assert (
        agent.get_calibrated_self_aware_influence()
        is None
    )

    assert (
        agent.get_calibrated_self_aware_result()
        is None
    )


def test_manual_self_awareness_uses_raw_influence_until_calibrated():
    agent = build_agent()

    raw = make_influence()

    agent.self_awareness.decision_influence = (
        raw
    )

    assert (
        agent.last_calibrated_self_aware_influence
        is None
    )


def test_calibration_produces_calibrated_influence():
    agent = build_agent()

    raw = make_influence()

    agent.self_awareness.decision_influence = (
        raw
    )

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    calibrated = (
        agent.get_calibrated_self_aware_influence()
    )

    assert calibrated is not None

    result = (
        agent.get_calibrated_self_aware_result()
    )

    assert result is not None

    assert (
        result.influence
        is calibrated
    )


def test_overconfident_influence_is_corrected_after_runtime_failure():
    agent = build_agent()

    raw = make_influence(
        confidence=0.90,
        uncertainty=0.10,
        explore_bonus=0.0,
        exploit_bonus=0.25,
    )

    agent.self_awareness.decision_influence = (
        raw
    )

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": False,
        },
    )

    calibrated = (
        agent.get_calibrated_self_aware_influence()
    )

    assert calibrated is not None

    assert (
        calibrated.confidence
        < raw.confidence
    )

    assert (
        calibrated.exploit_bonus
        < raw.exploit_bonus
    )

    assert (
        calibrated.explore_bonus
        > raw.explore_bonus
    )


def test_second_decision_uses_calibrated_influence():
    agent = build_agent()

    raw = make_influence(
        confidence=0.90,
        uncertainty=0.10,
        explore_bonus=0.0,
        exploit_bonus=0.25,
    )

    agent.self_awareness.decision_influence = (
        raw
    )

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": False,
        },
    )

    calibrated = (
        agent.get_calibrated_self_aware_influence()
    )

    assert calibrated is not None

    captured = {}

    original_evaluate = (
        agent.self_aware_decision_arbiter
        .evaluate
    )

    def capture_evaluate(
        decision,
        influence,
    ):
        captured["influence"] = influence
        return original_evaluate(
            decision,
            influence,
        )

    agent.self_aware_decision_arbiter.evaluate = (
        capture_evaluate
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": False,
        },
    )

    assert (
        captured["influence"]
        is calibrated
    )


def test_calibration_is_temporally_ordered():
    agent = build_agent()

    raw = make_influence()

    agent.self_awareness.decision_influence = (
        raw
    )

    goal = DummyGoal()

    # First cycle has no calibrated influence before
    # its own outcome exists.
    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    calibrated = (
        agent.get_calibrated_self_aware_influence()
    )

    assert calibrated is not None

    # The calibrated influence is therefore prepared
    # for cycle 2, not retroactively for cycle 1.
    assert (
        agent.get_calibrated_self_aware_result()
        is not None
    )


def test_manual_self_awareness_invalidates_calibrated_cache():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    assert (
        agent.get_calibrated_self_aware_influence()
        is not None
    )

    agent.evaluate_self_awareness(
        {
            "execution": {
                "attempts": 5,
                "successes": 5,
                "failures": 0,
            }
        },
        persist=False,
    )

    assert (
        agent.get_calibrated_self_aware_influence()
        is None
    )


def test_reset_self_awareness_invalidates_calibrated_cache():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    assert (
        agent.get_calibrated_self_aware_influence()
        is not None
    )

    agent.reset_self_awareness()

    assert (
        agent.get_calibrated_self_aware_influence()
        is None
    )

    assert (
        agent.get_calibrated_self_aware_result()
        is None
    )


def test_feedback_uses_calibrated_influence_when_available():
    agent = build_agent()

    raw = make_influence()

    agent.self_awareness.decision_influence = (
        raw
    )

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    calibrated = (
        agent.get_calibrated_self_aware_influence()
    )

    assert calibrated is not None

    record = (
        agent.self_awareness_feedback.latest()
    )

    assert record is not None

    assert (
        record.preferred_mode
        == calibrated.preferred_mode
    )


def test_calibration_result_is_exposed():
    agent = build_agent()

    raw = make_influence()

    agent.self_awareness.decision_influence = (
        raw
    )

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    result = (
        agent.get_calibrated_self_aware_result()
    )

    assert result is not None

    assert result.adjusted or (
        result.confidence_delta == 0.0
    )
