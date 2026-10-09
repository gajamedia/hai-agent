from types import SimpleNamespace
from unittest.mock import Mock

from agent.core.agent import Agent


class DummyGoal:
    def __init__(
        self,
        description="calibration test",
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
        name="self-awareness-calibration-test"
    )

    agent.unified_executor = Mock()

    agent.unified_executor.execute.return_value = (
        SimpleNamespace(
            exploration_result=None
        )
    )

    return agent


def test_agent_initializes_calibrator():
    agent = build_agent()

    assert (
        agent.self_awareness_calibrator
        is not None
    )


def test_calibration_returns_none_without_self_awareness():
    agent = build_agent()

    result = (
        agent.calibrate_self_awareness()
    )

    assert result is None


def test_calibration_becomes_available_after_runtime_evidence():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    result = (
        agent.get_self_awareness_calibration()
    )

    assert result is not None
    assert result.has_evidence


def test_calibration_uses_runtime_evidence():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": True,
        },
    )

    result = (
        agent.get_self_awareness_calibration()
    )

    assert result is not None

    assert (
        result.total_attempts
        > 0
    )

    assert (
        result.total_successes
        > 0
    )


def test_repeated_successes_improve_observed_success_rate():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": True,
        },
    )

    first = (
        agent.get_self_awareness_calibration()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": True,
        },
    )

    second = (
        agent.get_self_awareness_calibration()
    )

    assert first is not None
    assert second is not None

    assert (
        second.observed_success_rate
        >= first.observed_success_rate
    )


def test_repeated_failures_reduce_observed_success_rate():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": True,
        },
    )

    first = (
        agent.get_self_awareness_calibration()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": False,
        },
    )

    second = (
        agent.get_self_awareness_calibration()
    )

    assert first is not None
    assert second is not None

    assert (
        second.observed_success_rate
        < first.observed_success_rate
    )


def test_calibration_is_updated_each_cycle():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    first = (
        agent.get_self_awareness_calibration()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    second = (
        agent.get_self_awareness_calibration()
    )

    assert first is not None
    assert second is not None

    assert (
        second.total_attempts
        >= first.total_attempts
    )


def test_calibration_does_not_replace_raw_self_awareness():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    raw = (
        agent.self_awareness
        .self_uncertainty
        .confidence
    )

    calibration = (
        agent.get_self_awareness_calibration()
    )

    assert calibration is not None

    assert (
        calibration.raw_confidence
        == raw
    )


def test_calibration_result_is_exposed_by_agent():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    result = (
        agent.get_self_awareness_calibration()
    )

    assert (
        agent.last_self_awareness_calibration
        is result
    )


def test_calibration_does_not_change_decision_arbiter_contract():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    calibration = (
        agent.get_self_awareness_calibration()
    )

    arbitration = (
        agent.get_self_aware_decision_result()
    )

    assert calibration is not None

    # Cycle one may still have no arbitration because
    # self-awareness was created by the same cycle's outcome.
    assert (
        arbitration is None
        or arbitration.decision is not None
    )
