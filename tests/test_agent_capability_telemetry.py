from types import SimpleNamespace
from unittest.mock import Mock

from agent.core.agent import Agent
from agent.cognition.agent_decision import DecisionMode


class DummyGoal:
    def __init__(
        self,
        description: str = "telemetry test",
        target_state: dict | None = None,
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
    ) -> bool:
        if not self.target_state:
            return False

        return all(
            world_state.get(key) == value
            for key, value in self.target_state.items()
        )

    def complete(self) -> None:
        self.completed = True


def build_agent() -> Agent:
    agent = Agent(
        name="capability-telemetry-agent"
    )

    agent.unified_executor = Mock()

    agent.unified_executor.execute.return_value = (
        SimpleNamespace(
            exploration_result=None
        )
    )

    return agent


def test_agent_initializes_capability_telemetry():
    agent = build_agent()

    assert (
        agent.capability_telemetry
        is not None
    )

    assert (
        agent.get_capability_telemetry_evidence()
        == {}
    )


def test_agent_can_record_manual_capability_telemetry():
    agent = build_agent()

    event = (
        agent.record_capability_telemetry(
            "Planning",
            True,
            source="planner",
        )
    )

    assert event.capability == "planning"
    assert event.success is True
    assert event.source == "planner"

    assert (
        agent.get_capability_telemetry_evidence()
        == {
            "planning": {
                "attempts": 1,
                "successes": 1,
                "failures": 0,
            }
        }
    )


def test_agent_exposes_latest_telemetry_event():
    agent = build_agent()

    event = (
        agent.record_capability_telemetry(
            "execution",
            False,
        )
    )

    assert (
        agent.last_capability_telemetry_event
        is event
    )


def test_agent_returns_telemetry_snapshot():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        True,
    )

    snapshot = (
        agent.get_capability_telemetry_snapshot()
    )

    assert snapshot.capability_count == 1
    assert snapshot.event_count == 1
    assert snapshot.total_attempts == 1
    assert snapshot.total_successes == 1
    assert snapshot.total_failures == 0


def test_agent_returns_telemetry_events():
    agent = build_agent()

    first = (
        agent.record_capability_telemetry(
            "planning",
            True,
        )
    )

    second = (
        agent.record_capability_telemetry(
            "execution",
            False,
        )
    )

    events = (
        agent.get_capability_telemetry_events()
    )

    assert events == [
        first,
        second,
    ]


def test_agent_can_reset_capability_telemetry():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        True,
    )

    agent.reset_capability_telemetry()

    assert (
        agent.get_capability_telemetry_evidence()
        == {}
    )

    assert (
        agent.last_capability_telemetry_event
        is None
    )


def test_decide_and_act_records_runtime_capabilities():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": True
        },
    )

    evidence = (
        agent.get_capability_telemetry_evidence()
    )

    assert (
        "decision_making"
        in evidence
    )

    assert (
        "execution"
        in evidence
    )

    assert (
        "goal_achievement"
        in evidence
    )


def test_decide_and_act_records_success():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": True
        },
    )

    evidence = (
        agent.get_capability_telemetry_evidence()
    )

    assert (
        evidence[
            "decision_making"
        ]["successes"]
        == 1
    )

    assert (
        evidence[
            "execution"
        ]["successes"]
        == 1
    )

    assert (
        evidence[
            "goal_achievement"
        ]["successes"]
        == 1
    )


def test_decide_and_act_records_failure():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": False
        },
    )

    evidence = (
        agent.get_capability_telemetry_evidence()
    )

    assert (
        evidence[
            "decision_making"
        ]["failures"]
        == 1
    )

    assert (
        evidence[
            "execution"
        ]["failures"]
        == 1
    )

    assert (
        evidence[
            "goal_achievement"
        ]["failures"]
        == 1
    )


def test_runtime_telemetry_preserves_multiple_decisions():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True
        }
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": True
        },
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "done": False
        },
    )

    evidence = (
        agent.get_capability_telemetry_evidence()
    )

    assert (
        evidence[
            "decision_making"
        ]
        == {
            "attempts": 2,
            "successes": 1,
            "failures": 1,
        }
    )

    assert (
        evidence[
            "execution"
        ]
        == {
            "attempts": 2,
            "successes": 1,
            "failures": 1,
        }
    )

    assert (
        evidence[
            "goal_achievement"
        ]
        == {
            "attempts": 2,
            "successes": 1,
            "failures": 1,
        }
    )


def test_explore_mode_records_exploration_telemetry():
    agent = build_agent()

    decision = SimpleNamespace(
        mode=DecisionMode.EXPLORE
    )

    agent._record_decision_capability_telemetry(
        decision=decision,
        success=True,
    )

    evidence = (
        agent.get_capability_telemetry_evidence()
    )

    assert (
        evidence["exploration"]
        == {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        }
    )


def test_exploit_mode_records_exploitation_telemetry():
    agent = build_agent()

    decision = SimpleNamespace(
        mode=DecisionMode.EXPLOIT
    )

    agent._record_decision_capability_telemetry(
        decision=decision,
        success=False,
    )

    evidence = (
        agent.get_capability_telemetry_evidence()
    )

    assert (
        evidence["exploitation"]
        == {
            "attempts": 1,
            "successes": 0,
            "failures": 1,
        }
    )


def test_decision_capability_telemetry_does_not_change_self_awareness():
    agent = build_agent()

    before = (
        agent.get_self_awareness_snapshot()
    )

    decision = SimpleNamespace(
        mode=DecisionMode.EXPLOIT
    )

    agent._record_decision_capability_telemetry(
        decision=decision,
        success=True,
    )

    after = (
        agent.get_self_awareness_snapshot()
    )

    assert before == after
