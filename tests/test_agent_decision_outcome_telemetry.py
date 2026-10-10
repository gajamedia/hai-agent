from types import SimpleNamespace

from agent.core.agent import Agent
from agent.cognition.agent_decision import DecisionMode


def test_runtime_telemetry_separates_execution_from_goal_achievement():
    agent = Agent(name="agent-separated-outcomes")
    decision = SimpleNamespace(mode=DecisionMode.EXPLORE)

    agent._record_decision_capability_telemetry(
        decision=decision,
        success=True,
        decision_making_success=True,
        execution_success=True,
        goal_achievement_success=False,
        mode_success=True,
    )

    evidence = agent.get_capability_telemetry_evidence()

    assert evidence["execution"] == {
        "attempts": 1,
        "successes": 1,
        "failures": 0,
    }
    assert evidence["goal_achievement"] == {
        "attempts": 1,
        "successes": 0,
        "failures": 1,
    }
    assert evidence["decision_making"] == {
        "attempts": 1,
        "successes": 1,
        "failures": 0,
    }
    assert evidence["exploration"] == {
        "attempts": 1,
        "successes": 1,
        "failures": 0,
    }

    events = agent.get_capability_telemetry_events()
    decision_event = next(
        event for event in events if event.capability == "decision_making"
    )
    assert decision_event.success is True


def test_legacy_telemetry_call_keeps_existing_behavior():
    agent = Agent(name="agent-legacy-telemetry")
    decision = SimpleNamespace(mode=DecisionMode.EXPLOIT)

    agent._record_decision_capability_telemetry(
        decision=decision,
        success=True,
    )

    evidence = agent.get_capability_telemetry_evidence()
    for capability in (
        "decision_making",
        "execution",
        "goal_achievement",
        "exploitation",
    ):
        assert evidence[capability] == {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        }
