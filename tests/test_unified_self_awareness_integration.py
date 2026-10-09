from types import SimpleNamespace
from unittest.mock import Mock

from agent.core.agent import Agent
from agent.cognition.agent_decision import DecisionMode


class DummyGoal:
    def __init__(
        self,
        description="unified evidence test",
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
        name="unified-evidence-test"
    )

    agent.unified_executor = Mock()

    agent.unified_executor.execute.return_value = (
        SimpleNamespace(
            exploration_result=None
        )
    )

    return agent


def test_agent_builds_unified_evidence():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        True,
        source="test",
    )

    result = (
        agent.build_unified_capability_evidence()
    )

    assert result.has_evidence

    assert (
        result.capabilities[
            "execution"
        ] == {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        }
    )


def test_unified_evidence_contains_both_sources():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        True,
        source="test",
    )

    episodes = [
        SimpleNamespace(
            goal="Planning",
            action=None,
            result={
                "success": True,
            },
        )
    ]

    agent.episodic_memory.save_all = Mock()

    for episode in episodes:
        agent.episodic_memory.add(
            episode
        )

    result = (
        agent.build_unified_capability_evidence()
    )

    assert (
        "execution"
        in result.capabilities
    )

    assert (
        "planning"
        in result.capabilities
    )


def test_telemetry_has_priority_over_experience():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        False,
        source="runtime",
    )

    episode = SimpleNamespace(
        goal="execution",
        action=None,
        result={
            "success": True,
        },
    )

    agent.episodic_memory.add(
        episode
    )

    result = (
        agent.build_unified_capability_evidence()
    )

    assert (
        result.capabilities[
            "execution"
        ]["failures"]
        == 1
    )

    assert (
        result.capabilities[
            "execution"
        ]["successes"]
        == 0
    )

    assert result.overlap_count == 1


def test_unified_evidence_can_update_self_awareness():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        True,
        source="runtime",
    )

    snapshot = (
        agent.update_self_awareness_from_all_evidence(
            persist=False
        )
    )

    assert snapshot is not None

    assert (
        agent.self_awareness.decision_influence
        is not None
    )


def test_runtime_decision_updates_self_awareness_for_next_cycle():
    agent = build_agent()

    goal = DummyGoal()

    assert (
        agent.self_awareness.decision_influence
        is None
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    assert (
        agent.self_awareness.decision_influence
        is not None
    )


def test_second_decision_can_use_updated_self_awareness():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    arbitration = (
        agent.get_self_aware_decision_result()
    )

    assert arbitration is not None


def test_unified_evidence_is_exposed_by_agent():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        True,
    )

    result = (
        agent.build_unified_capability_evidence()
    )

    assert (
        agent.get_unified_capability_evidence()
        is result
    )


def test_unified_evidence_does_not_double_count_overlap():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        True,
    )

    episode = SimpleNamespace(
        goal="execution",
        action=None,
        result={
            "success": True,
        },
    )

    agent.episodic_memory.add(
        episode
    )

    result = (
        agent.build_unified_capability_evidence()
    )

    assert (
        result.capabilities[
            "execution"
        ]["attempts"]
        == 1
    )


def test_runtime_telemetry_produces_multiple_capabilities():
    agent = build_agent()

    decision = SimpleNamespace(
        mode=DecisionMode.EXPLORE
    )

    agent._record_decision_capability_telemetry(
        decision=decision,
        success=True,
    )

    result = (
        agent.build_unified_capability_evidence()
    )

    assert {
        "decision_making",
        "execution",
        "goal_achievement",
        "exploration",
    }.issubset(
        result.capabilities.keys()
    )


def test_self_awareness_update_uses_unified_capabilities():
    agent = build_agent()

    agent.record_capability_telemetry(
        "execution",
        False,
    )

    agent.record_capability_telemetry(
        "planning",
        True,
    )

    snapshot = (
        agent.update_self_awareness_from_all_evidence(
            persist=False
        )
    )

    assert snapshot is not None

    assert (
        agent.self_awareness.capability_summary
        is not None
    )


def test_update_without_any_evidence_returns_none():
    agent = build_agent()

    result = (
        agent.update_self_awareness_from_all_evidence(
            persist=False
        )
    )

    assert result is None
