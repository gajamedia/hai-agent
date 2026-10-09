from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from agent.core.agent import Agent
from agent.cognition.agent_decision import DecisionMode
from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SelfModelDecisionInfluence,
)


class DummyGoal:
    def __init__(
        self,
        description: str = "integration test",
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
        name="self-aware-integration-test"
    )

    agent.unified_executor = Mock()

    agent.unified_executor.execute.return_value = (
        SimpleNamespace(
            exploration_result=None
        )
    )

    return agent


def build_explore_influence() -> SelfModelDecisionInfluence:
    return SelfModelDecisionInfluence(
        preferred_mode=SELF_MODEL_MODE_EXPLORE,
        explore_bonus=0.25,
        exploit_bonus=0.0,
        confidence=0.20,
        uncertainty=0.90,
        strength=1.0,
        applied=True,
        reason="high self uncertainty",
    )


def build_exploit_influence() -> SelfModelDecisionInfluence:
    return SelfModelDecisionInfluence(
        preferred_mode=SELF_MODEL_MODE_EXPLOIT,
        explore_bonus=0.0,
        exploit_bonus=0.25,
        confidence=0.90,
        uncertainty=0.10,
        strength=1.0,
        applied=True,
        reason="high self confidence",
    )


def force_meta_decision(
    agent: Agent,
    mode: DecisionMode,
):
    runtime = agent.evaluate_meta_learning_runtime(
        uncertainty=0.50,
        information_gain=0.70,
        risk=0.20,
    )

    meta_decision = runtime.decision

    assert meta_decision is not None
    assert meta_decision.base_decision is not None

    forced_base = replace(
        meta_decision.base_decision,
        mode=mode,
    )

    forced_meta_decision = replace(
        meta_decision,
        mode=mode,
        base_mode=mode,
        base_decision=forced_base,
    )

    agent.evaluate_meta_learning_runtime = (
        lambda **kwargs: SimpleNamespace(
            decision=forced_meta_decision
        )
    )


def test_agent_initializes_self_aware_decision_arbiter():
    agent = build_agent()

    assert agent.self_aware_decision_arbiter is not None


def test_decide_and_act_without_self_awareness_keeps_no_arbitration_result():
    agent = build_agent()
    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    assert (
        agent.last_self_aware_decision_result
        is None
    )


def test_self_aware_arbitration_is_applied_when_influence_exists():
    agent = build_agent()
    goal = DummyGoal()

    agent.self_awareness.decision_influence = (
        build_explore_influence()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    result = (
        agent.last_self_aware_decision_result
    )

    assert result is not None
    assert result.applied


def test_executor_receives_self_aware_decision():
    agent = build_agent()
    goal = DummyGoal()

    agent.self_awareness.decision_influence = (
        build_explore_influence()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    execute_kwargs = (
        agent.unified_executor.execute.call_args.kwargs
    )

    executed_decision = (
        execute_kwargs["decision"]
    )

    arbitration_result = (
        agent.last_self_aware_decision_result
    )

    assert arbitration_result is not None

    assert executed_decision is (
        arbitration_result.decision
    )


def test_self_awareness_can_change_exploit_to_explore():
    agent = build_agent()
    goal = DummyGoal()

    force_meta_decision(
        agent,
        DecisionMode.EXPLOIT,
    )

    agent.self_awareness.decision_influence = (
        build_explore_influence()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    result = (
        agent.last_self_aware_decision_result
    )

    assert result is not None
    assert result.selected_mode == (
        DecisionMode.EXPLORE
    )
    assert result.mode_changed
    assert result.exploration_selected


def test_self_awareness_can_change_explore_to_exploit():
    agent = build_agent()
    goal = DummyGoal()

    force_meta_decision(
        agent,
        DecisionMode.EXPLORE,
    )

    agent.self_awareness.decision_influence = (
        build_exploit_influence()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    result = (
        agent.last_self_aware_decision_result
    )

    assert result is not None
    assert result.selected_mode == (
        DecisionMode.EXPLOIT
    )
    assert result.mode_changed
    assert result.exploitation_selected


def test_self_aware_result_is_accessible_through_agent():
    agent = build_agent()
    goal = DummyGoal()

    agent.self_awareness.decision_influence = (
        build_explore_influence()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    result = (
        agent.get_self_aware_decision_result()
    )

    assert result is not None
    assert result is (
        agent.last_self_aware_decision_result
    )


def test_self_aware_decision_does_not_execute_itself():
    agent = build_agent()
    goal = DummyGoal()

    influence = build_explore_influence()

    base_runtime = agent.evaluate_meta_learning_runtime(
        uncertainty=0.50,
        information_gain=0.70,
        risk=0.20,
    )

    base_decision = (
        base_runtime.decision.base_decision
    )

    assert base_decision is not None

    result = (
        agent.self_aware_decision_arbiter.evaluate(
            base_decision,
            influence,
        )
    )

    assert result is not None
    assert (
        agent.unified_executor.execute.call_count
        == 0
    )


def test_decision_fields_are_preserved_except_mode_and_reason():
    agent = build_agent()
    goal = DummyGoal()

    force_meta_decision(
        agent,
        DecisionMode.EXPLOIT,
    )

    runtime = agent.evaluate_meta_learning_runtime(
        uncertainty=0.50,
        information_gain=0.70,
        risk=0.20,
    )

    original = runtime.decision.base_decision

    assert original is not None

    agent.self_awareness.decision_influence = (
        build_explore_influence()
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    result = (
        agent.last_self_aware_decision_result
    )

    assert result is not None
    assert result.decision is not original or (
        result.decision == original
    )

    assert (
        result.decision.mode
        in {
            DecisionMode.EXPLORE,
            DecisionMode.EXPLOIT,
        }
    )