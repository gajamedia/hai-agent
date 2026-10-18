from __future__ import annotations

from dataclasses import dataclass

from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.active_explorer import ActiveExplorer
from agent.cognition.agent_decision import AgentDecision, DecisionMode
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.cognition.planner import Planner
from agent.cognition.risk_aware_hierarchical_planner import RiskAwareHierarchicalPlanner
from agent.cognition.risk_strategy_selector import RiskAwareStrategySelector
from agent.cognition.unified_executor import UnifiedDecisionExecutor
from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.planning import LLMPlan, LLMPlanStep
from agent.world.room import RoomEnvironment


def create_actions():
    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda **kwargs: {"success": True},
            preconditions={"box": "closed"},
            effects={"box": "open"},
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=lambda **kwargs: {"success": True},
            preconditions={"box": "open", "key": "inside_box"},
            effects={"key": "agent"},
        ),
        Action(
            name="unlock_door",
            description="Unlock door",
            handler=lambda **kwargs: {"success": True},
            preconditions={"key": "agent", "door": "locked"},
            effects={"door": "unlocked"},
        ),
    ]


def create_agent():
    agent = Agent(name="step34-assurance-agent")
    agent.register_actions(create_actions())
    return agent


def create_hierarchical_planner():
    planner = Planner(create_actions())
    decomposer = GoalDecomposer()
    decomposer.register_rule(
        "open the door",
        lambda goal: [
            Goal("unlock the door", target_state={"door": "unlocked"})
        ],
    )
    hierarchical = HierarchicalPlanner(planner=planner, decomposer=decomposer)
    return RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical,
        risk_strategy_selector=RiskAwareStrategySelector(),
    )


@dataclass
class FakeResult:
    success: bool
    result: object


class FakeAPI:
    def __init__(self, actions=("open_box", "take_key")):
        self.calls = []
        self.agent = type(
            "AgentStub", (), {"actions": {name: object() for name in actions}}
        )()
        self.results = []

    def execute(self, request):
        self.calls.append(request)
        if self.results:
            return self.results.pop(0)
        return FakeResult(True, {})


def make_llm_plan(actions=("open_box", "take_key")):
    return LLMPlan(
        goal="open the box",
        rationale="Complete the goal in order.",
        steps=tuple(
            LLMPlanStep(
                index=index,
                action=action,
                purpose=f"Run {action}.",
                expected_observation=f"{action} completed",
                success_criteria=f"{action} completed",
                risk="low",
            )
            for index, action in enumerate(actions, start=1)
        ),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.9,
        provider="test",
        model="step34-test",
    )


def test_exploit_plan_is_not_execution_success_before_actions_run():
    executor = UnifiedDecisionExecutor(
        hierarchical_planner=create_hierarchical_planner()
    )
    decision = AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason="Use known actions.",
        uncertainty=0.1,
        expected_information_gain=0.1,
        risk=0.1,
    )

    result = executor.execute(
        decision,
        goal=Goal("open the door", target_state={"door": "unlocked"}),
        world_state={"door": "locked", "box": "open", "key": "agent"},
        strategies=[],
    )

    assert result.plan is not None
    assert result.success is False


def test_exploration_success_does_not_claim_goal_satisfied():
    agent = create_agent()
    environment = RoomEnvironment()
    executor = UnifiedDecisionExecutor(explorer=ActiveExplorer())
    decision = AgentDecision(
        mode=DecisionMode.EXPLORE,
        reason="Gather information.",
        uncertainty=1.0,
        expected_information_gain=1.0,
        risk=0.1,
    )

    result = executor.execute(
        decision,
        agent=agent,
        environment=environment,
        actions=[agent.actions["open_box"]],
        information_gain={"open_box": 1.0},
        action_risk={"open_box": 0.1},
        exploration_risk=0.1,
        experience_id="step34-explore-contract",
        goal=Goal("unlock the door", target_state={"door": "unlocked"}),
    )

    assert result.success is True
    assert result.exploration_result is not None
    assert result.exploration_result.success is True
    assert result.exploration_result.goal_satisfied is False


def test_hierarchical_plan_verifies_goal_and_closes_one_experience():
    agent = create_agent()
    environment = RoomEnvironment()
    executor = UnifiedDecisionExecutor(
        explorer=ActiveExplorer(),
        hierarchical_planner=create_hierarchical_planner(),
    )
    decision = AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason="Execute the plan.",
        uncertainty=0.1,
        expected_information_gain=0.0,
        risk=0.1,
    )
    goal = Goal("open the door", target_state={"door": "unlocked"})
    before = agent.episodic_memory.count()

    result = executor.execute_plan(
        decision=decision,
        agent=agent,
        environment=environment,
        goal=goal,
        actions=create_actions(),
        world_state=environment.observe(),
        experience_id="step34-hierarchical-once",
    )

    assert result.success is True
    assert result.goal_satisfied is True
    assert environment.observe()["door"] == "unlocked"
    assert agent.current_experience is None
    assert agent.episodic_memory.count() == before + 3
    stored = agent.experience_recorder.store.latest()
    assert stored is not None
    assert stored.experience_id == "step34-hierarchical-once"
    assert stored.completed is True
    assert stored.failed is False
    assert stored.action_count == 3


def test_llm_plan_records_each_attempt_once_and_stops_after_failure():
    api = FakeAPI()
    api.results = [FakeResult(True, {}), FakeResult(False, {"reason": "blocked"})]
    bridge = LLMPlanSingleActionBridge.for_api(api)
    runtime = LLMPlanRuntimeExecutor.for_bridge(bridge)

    report = runtime.execute(make_llm_plan())

    assert report.stopped is True
    assert report.success is False
    assert report.completed is False
    assert report.stop_reason == "step_failed"
    assert [call.action for call in api.calls] == ["open_box", "take_key"]
    assert len(api.calls) == len(report.records) == 2
    assert report.failed_step_indices == (2,)


def test_invalid_llm_plan_is_rejected_before_any_action_executes():
    api = FakeAPI(actions=("open_box",))
    bridge = LLMPlanSingleActionBridge.for_api(api)
    runtime = LLMPlanRuntimeExecutor.for_bridge(bridge)

    try:
        runtime.execute(make_llm_plan(actions=("not_registered",)))
    except Exception as exc:
        assert "validation" in type(exc).__name__.lower()
    else:
        raise AssertionError("An unregistered action should reject the plan.")

    assert api.calls == []
