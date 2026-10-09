from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal

from agent.cognition.agent_decision import (
    AgentDecision,
    DecisionMode,
)

from agent.cognition.active_explorer import (
    ActiveExplorer,
)

from agent.cognition.exploration_loop import (
    ActiveExplorationLoop,
)

from agent.cognition.hierarchical_planner import (
    HierarchicalPlanner,
)

from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)

from agent.cognition.goal_decomposer import (
    GoalDecomposer,
)

from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)

from agent.cognition.unified_executor import (
    UnifiedDecisionExecutor,
)

from agent.learning.strategy import Strategy
from agent.cognition.planner import Planner
from agent.world.room import RoomEnvironment


def create_actions():

    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock door",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={
                "door": "unlocked",
            },
        ),
    ]


def create_agent():

    agent = Agent(
        name="unified-agent"
    )

    agent.register_actions(
        create_actions()
    )

    return agent


def create_hierarchical_planner():

    planner = Planner(
        create_actions()
    )

    decomposer = GoalDecomposer()

    def decompose_open_door(
        goal,
    ):

        return [
            Goal(
                "unlock the door",
                target_state={
                    "door": "unlocked",
                },
            )
        ]

    decomposer.register_rule(
        "open the door",
        decompose_open_door,
    )

    hierarchical = HierarchicalPlanner(
        planner=planner,
        decomposer=decomposer,
    )

    return RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical,
        risk_strategy_selector=(
            RiskAwareStrategySelector()
        ),
    )


def test_explore_decision_is_dispatched():

    agent = create_agent()

    environment = RoomEnvironment()

    explorer = ActiveExplorer()

    executor = UnifiedDecisionExecutor(
        explorer=explorer,
    )

    decision = AgentDecision(
        mode=DecisionMode.EXPLORE,
        reason="Need more knowledge.",
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.2,
    )

    result = executor.execute(
        decision,
        agent=agent,
        environment=environment,
        actions=[
            create_actions()[0]
        ],
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        exploration_risk=0.2,
        experience_id="unified-explore-1",
        goal=Goal(
            "open the box"
        ),
    )

    assert (
        result.mode
        == DecisionMode.EXPLORE
    )

    assert (
        result.exploration_result
        is not None
    )

    assert (
        result.exploration_result
        .action.name
        == "open_box"
    )


def test_exploit_decision_is_dispatched():

    planner = create_hierarchical_planner()

    executor = UnifiedDecisionExecutor(
        hierarchical_planner=planner,
    )

    decision = AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason="Knowledge is sufficient.",
        uncertainty=0.1,
        expected_information_gain=0.1,
        risk=0.1,
    )

    goal = Goal(
        "open the door",
    )

    result = executor.execute(
        decision,
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "agent",
        },
        strategies=[],
    )

    assert (
        result.mode
        == DecisionMode.EXPLOIT
    )

    assert result.plan is not None

    assert [
        action.name
        for action in result.plan
    ] == [
        "unlock_door",
    ]


def test_explore_result_is_returned_unchanged():

    agent = create_agent()

    environment = RoomEnvironment()

    explorer = ActiveExplorer()

    executor = UnifiedDecisionExecutor(
        explorer=explorer,
    )

    decision = AgentDecision(
        mode=DecisionMode.EXPLORE,
        reason="Need information.",
        uncertainty=1.0,
        expected_information_gain=1.0,
        risk=0.1,
    )

    result = executor.execute(
        decision,
        agent=agent,
        environment=environment,
        actions=[
            create_actions()[0]
        ],
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        exploration_risk=0.1,
        experience_id="unified-explore-2",
        goal=Goal(
            "open the box"
        ),
    )

    assert (
        result.exploration_result
        .execution
        is not None
    )

    assert (
        result.exploration_result
        .execution
        .result["success"]
        is True
    )


def test_exploit_returns_empty_plan_when_goal_is_unreachable():

    planner = create_hierarchical_planner()

    executor = UnifiedDecisionExecutor(
        hierarchical_planner=planner,
    )

    decision = AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason="Exploit existing knowledge.",
        uncertainty=0.1,
        expected_information_gain=0.1,
        risk=0.1,
    )

    goal = Goal(
        "open the door",
    )

    result = executor.execute(
        decision,
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "missing",
        },
    )

    assert result.plan == []


def test_none_decision_is_rejected():

    executor = UnifiedDecisionExecutor()

    try:
        executor.execute(None)
    except ValueError as exc:
        assert str(exc) == (
            "decision cannot be None."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_explore_requires_explorer():

    executor = UnifiedDecisionExecutor()

    decision = AgentDecision(
        mode=DecisionMode.EXPLORE,
        reason="Explore.",
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.1,
    )

    try:
        executor.execute(
            decision,
            agent=create_agent(),
            environment=RoomEnvironment(),
            actions=create_actions(),
            goal=Goal("open the box"),
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "ActiveExplorer is not configured."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_exploit_requires_hierarchical_planner():

    executor = UnifiedDecisionExecutor()

    decision = AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason="Exploit.",
        uncertainty=0.1,
        expected_information_gain=0.1,
        risk=0.1,
    )

    try:
        executor.execute(
            decision,
            goal=Goal("open the door"),
            world_state={
                "door": "locked",
            },
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "RiskAwareHierarchicalPlanner "
            "is not configured."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_unsupported_decision_mode_is_rejected():

    executor = UnifiedDecisionExecutor()

    class FakeDecision:

        mode = "unknown"

    try:
        executor.execute(
            FakeDecision()
        )
    except ValueError as exc:
        assert "Unsupported decision mode" in str(
            exc
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )
