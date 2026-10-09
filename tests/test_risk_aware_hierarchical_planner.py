from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.cognition.goal_decomposer import (
    GoalDecomposer,
)
from agent.cognition.hierarchical_planner import (
    HierarchicalPlanner,
)
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)
from agent.learning.strategy import Strategy


def create_actions():

    return [
        Action(
            name="take_key",
            description="Take the key",
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
            name="find_spare_key",
            description="Find spare key",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={
                "spare_key": "available",
            },
            effects={
                "spare_key": "found",
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


def create_planner():

    return Planner(
        create_actions()
    )


def create_hierarchical_planner():

    planner = create_planner()

    decomposer = GoalDecomposer()

    def decompose_open_door(goal):

        obtain_key = Goal(
            "obtain the key"
        )

        unlock_door = Goal(
            "unlock the door",
            target_state={
                "door": "unlocked",
            },
        )

        unlock_door.add_dependency(
            obtain_key
        )

        take_key = Goal(
            "take the key",
            target_state={
                "key": "agent",
            },
        )

        obtain_key.add_sub_goal(
            take_key
        )

        return [
            obtain_key,
            unlock_door,
        ]

    decomposer.register_rule(
        "open the door",
        decompose_open_door,
    )

    return HierarchicalPlanner(
        planner=planner,
        decomposer=decomposer,
    )


def create_reliable_strategy():

    strategy = Strategy(
        name="reliable",
        goal="open the door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    for index in range(9):
        strategy.record_success(
            experience_id=f"reliable-{index}",
            action_count=2,
        )

    strategy.record_failure(
        experience_id="reliable-failure",
        action_count=2,
    )

    return strategy


def create_risky_strategy():

    strategy = Strategy(
        name="risky",
        goal="open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
    )

    for index in range(2):
        strategy.record_success(
            experience_id=f"risky-{index}",
            action_count=2,
        )

    for index in range(8):
        strategy.record_failure(
            experience_id=f"risky-failure-{index}",
            action_count=2,
        )

    return strategy


def test_risk_aware_hierarchical_planner_prefers_lower_risk_strategy():

    hierarchical_planner = (
        create_hierarchical_planner()
    )

    planner = RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical_planner,
        risk_strategy_selector=(
            RiskAwareStrategySelector()
        ),
    )

    reliable = (
        create_reliable_strategy()
    )

    risky = (
        create_risky_strategy()
    )

    goal = Goal(
        "open the door"
    )

    plan = planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
            "spare_key": "available",
        },
        strategies=[
            risky,
            reliable,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]


def test_risk_aware_hierarchical_planner_preserves_dependencies():

    hierarchical_planner = (
        create_hierarchical_planner()
    )

    planner = RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical_planner
    )

    strategy = create_reliable_strategy()

    goal = Goal(
        "open the door"
    )

    plan = planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
        strategies=[
            strategy,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]


def test_invalid_strategy_falls_back_to_hierarchical_planner():

    hierarchical_planner = (
        create_hierarchical_planner()
    )

    planner = RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical_planner
    )

    invalid = Strategy(
        name="invalid",
        goal="open the door",
        steps=[
            "unlock_door",
        ],
    )

    for index in range(10):
        invalid.record_success(
            experience_id=f"invalid-{index}",
            action_count=1,
        )

    goal = Goal(
        "open the door"
    )

    plan = planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
        strategies=[
            invalid,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]


def test_unknown_action_strategy_falls_back():

    hierarchical_planner = (
        create_hierarchical_planner()
    )

    planner = RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical_planner
    )

    invalid = Strategy(
        name="unknown",
        goal="open the door",
        steps=[
            "unknown_action",
        ],
    )

    invalid.record_success(
        experience_id="experience-1",
        action_count=1,
    )

    goal = Goal(
        "open the door"
    )

    plan = planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
        strategies=[
            invalid,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]


def test_no_strategies_uses_hierarchical_planner():

    hierarchical_planner = (
        create_hierarchical_planner()
    )

    planner = RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical_planner
    )

    goal = Goal(
        "open the door"
    )

    plan = planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
        strategies=[],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]


def test_strategy_must_match_goal():

    hierarchical_planner = (
        create_hierarchical_planner()
    )

    planner = RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical_planner
    )

    wrong_goal = Strategy(
        name="wrong",
        goal="close the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
    )

    for index in range(10):
        wrong_goal.record_success(
            experience_id=f"wrong-{index}",
            action_count=1,
        )

    goal = Goal(
        "open the door"
    )

    plan = planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
            "spare_key": "available",
        },
        strategies=[
            wrong_goal,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]
