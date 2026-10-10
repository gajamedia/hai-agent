from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
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
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key",
            description="Find spare key",
            handler=lambda **kwargs: {"success": True},
            preconditions={"spare_key": "available"},
            effects={
                "spare_key": "found",
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock door",
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={"door": "unlocked"},
        ),
    ]


def create_hierarchical_planner():
    planner = Planner(create_actions())
    decomposer = GoalDecomposer()

    def decompose_open_door(goal):
        obtain_key = Goal("obtain the key")

        unlock_door = Goal(
            "unlock the door",
            target_state={"door": "unlocked"},
        )
        unlock_door.add_dependency(obtain_key)

        take_key = Goal(
            "take the key",
            target_state={"key": "agent"},
        )
        obtain_key.add_sub_goal(take_key)

        return [obtain_key, unlock_door]

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
        steps=["take_key", "unlock_door"],
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
        steps=["find_spare_key", "unlock_door"],
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


def create_planner():
    return RiskAwareHierarchicalPlanner(
        hierarchical_planner=create_hierarchical_planner(),
        risk_strategy_selector=RiskAwareStrategySelector(),
    )


def create_goal():
    return Goal("open the door")


def create_world_state():
    return {
        "door": "locked",
        "box": "open",
        "key": "inside_box",
        "spare_key": "available",
    }


def action_names(plan):
    return [action.name for action in plan]


def test_prefers_lower_risk_strategy():
    planner = create_planner()

    plan = planner.plan(
        goal=create_goal(),
        world_state=create_world_state(),
        strategies=[
            create_risky_strategy(),
            create_reliable_strategy(),
        ],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]


def test_preserves_dependencies():
    planner = create_planner()

    plan = planner.plan(
        goal=create_goal(),
        world_state=create_world_state(),
        strategies=[create_reliable_strategy()],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]


def test_invalid_strategy_falls_back():
    planner = create_planner()

    invalid = Strategy(
        name="invalid",
        goal="open the door",
        steps=["unlock_door"],
    )
    for index in range(10):
        invalid.record_success(
            experience_id=f"invalid-{index}",
            action_count=1,
        )

    plan = planner.plan(
        goal=create_goal(),
        world_state=create_world_state(),
        strategies=[invalid],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]


def test_unknown_action_strategy_falls_back():
    planner = create_planner()

    invalid = Strategy(
        name="unknown",
        goal="open the door",
        steps=["unknown_action"],
    )
    invalid.record_success(
        experience_id="unknown-1",
        action_count=1,
    )

    plan = planner.plan(
        goal=create_goal(),
        world_state=create_world_state(),
        strategies=[invalid],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]


def test_no_strategies_uses_hierarchical_planner():
    planner = create_planner()

    plan = planner.plan(
        goal=create_goal(),
        world_state=create_world_state(),
        strategies=[],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]


def test_strategy_must_match_goal():
    planner = create_planner()

    wrong_goal = Strategy(
        name="wrong",
        goal="close the door",
        steps=["find_spare_key", "unlock_door"],
    )
    for index in range(10):
        wrong_goal.record_success(
            experience_id=f"wrong-{index}",
            action_count=1,
        )

    plan = planner.plan(
        goal=create_goal(),
        world_state=create_world_state(),
        strategies=[wrong_goal],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]


def test_paused_strategy_is_not_selected():
    planner = create_planner()
    paused = create_reliable_strategy()
    paused.pause()

    plan = planner.plan(
        goal=create_goal(),
        world_state=create_world_state(),
        strategies=[paused],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]


def test_retired_strategy_is_not_selected():
    planner = create_planner()
    retired = create_reliable_strategy()
    retired.retire()

    plan = planner.plan(
        goal=create_goal(),
        world_state=create_world_state(),
        strategies=[retired],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]

def test_strategy_with_empty_hierarchy_target_falls_back():
    planner = create_planner()

    strategy = Strategy(
        name="empty-target",
        goal="open the door",
        steps=["take_key", "unlock_door"],
    )

    for index in range(10):
        strategy.record_success(
            experience_id=f"empty-target-{index}",
            action_count=2,
        )

    goal = Goal("open the door")

    plan = planner.plan(
        goal=goal,
        world_state=create_world_state(),
        strategies=[strategy],
    )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]

def test_empty_hierarchy_target_rejects_learned_strategy():
    planner = create_planner()

    goal = Goal("open the door")
    strategy = create_reliable_strategy()

    root = Goal("root without target")

    assert planner._validate_strategy(
        actions=[
            Action(
                name="noop",
                description="No operation",
                handler=lambda **kwargs: {"success": True},
                effects={},
            )
        ],
        root=root,
        world_state={},
    ) is False


def test_conflicting_hierarchy_target_raises_value_error():
    planner = create_planner()

    root = Goal(
        "root",
        target_state={"door": "locked"},
    )
    root.add_sub_goal(
        Goal(
            "unlock",
            target_state={"door": "unlocked"},
        )
    )

    try:
        planner._hierarchy_target_state(root)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Conflicting hierarchy targets should be rejected."
        )

def test_conflicting_strategy_target_does_not_break_fallback():
    planner = create_planner()

    original_decompose = (
        planner.hierarchical_planner.decomposer
        .decompose_recursive
    )

    def decompose_with_conflict(goal):
        root = Goal(
            goal.description,
            target_state={"door": "locked"},
        )
        root.add_sub_goal(
            Goal(
                "unlock",
                target_state={"door": "unlocked"},
            )
        )
        return root

    planner.hierarchical_planner.decomposer.decompose_recursive = (
        decompose_with_conflict
    )

    try:
        plan = planner.plan(
            goal=create_goal(),
            world_state=create_world_state(),
            strategies=[create_reliable_strategy()],
        )
    finally:
        planner.hierarchical_planner.decomposer.decompose_recursive = (
            original_decompose
        )

    assert action_names(plan) == [
        "take_key",
        "unlock_door",
    ]