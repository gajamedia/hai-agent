from agent.core.action import Action
from agent.cognition.planner import Planner
from agent.cognition.adaptive_planner import AdaptivePlanner
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import (
    StrategySelector,
)


def create_actions():

    return [
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

    actions = create_actions()

    planner = Planner(
        actions
    )

    return planner


def create_reliable_strategy():

    strategy = Strategy(
        name="reliable",
        goal="open door",
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
        goal="open door",
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


def test_adaptive_planner_uses_risk_aware_strategy():

    planner = create_planner()

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=(
            StrategySelector()
        ),
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

    plan = adaptive_planner.plan(
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
            "spare_key": "available",
        },
        goal_state={
            "door": "unlocked",
        },
        goal="open door",
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


def test_adaptive_planner_falls_back_to_classical_planner():

    planner = create_planner()

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=(
            StrategySelector()
        ),
        risk_strategy_selector=(
            RiskAwareStrategySelector()
        ),
    )

    invalid = Strategy(
        name="invalid",
        goal="open door",
        steps=[
            "unknown_action",
        ],
    )

    invalid.record_success(
        experience_id="experience-1",
        action_count=1,
    )

    plan = adaptive_planner.plan(
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
            "spare_key": "available",
        },
        goal_state={
            "door": "unlocked",
        },
        goal="open door",
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


def test_adaptive_planner_keeps_backward_compatible_constructor():

    planner = create_planner()

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=(
            StrategySelector()
        ),
    )

    strategy = create_reliable_strategy()

    plan = adaptive_planner.plan(
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
            "spare_key": "available",
        },
        goal_state={
            "door": "unlocked",
        },
        goal="open door",
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


def test_adaptive_planner_uses_goal_matching_strategies_first():

    planner = create_planner()

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=(
            StrategySelector()
        ),
        risk_strategy_selector=(
            RiskAwareStrategySelector()
        ),
    )

    other_goal = Strategy(
        name="other",
        goal="other goal",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
    )

    for index in range(10):
        other_goal.record_success(
            experience_id=f"other-{index}",
            action_count=2,
        )

    matching = create_reliable_strategy()

    plan = adaptive_planner.plan(
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
            "spare_key": "available",
        },
        goal_state={
            "door": "unlocked",
        },
        goal="open door",
        strategies=[
            other_goal,
            matching,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]
