from agent.cognition.adaptive_planner import (
    AdaptivePlanner,
)
from agent.cognition.planner import Planner
from agent.core.action import Action
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import (
    StrategySelector,
)


def create_actions():

    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda: {},
            effects={
                "box": "open",
            },
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=lambda: {},
            preconditions={
                "box": "open",
            },
            effects={
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock door",
            handler=lambda: {},
            preconditions={
                "key": "agent",
            },
            effects={
                "door": "unlocked",
            },
        ),
        # Action(
        #     name="find_spare_key",
        #     description="Find spare key",
        #     handler=lambda: {},
        #     effects={
        #         "key": "agent",
        #     },
        # ),
        Action(
            name="find_spare_key",
            description="Find spare key",
            handler=lambda: {},
            preconditions={
                "spare_key": "available",
            },
            effects={
                "spare_key": "found",
                "key": "agent",
            },
        ),
    ]


def test_adaptive_planner_uses_learned_strategy():

    actions = create_actions()

    planner = Planner(actions)

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=StrategySelector(),
    )

    strategy = Strategy(
        name="open_the_door_fast",
        goal="Open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
        success_count=5,
        failure_count=0,
        total_actions=10,
    )

    # plan = adaptive_planner.plan(
    #     world_state={
    #         "door": "locked",
    #         "key": "unknown",
    #     },
    #     goal_state={
    #         "door": "unlocked",
    #     },
    #     strategies=[
    #         strategy,
    #     ],
    # )
    plan = adaptive_planner.plan(
        world_state={
            "door": "locked",
            "key": "inside_box",
            "box": "closed",
            "spare_key": "available",
        },
        goal_state={
            "door": "unlocked",
        },
        goal="Open the door",
        strategies=[
            strategy,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "find_spare_key",
        "unlock_door",
    ]


def test_adaptive_planner_falls_back_to_classical_planner():

    actions = create_actions()

    planner = Planner(actions)

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=StrategySelector(),
    )

    plan = adaptive_planner.plan(
        world_state={
            "door": "locked",
            "key": "inside_box",
            "box": "closed",
            "spare_key": "available",
        },
        goal_state={
            "door": "unlocked",
        },
        goal="Open the door",
        strategies=[],
    )


    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
        "take_key",
        "unlock_door",
    ]


def test_adaptive_planner_ignores_unknown_strategy_action():

    actions = create_actions()

    planner = Planner(actions)

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=StrategySelector(),
    )

    strategy = Strategy(
        name="invalid",
        goal="Open the door",
        steps=[
            "nonexistent_action",
            "unlock_door",
        ],
        success_count=10,
        failure_count=0,
        total_actions=20,
    )

    plan = adaptive_planner.plan(
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": "available",
        },
        goal_state={
            "door": "unlocked",
        },
        goal="Open the door",
        strategies=[
            strategy,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

def test_adaptive_planner_falls_back_when_strategy_is_not_executable():

    actions = create_actions()

    planner = Planner(actions)

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=StrategySelector(),
    )

    strategy = Strategy(
        name="bad_strategy",
        goal="Open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
        success_count=10,
        failure_count=0,
        total_actions=20,
    )

    plan = adaptive_planner.plan(
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": "unavailable",
        },
        goal_state={
            "door": "unlocked",
        },
        goal="Open the door",
        strategies=[
            strategy,
        ],
    )

    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
        "take_key",
        "unlock_door",
    ]