from agent.core.action import Action
from agent.core.agent import Agent
from agent.learning.strategy import Strategy


def create_actions():

    return [
        Action(
            name="take_key",
            description="Take key from box",
            handler=lambda: {
                "success": False,
                "message": "The box is closed.",
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
            name="open_box",
            description="Open box",
            handler=lambda: {
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
            name="find_spare_key",
            description="Find spare key",
            handler=lambda: {
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
            handler=lambda: {
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


def test_agent_reselects_strategy_after_failure():

    agent = Agent("hAI")

    agent.register_actions(
        create_actions()
    )

    strategy_a = Strategy(
        name="strategy_a",
        goal="open the door",
        steps=[
            "take_key",
            "unlock_door",
        ],
        success_count=3,
        failure_count=0,
        total_actions=6,
    )

    strategy_b = Strategy(
        name="strategy_b",
        goal="open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
        success_count=3,
        failure_count=0,
        total_actions=6,
    )

    strategies = [
        strategy_a,
        strategy_b,
    ]

    world = {
        "door": "locked",
        "box": "open",
        "key": "inside_box",
        "spare_key": "available",
    }

    goal = {
        "door": "unlocked",
    }

    # Initial decision.
    plan = agent.choose_plan(
        world_state=world,
        goal_state=goal,
        strategies=strategies,
        goal="open the door",
    )

    assert plan

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]

    # Activate Strategy A.
    agent.set_active_strategy(
        strategy_a
    )

    # Strategy A fails.
    agent.record_experience(
        action="take_key",
        observation_before={
            "box": "closed",
            "key": "inside_box",
        },
        observation_after={
            "box": "closed",
            "key": "inside_box",
        },
        result={
            "success": False,
            "message": "The box is closed.",
        },
    )

    assert (
        strategy_a.reflection_penalty
        > 0.0
    )

    # Reselect.
    plan = agent.choose_plan(
        world_state=world,
        goal_state=goal,
        strategies=strategies,
        goal="open the door",
    )

    assert plan

    assert [
        action.name
        for action in plan
    ] == [
        "find_spare_key",
        "unlock_door",
    ]


def test_agent_does_not_penalize_unrelated_strategy():

    agent = Agent("hAI")

    agent.register_actions(
        create_actions()
    )

    strategy_a = Strategy(
        name="strategy_a",
        goal="open the door",
        steps=[
            "take_key",
            "unlock_door",
        ],
        success_count=3,
        failure_count=0,
        total_actions=6,
    )

    strategy_b = Strategy(
        name="strategy_b",
        goal="open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
        success_count=3,
        failure_count=0,
        total_actions=6,
    )

    agent.set_active_strategy(
        strategy_a
    )

    agent.record_experience(
        action="take_key",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "closed",
        },
        result={
            "success": False,
        },
    )

    assert (
        strategy_a.reflection_penalty
        > 0.0
    )

    assert (
        strategy_b.reflection_penalty
        == 0.0
    )
