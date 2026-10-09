from agent.core.action import Action
from agent.cognition.action_exploration import (
    ActionExplorationPolicy,
)


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
    ]


def test_only_executable_actions_become_candidates():

    policy = ActionExplorationPolicy()

    actions = create_actions()

    candidates = policy.candidates(
        actions=actions,
        world_state={
            "box": "closed",
            "key": "inside_box",
        },
        information_gain={
            "open_box": 0.9,
            "take_key": 0.9,
        },
        risk={
            "open_box": 0.1,
            "take_key": 0.1,
        },
    )

    assert [
        candidate.action
        for candidate in candidates
    ] == [
        "open_box",
    ]


def test_select_returns_executable_action():

    policy = ActionExplorationPolicy()

    actions = create_actions()

    selected = policy.select(
        actions=actions,
        world_state={
            "box": "closed",
            "key": "inside_box",
        },
        information_gain={
            "open_box": 0.9,
            "take_key": 0.9,
        },
        risk={
            "open_box": 0.1,
            "take_key": 0.1,
        },
    )

    assert selected is not None

    assert selected.name == "open_box"


def test_higher_information_gain_is_preferred():

    policy = ActionExplorationPolicy()

    actions = create_actions()

    ranked = policy.rank(
        actions=actions,
        world_state={
            "box": "open",
            "key": "inside_box",
        },
        information_gain={
            "open_box": 0.2,
            "take_key": 0.9,
        },
        risk={
            "open_box": 0.1,
            "take_key": 0.1,
        },
    )

    assert ranked[0][0].name == "take_key"


def test_high_risk_action_is_penalized():

    policy = ActionExplorationPolicy()

    actions = create_actions()

    ranked = policy.rank(
        actions=actions,
        world_state={
            "box": "open",
            "key": "inside_box",
        },
        information_gain={
            "open_box": 0.9,
            "take_key": 0.8,
        },
        risk={
            "open_box": 0.9,
            "take_key": 0.1,
        },
    )

    assert ranked[0][0].name == "take_key"


def test_action_cost_is_considered():

    policy = ActionExplorationPolicy()

    actions = create_actions()

    ranked = policy.rank(
        actions=actions,
        world_state={
            "box": "open",
            "key": "inside_box",
        },
        information_gain={
            "open_box": 0.9,
            "take_key": 0.9,
        },
        risk={
            "open_box": 0.1,
            "take_key": 0.1,
        },
        action_cost={
            "open_box": 5.0,
            "take_key": 1.0,
        },
    )

    assert ranked[0][0].name == "take_key"


def test_no_executable_action_returns_none():

    policy = ActionExplorationPolicy()

    actions = create_actions()

    selected = policy.select(
        actions=actions,
        world_state={
            "box": "open",
            "key": "agent",
        },
        information_gain={
            "open_box": 0.9,
            "take_key": 0.9,
        },
        risk={
            "open_box": 0.1,
            "take_key": 0.1,
        },
    )

    assert selected is None
