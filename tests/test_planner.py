from agent.cognition.planner import Planner
from agent.tools.room_actions import create_room_actions
from agent.world.room import RoomEnvironment


def create_planner():
    environment = RoomEnvironment()
    actions = create_room_actions(environment)

    return Planner(actions)


def test_planner_opens_box_first():

    planner = create_planner()

    world_state = {
        "door": "locked",
        "box": "closed",
        "key": "inside_box",
    }

    goal_state = {
        "box": "open",
    }

    plan = planner.plan(
        world_state,
        goal_state,
    )

    action_names = [
        action.name
        for action in plan
    ]

    assert action_names == [
        "open_box",
    ]


def test_planner_takes_key_after_box_is_open():

    planner = create_planner()

    world_state = {
        "door": "locked",
        "box": "open",
        "key": "inside_box",
    }

    goal_state = {
        "key": "agent",
    }

    plan = planner.plan(
        world_state,
        goal_state,
    )

    action_names = [
        action.name
        for action in plan
    ]

    assert action_names == [
        "take_key",
    ]


def test_planner_unlocks_door_when_key_is_available():

    planner = create_planner()

    world_state = {
        "door": "locked",
        "box": "open",
        "key": "agent",
    }

    goal_state = {
        "door": "unlocked",
    }

    plan = planner.plan(
        world_state,
        goal_state,
    )

    action_names = [
        action.name
        for action in plan
    ]

    assert action_names == [
        "unlock_door",
    ]


def test_planner_creates_multi_step_plan():

    planner = create_planner()

    world_state = {
        "door": "locked",
        "box": "closed",
        "key": "inside_box",
    }

    goal_state = {
        "door": "unlocked",
    }

    plan = planner.plan(
        world_state,
        goal_state,
    )

    action_names = [
        action.name
        for action in plan
    ]

    assert action_names == [
        "open_box",
        "take_key",
        "unlock_door",
    ]
