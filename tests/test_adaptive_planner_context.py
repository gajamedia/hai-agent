from agent.core.action import Action
from agent.cognition.adaptive_planner import AdaptivePlanner
from agent.cognition.planner import Planner
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector


def create_actions():
    def inert(**kwargs):
        return {"success": True}

    return [
        Action("open_box", "Open box", inert,
               preconditions={"box": "closed"}, effects={"box": "open"}),
        Action("take_key", "Take key", inert,
               preconditions={"box": "open", "key": "inside_box"},
               effects={"key": "agent"}),
        Action("find_spare_key", "Find spare key", inert,
               preconditions={"spare_key": "available"},
               effects={"key": "agent", "spare_key": "found"}),
        Action("unlock_door", "Unlock door", inert,
               preconditions={"key": "agent", "door": "locked"},
               effects={"door": "unlocked"}),
    ]


def make_strategy(name, steps, action_count):
    strategy = Strategy(name=name, goal="unlock the door", steps=steps)
    strategy.record_success(
        experience_id=f"{name}-training",
        action_count=action_count,
        progress_delta=1.0,
    )
    return strategy


def setup():
    actions = create_actions()
    return AdaptivePlanner(Planner(actions), StrategySelector())


def test_adaptive_planner_filters_infeasible_spare_key_route():
    adaptive = setup()
    spare = make_strategy("find_spare_key", ["find_spare_key", "unlock_door"], 2)
    box = make_strategy("open_box_then_take_key",
                        ["open_box", "take_key", "unlock_door"], 3)

    plan = adaptive.plan(
        world_state={"box": "closed", "key": "inside_box",
                     "door": "locked", "spare_key": "unavailable"},
        goal_state={"door": "unlocked"},
        goal="unlock the door",
        strategies=[spare, box],
    )
    assert [action.name for action in plan] == [
        "open_box", "take_key", "unlock_door"
    ]


def test_adaptive_planner_uses_shorter_route_when_both_are_feasible():
    adaptive = setup()
    box = make_strategy("open_box_then_take_key",
                        ["open_box", "take_key", "unlock_door"], 3)
    spare = make_strategy("find_spare_key", ["find_spare_key", "unlock_door"], 2)

    plan = adaptive.plan(
        world_state={"box": "closed", "key": "inside_box",
                     "door": "locked", "spare_key": "available"},
        goal_state={"door": "unlocked"},
        goal="unlock the door",
        strategies=[box, spare],
    )
    assert [action.name for action in plan] == [
        "find_spare_key", "unlock_door"
    ]


def test_unknown_or_infeasible_strategy_falls_back_to_base_planner():
    adaptive = setup()
    invalid = make_strategy("invalid", ["not_registered"], 1)

    plan = adaptive.plan(
        world_state={"box": "closed", "key": "inside_box",
                     "door": "locked", "spare_key": "unavailable"},
        goal_state={"door": "unlocked"},
        goal="unlock the door",
        strategies=[invalid],
    )

    assert plan
    simulated = {"box": "closed", "key": "inside_box",
                 "door": "locked", "spare_key": "unavailable"}
    for action in plan:
        assert action.can_execute(simulated)
        simulated.update(action.effects)
    assert simulated["door"] == "unlocked"


def test_paused_strategy_is_not_selected():
    adaptive = setup()
    paused = make_strategy("paused_spare", ["find_spare_key", "unlock_door"], 2)
    paused.pause()

    plan = adaptive.plan(
        world_state={"box": "closed", "key": "inside_box",
                     "door": "locked", "spare_key": "available"},
        goal_state={"door": "unlocked"},
        goal="unlock the door",
        strategies=[paused],
    )
    # With no active learned strategies, the base planner still makes progress.
    simulated = {"box": "closed", "key": "inside_box",
                 "door": "locked", "spare_key": "available"}
    for action in plan:
        assert action.can_execute(simulated)
        simulated.update(action.effects)
    assert simulated["door"] == "unlocked"
