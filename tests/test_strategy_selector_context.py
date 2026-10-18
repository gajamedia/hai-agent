from agent.core.action import Action
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector


def make_actions():
    # Handlers are intentionally inert: selection must never execute them.
    def inert(**kwargs):
        raise AssertionError("Strategy selection must not execute action handlers.")

    return [
        Action(
            name="open_box",
            description="Open box",
            handler=inert,
            preconditions={"box": "closed"},
            effects={"box": "open"},
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=inert,
            preconditions={"box": "open", "key": "inside_box"},
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key",
            description="Find spare key",
            handler=inert,
            preconditions={"spare_key": "available"},
            effects={"key": "agent", "spare_key": "found"},
        ),
        Action(
            name="unlock_door",
            description="Unlock door",
            handler=inert,
            preconditions={"key": "agent", "door": "locked"},
            effects={"door": "unlocked"},
        ),
    ]


def make_strategy(name, steps, action_count):
    strategy = Strategy(name=name, goal="unlock the door", steps=steps)
    strategy.record_success(
        experience_id=f"{name}-training",
        action_count=action_count,
        progress_delta=1.0,
    )
    return strategy


def test_context_aware_selector_chooses_shortest_feasible_strategy():
    box_route = make_strategy(
        "open_box_then_take_key",
        ["open_box", "take_key", "unlock_door"],
        3,
    )
    spare_route = make_strategy(
        "find_spare_key",
        ["find_spare_key", "unlock_door"],
        2,
    )
    selector = StrategySelector()
    state = {
        "box": "closed",
        "key": "inside_box",
        "door": "locked",
        "spare_key": "unavailable",
    }

    selected = selector.select(
        [spare_route, box_route],
        goal="unlock the door",
        world_state=state,
        actions=make_actions(),
        goal_state={"door": "unlocked"},
    )

    assert selected is box_route


def test_context_aware_selector_uses_learned_score_when_multiple_routes_feasible():
    box_route = make_strategy(
        "open_box_then_take_key",
        ["open_box", "take_key", "unlock_door"],
        3,
    )
    spare_route = make_strategy(
        "find_spare_key",
        ["find_spare_key", "unlock_door"],
        2,
    )
    state = {
        "box": "closed",
        "key": "inside_box",
        "door": "locked",
        "spare_key": "available",
    }

    selected = StrategySelector().select(
        [box_route, spare_route],
        goal="unlock the door",
        world_state=state,
        actions=make_actions(),
        goal_state={"door": "unlocked"},
    )

    assert selected is spare_route


def test_context_aware_selector_returns_none_if_no_strategy_is_feasible():
    spare_route = make_strategy(
        "find_spare_key",
        ["find_spare_key", "unlock_door"],
        2,
    )
    state = {
        "box": "closed",
        "key": "inside_box",
        "door": "locked",
        "spare_key": "unavailable",
    }

    selected = StrategySelector().select(
        [spare_route],
        goal="unlock the door",
        world_state=state,
        actions=make_actions(),
        goal_state={"door": "unlocked"},
    )

    assert selected is None


def test_context_aware_selector_rejects_unknown_action_name():
    unknown = make_strategy("unknown", ["not_registered"], 1)
    selected = StrategySelector().select(
        [unknown],
        world_state={"door": "locked"},
        actions=make_actions(),
    )
    assert selected is None


def test_legacy_goal_only_selection_remains_backward_compatible():
    slower = make_strategy("slower", ["a", "b", "c"], 3)
    faster = make_strategy("faster", ["x", "y"], 2)

    selected = StrategySelector().select(
        [slower, faster],
        goal="unlock the door",
    )

    assert selected is faster


def test_world_state_and_actions_must_be_provided_together():
    selector = StrategySelector()
    strategy = make_strategy("one", ["a"], 1)

    try:
        selector.select([strategy], world_state={"door": "locked"})
    except ValueError as exc:
        assert "must be supplied together" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incomplete context.")
