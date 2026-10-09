from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import (
    StrategySelector,
)


def test_strategy_selector_prefers_higher_success_rate():

    first = Strategy(
        name="strategy_a",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=9,
        failure_count=1,
        total_actions=30,
    )

    second = Strategy(
        name="strategy_b",
        goal="Open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
        success_count=6,
        failure_count=4,
        total_actions=20,
    )

    selector = StrategySelector()

    selected = selector.select(
        [first, second]
    )

    assert selected is first


def test_strategy_selector_prefers_lower_action_cost_when_success_is_equal():

    first = Strategy(
        name="strategy_a",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=10,
        failure_count=0,
        total_actions=30,
    )

    second = Strategy(
        name="strategy_b",
        goal="Open the door",
        steps=[
            "find_key",
            "unlock_door",
        ],
        success_count=10,
        failure_count=0,
        total_actions=20,
    )

    selector = StrategySelector()

    selected = selector.select(
        [first, second]
    )

    assert selected is second

def test_strategy_selector_filters_by_goal():

    first = Strategy(
        name="strategy_a",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=10,
        failure_count=0,
        total_actions=30,
    )

    second = Strategy(
        name="strategy_b",
        goal="Close the door",
        steps=[
            "close_door",
        ],
        success_count=10,
        failure_count=0,
        total_actions=10,
    )

    selector = StrategySelector()

    selected = selector.select(
        [first, second],
        goal="Open the door",
    )

    assert selected is first

def test_strategy_selector_uses_reflection_penalty():

    first = Strategy(
        name="strategy_a",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=8,
        failure_count=2,
        total_actions=30,
        reflection_penalty=0.3,
    )

    second = Strategy(
        name="strategy_b",
        goal="Open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
        success_count=7,
        failure_count=3,
        total_actions=20,
        reflection_penalty=0.0,
    )

    selector = StrategySelector()

    selected = selector.select(
        [first, second],
        goal="Open the door",
    )

    assert selected is second
