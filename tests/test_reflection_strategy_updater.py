from agent.learning.reflection_strategy_updater import (
    ReflectionStrategyUpdater,
)
from agent.learning.strategy import Strategy
from agent.memory.reflection import ReflectionMemory


def test_reflection_updates_related_strategy():

    strategy = Strategy(
        name="open_the_door",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=2,
        failure_count=1,
        total_actions=9,
    )

    reflection = ReflectionMemory(
        content="Failed take_key.",
        goal="Open the door",
        action="take_key",
        problem="Key unavailable.",
        cause="key was missing.",
        lesson="Check key availability.",
    )

    updater = ReflectionStrategyUpdater()

    updater.apply(
        strategy=strategy,
        reflection=reflection,
        penalty=0.2,
    )

    assert strategy.reflection_penalty == 0.2
    assert reflection.applied is True


def test_reflection_does_not_update_unrelated_strategy():

    strategy = Strategy(
        name="other_strategy",
        goal="Open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
    )

    reflection = ReflectionMemory(
        content="Failed take_key.",
        action="take_key",
    )

    updater = ReflectionStrategyUpdater()

    updater.apply(
        strategy=strategy,
        reflection=reflection,
        penalty=0.2,
    )

    assert strategy.reflection_penalty == 0.0
    assert reflection.applied is False


def test_reflection_penalty_accumulates():

    strategy = Strategy(
        name="open_the_door",
        goal="Open the door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    reflection = ReflectionMemory(
        content="Failed.",
        action="take_key",
    )

    updater = ReflectionStrategyUpdater()

    updater.apply(
        strategy=strategy,
        reflection=reflection,
        penalty=0.2,
    )

    reflection.applied = False

    updater.apply(
        strategy=strategy,
        reflection=reflection,
        penalty=0.1,
    )

    assert strategy.reflection_penalty == 0.3
