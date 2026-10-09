from agent.learning.strategy import Strategy
from agent.learning.strategy_store import StrategyStore
from agent.memory.strategy_store import StrategyMemoryStore


def test_learning_strategy_store_is_memory_strategy_store():
    assert StrategyStore is StrategyMemoryStore


def test_legacy_store_api_still_works():
    store = StrategyStore()

    strategy = store.get_or_create(
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    assert strategy.goal == "Open door"

    assert strategy.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    assert store.count() == 1


def test_get_or_create_reuses_existing_strategy():
    store = StrategyMemoryStore()

    first = store.get_or_create(
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    second = store.get_or_create(
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    assert first is second
    assert store.count() == 1


def test_for_goal_returns_matching_strategies():
    store = StrategyMemoryStore()

    store.add(
        Strategy(
            name="open_door_1",
            goal="Open door",
            steps=["open_box", "take_key", "unlock_door"],
        )
    )

    store.add(
        Strategy(
            name="open_door_2",
            goal="Open door",
            steps=["find_spare_key", "unlock_door"],
        )
    )

    store.add(
        Strategy(
            name="other_goal",
            goal="Close door",
            steps=["lock_door"],
        )
    )

    strategies = store.for_goal("Open door")

    assert len(strategies) == 2
    assert all(strategy.goal == "Open door" for strategy in strategies)


def test_strategy_store_has_single_runtime_source():
    memory_store = StrategyMemoryStore()

    learning_store = StrategyStore()

    assert isinstance(
        memory_store,
        StrategyMemoryStore,
    )

    assert isinstance(
        learning_store,
        StrategyMemoryStore,
    )
