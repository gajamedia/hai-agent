
import pytest

from agent.learning.strategy import Strategy
from agent.learning.strategy_consolidation import StrategyConsolidator
from agent.memory.strategy_repository import StrategyRepository
from agent.memory.strategy_store import StrategyMemoryStore


@pytest.mark.parametrize(
    "status",
    ["active", "paused", "retired"],
)
def test_all_lifecycle_statuses_survive_sqlite_reload(
    tmp_path,
    status,
):
    database_path = tmp_path / "lifecycle.db"

    repository = StrategyRepository(database_path)
    store = StrategyMemoryStore(repository=repository)

    strategy = Strategy(
        name=f"strategy_{status}",
        goal="open the box",
        steps=["open_box"],
        status=status,
    )

    store.add(strategy)
    store.save_all()

    reloaded_store = StrategyMemoryStore(
        repository=StrategyRepository(database_path)
    )

    restored = reloaded_store.find_matching(strategy)

    assert restored is not None
    assert restored.status == status


def test_strategies_with_same_goal_and_steps_but_different_names_remain_distinct(
    tmp_path,
):
    database_path = tmp_path / "identity.db"

    store = StrategyMemoryStore(
        repository=StrategyRepository(database_path)
    )

    first = Strategy(
        name="strategy_alpha",
        goal="open the box",
        steps=["open_box"],
    )
    second = Strategy(
        name="strategy_beta",
        goal="open the box",
        steps=["open_box"],
    )

    store.add(first)
    store.add(second)
    store.save_all()

    reloaded_store = StrategyMemoryStore(
        repository=StrategyRepository(database_path)
    )

    assert reloaded_store.count() == 2
    assert reloaded_store.find_matching(first).name == "strategy_alpha"
    assert reloaded_store.find_matching(second).name == "strategy_beta"


def test_consolidated_representative_survives_sqlite_reload(
    tmp_path,
):
    database_path = tmp_path / "consolidation.db"

    store = StrategyMemoryStore(
        repository=StrategyRepository(database_path)
    )

    representative = Strategy(
        name="representative",
        goal="open the box",
        steps=["open_box"],
        success_count=2,
        total_actions=2,
        total_progress=2.0,
        progress_count=2,
    )

    source = Strategy(
        name="source",
        goal="open the box",
        steps=["open_box"],
        success_count=3,
        total_actions=3,
        total_progress=3.0,
        progress_count=3,
    )

    store.add(representative)
    store.add(source)

    result = StrategyConsolidator().consolidate(
        [representative, source]
    )

    assert result.merged_count == 1
    assert len(result.strategies) == 1
    assert source.is_retired

    # Persist the consolidated registry, as the Agent does.
    store.clear()
    store.add(result.strategies[0])
    store.save_all()

    reloaded_store = StrategyMemoryStore(
        repository=StrategyRepository(database_path)
    )

    restored = reloaded_store.find_exact(
        goal="open the box",
        steps=["open_box"],
    )

    assert restored is not None
    assert restored.name == "representative"
    assert restored.success_count == 5
    assert restored.total_actions == 5
    assert restored.total_progress == 5.0
    assert restored.progress_count == 5
    assert restored.status == "active"
    assert reloaded_store.count() == 1
