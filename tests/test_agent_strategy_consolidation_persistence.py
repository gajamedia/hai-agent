
from agent.core.agent import Agent
from agent.learning.strategy import Strategy


def make_strategy(name: str) -> Strategy:
    return Strategy(
        name=name,
        goal="open the box",
        steps=["open_box"],
    )


def test_agent_consolidation_redirects_active_strategy(tmp_path):
    agent = Agent(
        "consolidation-agent",
        memory_path=tmp_path / "agent.db",
    )

    representative = make_strategy("representative")
    source = make_strategy("source")

    agent.register_strategies([representative, source])
    agent.set_active_strategy(source)

    result = agent.consolidate_strategies()

    assert result.merged_count == 1
    assert source.is_retired
    assert representative.is_active

    assert agent.get_active_strategy() is representative
    assert agent.get_active_strategy().is_active

    assert agent.strategy_store.count() == 1


def test_agent_consolidation_persists_representative_after_restart(
    tmp_path,
):
    database_path = tmp_path / "agent.db"

    agent = Agent(
        "consolidation-agent",
        memory_path=database_path,
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

    agent.register_strategies([representative, source])
    agent.set_active_strategy(source)

    agent.consolidate_strategies()

    restarted_agent = Agent(
        "consolidation-agent",
        memory_path=database_path,
    )

    restored = restarted_agent.strategy_store.find_exact(
        goal="open the box",
        steps=["open_box"],
    )

    assert restored is not None
    assert restored.name == "representative"
    assert restored.success_count == 5
    assert restored.total_actions == 5
    assert restored.total_progress == 5.0
    assert restored.progress_count == 5
    assert restored.is_active

    assert restarted_agent.strategy_store.count() == 1
