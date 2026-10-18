
from agent.core.agent import Agent
from agent.learning.strategy import Strategy


def make_strategy(name: str) -> Strategy:
    return Strategy(
        name=name,
        goal="open the box",
        steps=["open_box"],
    )


def test_consolidation_redirects_active_strategy_to_representative():
    agent = Agent("lifecycle-consistency-agent")

    first = make_strategy("first")
    second = make_strategy("second")

    agent.register_strategies([first, second])
    agent.set_active_strategy(second)

    result = agent.consolidate_strategies()

    assert result.merged_count == 1
    assert second.is_retired
    assert agent.get_active_strategy() is first
    assert agent.get_active_strategy().is_active


def test_paused_active_strategy_is_not_used_for_attribution():
    agent = Agent("lifecycle-consistency-agent")

    strategy = make_strategy("paused-strategy")
    agent.register_strategy(strategy)
    agent.set_active_strategy(strategy)

    strategy.pause()

    # Inactive explicit strategies must not remain active.
    # Attribution can still match another eligible strategy,
    # or return unmatched when no eligible strategy exists.
    from agent.core.experience import Experience

    experience = Experience(
        experience_id="exp-paused-001",
        goal="unrelated goal",
    )

    result = agent.attribute_strategy(experience)

    assert agent.get_active_strategy() is None
    assert result.strategy is None or result.strategy.is_active


def test_retired_strategy_is_not_used_for_attribution():
    agent = Agent("lifecycle-consistency-agent")

    strategy = make_strategy("retired-strategy")
    agent.register_strategy(strategy)
    agent.set_active_strategy(strategy)

    strategy.retire()

    from agent.core.experience import Experience

    experience = Experience(
        experience_id="exp-retired-001",
        goal="unrelated goal",
    )

    result = agent.attribute_strategy(experience)

    assert agent.get_active_strategy() is None
    assert result.strategy is None or result.strategy.is_active
