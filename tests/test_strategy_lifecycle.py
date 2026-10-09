import pytest

from agent.learning.strategy import Strategy
from agent.learning.strategy_consolidation import (
    StrategyConsolidator,
    StrategyLifecycleManager,
)


def test_new_strategy_is_active():

    strategy = Strategy(
        name="open_box",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    assert strategy.is_active
    assert not strategy.is_paused
    assert not strategy.is_retired


def test_strategy_can_be_paused():

    strategy = Strategy(
        name="open_box",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    strategy.pause()

    assert strategy.is_paused
    assert not strategy.is_active


def test_strategy_can_be_reactivated():

    strategy = Strategy(
        name="open_box",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    strategy.pause()
    strategy.activate()

    assert strategy.is_active


def test_retired_strategy_cannot_be_activated():

    strategy = Strategy(
        name="bad_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    strategy.retire()
    strategy.activate()

    assert strategy.is_retired


def test_equivalent_strategies_are_detected():

    first = Strategy(
        name="strategy_a",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    second = Strategy(
        name="strategy_b",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    consolidator = (
        StrategyConsolidator()
    )

    assert (
        consolidator.equivalent(
            first,
            second,
        )
        is True
    )


def test_consolidation_merges_evidence():

    first = Strategy(
        name="strategy_a",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=3,
        failure_count=1,
        total_actions=4,
        total_progress=2.0,
        progress_count=4,
        source_experiences=[
            "a-1",
            "a-2",
        ],
    )

    second = Strategy(
        name="strategy_b",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=2,
        failure_count=1,
        total_actions=3,
        total_progress=1.0,
        progress_count=3,
        source_experiences=[
            "b-1",
        ],
    )

    consolidator = (
        StrategyConsolidator()
    )

    result = consolidator.consolidate(
        [
            first,
            second,
        ]
    )

    assert (
        result.merged_count
        == 1
    )

    assert (
        len(result.strategies)
        == 1
    )

    merged = result.strategies[0]

    assert (
        merged.success_count
        == 5
    )

    assert (
        merged.failure_count
        == 2
    )

    assert (
        merged.total_actions
        == 7
    )

    assert (
        merged.total_progress
        == 3.0
    )

    assert (
        merged.progress_count
        == 7
    )

    assert (
        set(merged.source_experiences)
        == {
            "a-1",
            "a-2",
            "b-1",
        }
    )

    assert second.is_retired


def test_duplicate_consolidation_uses_first_strategy():

    first = Strategy(
        name="first",
        goal="open the box",
        steps=["open_box"],
    )

    second = Strategy(
        name="second",
        goal="open the box",
        steps=["open_box"],
    )

    consolidator = (
        StrategyConsolidator()
    )

    result = consolidator.consolidate(
        [
            first,
            second,
        ]
    )

    assert (
        result.strategies[0]
        is first
    )


def test_different_strategies_are_not_merged():

    first = Strategy(
        name="open_box",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    second = Strategy(
        name="inspect_box",
        goal="open the box",
        steps=[
            "inspect_box",
            "open_box",
        ],
    )

    consolidator = (
        StrategyConsolidator()
    )

    result = consolidator.consolidate(
        [
            first,
            second,
        ]
    )

    assert (
        result.merged_count
        == 0
    )

    assert (
        len(result.strategies)
        == 2
    )


def test_bad_strategy_is_paused():

    strategy = Strategy(
        name="bad_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=1,
        failure_count=5,
        total_progress=0.0,
        progress_count=6,
    )

    lifecycle = (
        StrategyLifecycleManager()
    )

    lifecycle.evaluate(
        strategy
    )

    assert strategy.is_paused


def test_very_bad_strategy_is_retired():

    strategy = Strategy(
        name="terrible_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=0,
        failure_count=10,
        total_progress=-3.0,
        progress_count=10,
    )

    lifecycle = (
        StrategyLifecycleManager()
    )

    lifecycle.evaluate(
        strategy
    )

    assert strategy.is_retired


def test_good_strategy_remains_active():

    strategy = Strategy(
        name="good_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=8,
        failure_count=1,
        total_progress=7.0,
        progress_count=9,
    )

    lifecycle = (
        StrategyLifecycleManager()
    )

    lifecycle.evaluate(
        strategy
    )

    assert strategy.is_active

def test_agent_can_consolidate_registered_strategies():

    from agent.core.agent import Agent

    agent = Agent(
        "lifecycle-agent"
    )

    first = Strategy(
        name="first",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=2,
    )

    second = Strategy(
        name="second",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=3,
    )

    agent.register_strategies(
        [
            first,
            second,
        ]
    )

    result = (
        agent.consolidate_strategies()
    )

    assert (
        result.merged_count
        == 1
    )

    assert (
        len(
            agent.get_registered_strategies()
        )
        == 1
    )

    assert (
        first.success_count
        == 5
    )


def test_agent_can_evaluate_strategy_lifecycle():

    from agent.core.agent import Agent

    agent = Agent(
        "lifecycle-agent"
    )

    strategy = Strategy(
        name="poor_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=1,
        failure_count=5,
        total_progress=0.0,
        progress_count=6,
    )

    agent.register_strategy(
        strategy
    )

    strategies = (
        agent.evaluate_strategy_lifecycle()
    )

    assert strategies[0].is_paused
