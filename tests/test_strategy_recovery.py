from agent.learning.strategy import Strategy
from agent.learning.strategy_recovery import (
    StrategyRecoveryManager,
)


def make_paused_strategy() -> Strategy:
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    strategy.record_failure(
        experience_id="exp-1",
        action_count=4,
        progress_delta=-0.5,
    )

    strategy.record_failure(
        experience_id="exp-2",
        action_count=4,
        progress_delta=-0.4,
    )

    strategy.record_failure(
        experience_id="exp-3",
        action_count=4,
        progress_delta=-0.3,
    )

    strategy.pause()

    return strategy


def test_paused_strategy_can_recover():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    strategy.record_failure(
        experience_id="exp-1",
        action_count=5,
        progress_delta=-0.2,
    )

    strategy.record_failure(
        experience_id="exp-2",
        action_count=5,
        progress_delta=-0.1,
    )

    strategy.record_failure(
        experience_id="exp-3",
        action_count=5,
        progress_delta=-0.1,
    )

    strategy.pause()

    # Positive evidence
    strategy.record_success(
        experience_id="exp-4",
        action_count=3,
        progress_delta=0.5,
    )

    strategy.record_success(
        experience_id="exp-5",
        action_count=3,
        progress_delta=0.5,
    )

    strategy.record_success(
        experience_id="exp-6",
        action_count=3,
        progress_delta=0.5,
    )

    manager = StrategyRecoveryManager()

    result = manager.recover(strategy)

    assert result.recovered is True
    assert strategy.is_active is True


def test_paused_strategy_does_not_recover_with_weak_evidence():
    strategy = make_paused_strategy()

    manager = StrategyRecoveryManager()

    result = manager.recover(strategy)

    assert result.recovered is False
    assert strategy.is_paused is True


def test_retired_strategy_cannot_recover():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    strategy.retire()

    manager = StrategyRecoveryManager()

    result = manager.recover(strategy)

    assert result.recovered is False
    assert strategy.is_retired is True


def test_active_strategy_is_not_recovered():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    manager = StrategyRecoveryManager()

    result = manager.recover(strategy)

    assert result.recovered is False
    assert result.reason == "strategy is already active"


def test_recovery_requires_minimum_attempts():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    strategy.record_success(
        experience_id="exp-1",
        action_count=3,
        progress_delta=1.0,
    )

    strategy.pause()

    manager = StrategyRecoveryManager(
        min_attempts=3,
    )

    result = manager.recover(strategy)

    assert result.recovered is False
    assert result.reason == "insufficient attempts"
    assert strategy.is_paused is True


def test_recovery_requires_positive_average_progress():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    for index in range(3):
        strategy.record_success(
            experience_id=f"exp-{index}",
            action_count=3,
            progress_delta=0.0,
        )

    strategy.pause()

    manager = StrategyRecoveryManager()

    result = manager.recover(strategy)

    assert result.recovered is False
    assert (
        result.reason
        == "average progress below recovery threshold"
    )

    assert strategy.is_paused is True
