from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)
from agent.learning.strategy import Strategy


def test_strategy_with_strong_success_history_scores_higher():

    safe = Strategy(
        name="safe",
        goal="open door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    for index in range(9):
        safe.record_success(
            experience_id=f"safe-{index}",
            action_count=2,
        )

    safe.record_failure(
        experience_id="safe-failure",
        action_count=2,
    )

    risky = Strategy(
        name="risky",
        goal="open door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
    )

    for index in range(2):
        risky.record_success(
            experience_id=f"risky-{index}",
            action_count=2,
        )

    for index in range(8):
        risky.record_failure(
            experience_id=f"risky-failure-{index}",
            action_count=2,
        )

    selector = RiskAwareStrategySelector()

    assert (
        selector.score(safe)
        > selector.score(risky)
    )


def test_selector_prefers_lower_risk_strategy():

    reliable = Strategy(
        name="reliable",
        goal="open door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    for index in range(8):
        reliable.record_success(
            experience_id=f"reliable-{index}",
            action_count=2,
        )

    unreliable = Strategy(
        name="unreliable",
        goal="open door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
    )

    for index in range(3):
        unreliable.record_success(
            experience_id=f"unreliable-{index}",
            action_count=2,
        )

    for index in range(7):
        unreliable.record_failure(
            experience_id=f"unreliable-failure-{index}",
            action_count=2,
        )

    selector = RiskAwareStrategySelector()

    selected = selector.select(
        [
            unreliable,
            reliable,
        ]
    )

    assert selected is reliable


def test_reflection_penalty_reduces_strategy_score():

    strategy = Strategy(
        name="strategy",
        goal="open door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    for index in range(5):
        strategy.record_success(
            experience_id=f"experience-{index}",
            action_count=2,
        )

    selector = RiskAwareStrategySelector()

    before = selector.score(
        strategy
    )

    strategy.apply_reflection_penalty(
        amount=0.4
    )

    after = selector.score(
        strategy
    )

    assert after < before


def test_more_actions_increase_strategy_cost():

    short = Strategy(
        name="short",
        goal="open door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    short.record_success(
        experience_id="short-1",
        action_count=2,
    )

    long = Strategy(
        name="long",
        goal="open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    long.record_success(
        experience_id="long-1",
        action_count=10,
    )

    selector = RiskAwareStrategySelector()

    assert (
        selector.score(short)
        > selector.score(long)
    )


def test_rank_returns_highest_score_first():

    first = Strategy(
        name="first",
        goal="open door",
    )

    second = Strategy(
        name="second",
        goal="open door",
    )

    for index in range(10):
        first.record_success(
            experience_id=f"first-{index}",
            action_count=1,
        )

    for index in range(3):
        second.record_success(
            experience_id=f"second-{index}",
            action_count=1,
        )

    for index in range(7):
        second.record_failure(
            experience_id=f"second-failure-{index}",
            action_count=1,
        )

    selector = RiskAwareStrategySelector()

    ranked = selector.rank(
        [
            second,
            first,
        ]
    )

    assert ranked[0][0] is first
    assert ranked[0][1] > ranked[1][1]


def test_empty_strategy_list_returns_none():

    selector = RiskAwareStrategySelector()

    assert (
        selector.select([])
        is None
    )


def test_none_strategy_is_rejected():

    selector = RiskAwareStrategySelector()

    try:
        selector.assess(None)
    except ValueError as exc:
        assert str(exc) == (
            "strategy cannot be None."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )
