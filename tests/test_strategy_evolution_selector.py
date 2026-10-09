from agent.learning.strategy import Strategy
from agent.learning.strategy_evolution_selector import (
    StrategyEvolutionSelector,
)


def make_strategy(
    name: str,
    steps: list[str],
) -> Strategy:
    return Strategy(
        name=name,
        goal="Open door",
        steps=steps,
    )


def test_empty_candidates_returns_no_selection():
    selector = StrategyEvolutionSelector()

    result = selector.select([])

    assert result.selected is False
    assert result.strategy is None
    assert result.score is None
    assert result.reason == (
        "no candidates available"
    )


def test_single_candidate_is_selected():
    candidate = make_strategy(
        name="candidate_a",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    selector = StrategyEvolutionSelector()

    result = selector.select(
        [candidate]
    )

    assert result.selected is True
    assert result.strategy is candidate
    assert result.score is not None


def test_higher_quality_candidate_is_selected():
    weak = make_strategy(
        name="weak_candidate",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    strong = make_strategy(
        name="strong_candidate",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    # Give the stronger strategy positive evidence.
    for index in range(5):
        strong.record_success(
            experience_id=f"success-{index}",
            action_count=3,
            progress_delta=0.5,
        )

    # Give the weak strategy negative evidence.
    for index in range(5):
        weak.record_failure(
            experience_id=f"failure-{index}",
            action_count=5,
            progress_delta=-0.3,
        )

    selector = StrategyEvolutionSelector()

    result = selector.select(
        [
            weak,
            strong,
        ]
    )

    assert result.selected is True
    assert result.strategy is strong


def test_fewer_steps_breaks_equal_score():
    candidate_short = make_strategy(
        name="short_candidate",
        steps=[
            "unlock_door",
        ],
    )

    candidate_long = make_strategy(
        name="long_candidate",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    selector = StrategyEvolutionSelector()

    ranked = selector.rank(
        [
            candidate_long,
            candidate_short,
        ]
    )

    assert ranked[0][0] is candidate_short
    assert ranked[1][0] is candidate_long


def test_equal_score_and_equal_length_preserves_order():
    first = make_strategy(
        name="candidate_a",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    second = make_strategy(
        name="candidate_b",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    selector = StrategyEvolutionSelector()

    ranked = selector.rank(
        [
            first,
            second,
        ]
    )

    assert ranked[0][0] is first
    assert ranked[1][0] is second


def test_none_candidates_are_rejected():
    selector = StrategyEvolutionSelector()

    try:
        selector.select(None)
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "candidates cannot be None."
        )


def test_none_candidate_inside_list_is_rejected():
    selector = StrategyEvolutionSelector()

    try:
        selector.select(
            [None]
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "candidate cannot be None."
        )


def test_ranking_returns_strategy_and_score():
    first = make_strategy(
        name="candidate_a",
        steps=[
            "open_box",
        ],
    )

    second = make_strategy(
        name="candidate_b",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    selector = StrategyEvolutionSelector()

    ranked = selector.rank(
        [
            first,
            second,
        ]
    )

    assert len(ranked) == 2

    assert isinstance(
        ranked[0][0],
        Strategy,
    )

    assert isinstance(
        ranked[0][1],
        float,
    )
