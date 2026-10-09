import pytest

from agent.cognition.exploration_policy import (
    ExplorationCandidate,
    ExplorationPolicy,
)


def test_candidate_score_rewards_information_gain():

    candidate = ExplorationCandidate(
        action="open_box",
        expected_information_gain=0.9,
        risk=0.1,
        action_cost=0.0,
    )

    assert (
        candidate.score
        == pytest.approx(0.8)
    )


def test_policy_prefers_high_information_gain():

    policy = ExplorationPolicy()

    useful = ExplorationCandidate(
        action="open_box",
        expected_information_gain=0.9,
        risk=0.1,
        action_cost=0.0,
    )

    weak = ExplorationCandidate(
        action="look_around",
        expected_information_gain=0.4,
        risk=0.1,
        action_cost=0.0,
    )

    selected = policy.select(
        [
            weak,
            useful,
        ]
    )

    assert selected is useful


def test_policy_penalizes_high_risk():

    policy = ExplorationPolicy()

    safe = ExplorationCandidate(
        action="open_box",
        expected_information_gain=0.7,
        risk=0.1,
    )

    risky = ExplorationCandidate(
        action="force_door",
        expected_information_gain=0.9,
        risk=0.9,
    )

    selected = policy.select(
        [
            risky,
            safe,
        ]
    )

    assert selected is safe


def test_policy_penalizes_action_cost():

    policy = ExplorationPolicy()

    cheap = ExplorationCandidate(
        action="inspect_box",
        expected_information_gain=0.7,
        risk=0.1,
        action_cost=1.0,
    )

    expensive = ExplorationCandidate(
        action="perform_large_experiment",
        expected_information_gain=0.8,
        risk=0.1,
        action_cost=5.0,
    )

    selected = policy.select(
        [
            expensive,
            cheap,
        ]
    )

    assert selected is cheap


def test_rank_returns_best_candidate_first():

    policy = ExplorationPolicy()

    first = ExplorationCandidate(
        action="first",
        expected_information_gain=0.9,
        risk=0.1,
    )

    second = ExplorationCandidate(
        action="second",
        expected_information_gain=0.5,
        risk=0.2,
    )

    ranked = policy.rank(
        [
            second,
            first,
        ]
    )

    assert ranked[0] is first
    assert ranked[1] is second


def test_empty_candidates_return_none():

    policy = ExplorationPolicy()

    assert (
        policy.select([])
        is None
    )


def test_none_candidate_is_rejected():

    policy = ExplorationPolicy()

    with pytest.raises(
        ValueError,
        match="candidate cannot be None",
    ):
        policy.score(None)


def test_values_are_clamped():

    policy = ExplorationPolicy()

    candidate = ExplorationCandidate(
        action="test",
        expected_information_gain=2.0,
        risk=-1.0,
        action_cost=0.0,
    )

    assert (
        policy.score(candidate)
        == pytest.approx(1.0)
    )


def test_custom_weights_change_selection():

    policy = ExplorationPolicy(
        information_gain_weight=1.0,
        risk_weight=2.0,
        action_cost_weight=0.1,
    )

    safe = ExplorationCandidate(
        action="safe",
        expected_information_gain=0.7,
        risk=0.1,
    )

    risky = ExplorationCandidate(
        action="risky",
        expected_information_gain=0.9,
        risk=0.5,
    )

    selected = policy.select(
        [
            risky,
            safe,
        ]
    )

    assert selected is safe
