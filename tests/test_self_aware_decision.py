import pytest

from agent.cognition.agent_decision import (
    AgentDecision,
    DecisionMode,
)

from agent.learning.self_aware_decision import (
    SelfAwareDecisionArbiter,
    SelfAwareDecisionResult,
    build_self_aware_decision_arbiter,
)

from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SELF_MODEL_MODE_NEUTRAL,
    SelfModelDecisionInfluence,
)


# ============================================================
# Helpers
# ============================================================


def make_decision(
    *,
    mode=DecisionMode.EXPLOIT,
    uncertainty=0.20,
    information_gain=0.10,
    risk=0.10,
):
    return AgentDecision(
        mode=mode,
        reason="base decision",
        uncertainty=uncertainty,
        expected_information_gain=(
            information_gain
        ),
        risk=risk,
    )


def make_influence(
    *,
    preferred_mode=SELF_MODEL_MODE_NEUTRAL,
    explore_bonus=0.0,
    exploit_bonus=0.0,
    confidence=0.50,
    uncertainty=0.50,
    strength=0.0,
    applied=False,
):
    return SelfModelDecisionInfluence(
        preferred_mode=preferred_mode,
        explore_bonus=explore_bonus,
        exploit_bonus=exploit_bonus,
        confidence=confidence,
        uncertainty=uncertainty,
        strength=strength,
        applied=applied,
        reason="test influence",
    )


# ============================================================
# Initialization
# ============================================================


def test_arbiter_initializes():
    arbiter = (
        SelfAwareDecisionArbiter()
    )

    assert (
        arbiter.enable_self_influence
        is True
    )


def test_arbiter_can_disable_self_influence():
    arbiter = (
        SelfAwareDecisionArbiter(
            enable_self_influence=False
        )
    )

    assert (
        arbiter.enable_self_influence
        is False
    )


def test_builder_returns_arbiter():
    arbiter = (
        build_self_aware_decision_arbiter()
    )

    assert isinstance(
        arbiter,
        SelfAwareDecisionArbiter,
    )


# ============================================================
# Base Decision
# ============================================================


def test_base_scores_are_calculated_correctly():
    decision = make_decision(
        uncertainty=0.40,
        information_gain=0.80,
        risk=0.20,
    )

    influence = make_influence()

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    expected_explore = (
        0.50 * 0.80
        + 0.30 * 0.40
        - 0.20 * 0.20
    )

    expected_exploit = (
        0.50 * (1.0 - 0.40)
        + 0.50 * (1.0 - 0.20)
    )

    assert result.base_explore_score == pytest.approx(
        expected_explore,
        abs=1e-6,
    )

    assert result.base_exploit_score == pytest.approx(
        expected_exploit,
        abs=1e-6,
    )


def test_base_decision_is_preserved_without_influence():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT
    )

    influence = make_influence()

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert result.selected_mode == (
        decision.mode
    )

    assert result.decision == (
        decision
    )

    assert not result.applied


# ============================================================
# Exploration Influence
# ============================================================


def test_self_uncertainty_can_shift_decision_to_exploration():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT,
        uncertainty=0.50,
        information_gain=0.90,
        risk=0.10,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLORE
        ),
        explore_bonus=0.25,
        exploit_bonus=0.0,
        confidence=0.20,
        uncertainty=0.90,
        strength=1.0,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert result.applied

    assert (
        result.selected_mode
        == DecisionMode.EXPLORE
    )

    assert result.mode_changed
    assert result.exploration_selected

def test_exploration_influence_increases_explore_score():
    decision = make_decision(
        uncertainty=0.50,
        information_gain=0.30,
        risk=0.20,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLORE
        ),
        explore_bonus=0.20,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert result.final_explore_score == pytest.approx(
        result.base_explore_score + 0.20,
        abs=1e-6,
    )


# ============================================================
# Exploitation Influence
# ============================================================


def test_self_confidence_can_shift_decision_to_exploitation():
    decision = make_decision(
        mode=DecisionMode.EXPLORE,
        uncertainty=0.60,
        information_gain=0.20,
        risk=0.20,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLOIT
        ),
        explore_bonus=0.0,
        exploit_bonus=0.25,
        confidence=0.90,
        uncertainty=0.10,
        strength=1.0,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert result.applied

    assert (
        result.selected_mode
        == DecisionMode.EXPLOIT
    )

    assert result.mode_changed
    assert result.exploitation_selected


def test_exploitation_influence_increases_exploit_score():
    decision = make_decision(
        uncertainty=0.40,
        information_gain=0.20,
        risk=0.20,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLOIT
        ),
        exploit_bonus=0.20,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert result.final_exploit_score == pytest.approx(
        result.base_exploit_score + 0.20,
        abs=1e-6,
    )


# ============================================================
# Neutral Influence
# ============================================================


def test_neutral_influence_does_not_change_decision():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT,
        uncertainty=0.50,
        information_gain=0.50,
        risk=0.30,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_NEUTRAL
        ),
        explore_bonus=0.20,
        exploit_bonus=0.20,
        applied=False,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert not result.applied

    assert (
        result.final_explore_score
        == result.base_explore_score
    )

    assert (
        result.final_exploit_score
        == result.base_exploit_score
    )


def test_unapplied_influence_does_not_change_decision():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLORE
        ),
        explore_bonus=0.25,
        applied=False,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert not result.applied

    assert (
        result.selected_mode
        == decision.mode
    )


# ============================================================
# Disable Influence
# ============================================================


def test_disabled_self_influence_preserves_base_decision():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT,
        uncertainty=0.60,
        information_gain=0.20,
        risk=0.30,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLORE
        ),
        explore_bonus=0.25,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter(
            enable_self_influence=False
        )
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert not result.applied

    assert (
        result.selected_mode
        == decision.mode
    )

    assert (
        result.decision
        == decision
    )


# ============================================================
# Tie-Break
# ============================================================


def test_tie_breaks_to_exploitation():
    decision = make_decision(
        mode=DecisionMode.EXPLORE,
        uncertainty=0.50,
        information_gain=0.50,
        risk=0.50,
    )

    influence = make_influence()

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert (
        result.selected_mode
        == DecisionMode.EXPLOIT
    )


# ============================================================
# Decision Immutability
# ============================================================


def test_base_decision_is_not_mutated():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT,
        uncertainty=0.50,
        information_gain=0.20,
        risk=0.40,
    )

    original = decision

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLORE
        ),
        explore_bonus=0.25,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert decision == original

    assert result.base_decision == original


def test_adapted_decision_preserves_measurements():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT,
        uncertainty=0.50,
        information_gain=0.20,
        risk=0.40,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLORE
        ),
        explore_bonus=0.25,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    adapted = result.decision

    assert (
        adapted.uncertainty
        == decision.uncertainty
    )

    assert (
        adapted.expected_information_gain
        == decision.expected_information_gain
    )

    assert (
        adapted.risk
        == decision.risk
    )


# ============================================================
# Mode Changed / Not Changed
# ============================================================


def test_mode_changed_is_true_when_mode_differs():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT,
        uncertainty=0.50,
        information_gain=0.70,
        risk=0.20,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLORE
        ),
        explore_bonus=0.25,
        exploit_bonus=0.0,
        confidence=0.20,
        uncertainty=0.90,
        strength=1.0,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert result.mode_changed

    assert (
        result.selected_mode
        == DecisionMode.EXPLORE
    )

def test_mode_changed_is_false_when_mode_is_same():
    decision = make_decision(
        mode=DecisionMode.EXPLOIT,
        uncertainty=0.10,
        information_gain=0.10,
        risk=0.10,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLOIT
        ),
        exploit_bonus=0.20,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert not result.mode_changed


# ============================================================
# Margin
# ============================================================


def test_base_margin_is_correct():
    decision = make_decision(
        uncertainty=0.40,
        information_gain=0.60,
        risk=0.20,
    )

    influence = make_influence()

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert result.base_margin == pytest.approx(
        result.base_explore_score
        - result.base_exploit_score,
        abs=1e-6,
    )


def test_final_margin_is_correct():
    decision = make_decision(
        uncertainty=0.40,
        information_gain=0.60,
        risk=0.20,
    )

    influence = make_influence(
        preferred_mode=(
            SELF_MODEL_MODE_EXPLORE
        ),
        explore_bonus=0.10,
        applied=True,
    )

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert result.final_margin == pytest.approx(
        result.final_explore_score
        - result.final_exploit_score,
        abs=1e-6,
    )


# ============================================================
# Bounds
# ============================================================


@pytest.mark.parametrize(
    "uncertainty",
    [-1.0, 0.0, 0.5, 1.0, 2.0],
)
def test_uncertainty_is_clamped(
    uncertainty,
):
    decision = make_decision(
        uncertainty=uncertainty
    )

    influence = make_influence()

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert 0.0 <= result.base_explore_score <= 1.0
    assert 0.0 <= result.base_exploit_score <= 1.0


# ============================================================
# Validation
# ============================================================


def test_none_decision_is_rejected():
    arbiter = (
        SelfAwareDecisionArbiter()
    )

    influence = make_influence()

    with pytest.raises(ValueError):
        arbiter.evaluate(
            None,
            influence,
        )


def test_wrong_decision_type_is_rejected():
    arbiter = (
        SelfAwareDecisionArbiter()
    )

    influence = make_influence()

    with pytest.raises(ValueError):
        arbiter.evaluate(
            object(),
            influence,
        )


def test_none_influence_is_rejected():
    arbiter = (
        SelfAwareDecisionArbiter()
    )

    decision = make_decision()

    with pytest.raises(ValueError):
        arbiter.evaluate(
            decision,
            None,
        )


def test_wrong_influence_type_is_rejected():
    arbiter = (
        SelfAwareDecisionArbiter()
    )

    decision = make_decision()

    with pytest.raises(ValueError):
        arbiter.evaluate(
            decision,
            object(),
        )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot_contains_expected_fields():
    decision = make_decision()

    influence = make_influence()

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    result = arbiter.evaluate(
        decision,
        influence,
    )

    assert isinstance(
        result,
        SelfAwareDecisionResult,
    )

    snapshot = result.snapshot()

    expected = {
        "base_decision",
        "influence",
        "base_explore_score",
        "base_exploit_score",
        "final_explore_score",
        "final_exploit_score",
        "selected_mode",
        "applied",
        "reason",
        "mode_changed",
        "self_influence_active",
        "final_margin",
        "base_margin",
        "exploration_selected",
        "exploitation_selected",
    }

    assert expected.issubset(
        snapshot.keys()
    )


# ============================================================
# Convenience API
# ============================================================


def test_arbitrate_matches_evaluate():
    decision = make_decision()

    influence = make_influence()

    arbiter = (
        SelfAwareDecisionArbiter()
    )

    first = arbiter.evaluate(
        decision,
        influence,
    )

    second = arbiter.arbitrate(
        decision,
        influence,
    )

    assert first == second
