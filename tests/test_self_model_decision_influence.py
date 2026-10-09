import pytest

from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SELF_MODEL_MODE_NEUTRAL,
    SelfModelDecisionInfluenceIntegrator,
    build_self_model_decision_influence,
)

from agent.learning.self_uncertainty import (
    SelfUncertaintyResult,
)


# ============================================================
# Helpers
# ============================================================


def make_result(
    *,
    uncertainty=0.50,
    confidence=0.50,
    evidence_strength=0.50,
    capability_count=1,
    observed_capability_count=1,
    high_uncertainty_count=0,
    low_confidence_count=0,
):
    return SelfUncertaintyResult(
        uncertainty=uncertainty,
        confidence=confidence,
        evidence_strength=evidence_strength,
        average_capability_uncertainty=uncertainty,
        average_capability_confidence=confidence,
        confidence_dispersion=0.0,
        capability_count=capability_count,
        observed_capability_count=(
            observed_capability_count
        ),
        high_uncertainty_count=(
            high_uncertainty_count
        ),
        low_confidence_count=(
            low_confidence_count
        ),
        strongest_capability="planning",
        weakest_capability="planning",
    )


def make_no_evidence_result():
    return SelfUncertaintyResult(
        uncertainty=1.0,
        confidence=0.0,
        evidence_strength=0.0,
        average_capability_uncertainty=1.0,
        average_capability_confidence=0.0,
        confidence_dispersion=0.0,
        capability_count=0,
        observed_capability_count=0,
        high_uncertainty_count=0,
        low_confidence_count=0,
        strongest_capability=None,
        weakest_capability=None,
    )


# ============================================================
# Basic Behavior
# ============================================================


def test_high_self_uncertainty_prefers_exploration():
    result = make_result(
        uncertainty=0.90,
        confidence=0.10,
        evidence_strength=0.80,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert (
        influence.preferred_mode
        == SELF_MODEL_MODE_EXPLORE
    )

    assert influence.should_explore
    assert not influence.should_exploit


def test_high_self_confidence_prefers_exploitation():
    result = make_result(
        uncertainty=0.10,
        confidence=0.90,
        evidence_strength=0.90,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert (
        influence.preferred_mode
        == SELF_MODEL_MODE_EXPLOIT
    )

    assert influence.should_exploit
    assert not influence.should_explore


def test_balanced_self_model_can_be_neutral():
    result = make_result(
        uncertainty=0.50,
        confidence=0.50,
        evidence_strength=1.00,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert (
        influence.preferred_mode
        == SELF_MODEL_MODE_NEUTRAL
    )

    assert influence.is_neutral
    assert not influence.applied


# ============================================================
# No Evidence
# ============================================================


def test_no_self_model_evidence_prefers_exploration():
    result = make_no_evidence_result()

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert (
        influence.preferred_mode
        == SELF_MODEL_MODE_EXPLORE
    )

    assert influence.explore_bonus == 0.25
    assert influence.exploit_bonus == 0.0

    assert influence.should_explore
    assert influence.applied


# ============================================================
# Bonus Direction
# ============================================================


def test_uncertainty_increases_explore_bonus():
    low = make_result(
        uncertainty=0.20,
        confidence=0.80,
        evidence_strength=0.80,
    )

    high = make_result(
        uncertainty=0.80,
        confidence=0.20,
        evidence_strength=0.80,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    low_result = integrator.assess(low)
    high_result = integrator.assess(high)

    assert (
        high_result.explore_bonus
        > low_result.explore_bonus
    )


def test_confidence_and_evidence_increase_exploit_bonus():
    weak = make_result(
        uncertainty=0.30,
        confidence=0.70,
        evidence_strength=0.20,
    )

    strong = make_result(
        uncertainty=0.30,
        confidence=0.70,
        evidence_strength=0.90,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    weak_result = integrator.assess(weak)
    strong_result = integrator.assess(strong)

    assert (
        strong_result.exploit_bonus
        > weak_result.exploit_bonus
    )


def test_low_evidence_reduces_exploit_bonus():
    high_evidence = make_result(
        uncertainty=0.20,
        confidence=0.80,
        evidence_strength=0.90,
    )

    low_evidence = make_result(
        uncertainty=0.20,
        confidence=0.80,
        evidence_strength=0.20,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    high_result = integrator.assess(
        high_evidence
    )

    low_result = integrator.assess(
        low_evidence
    )

    assert (
        low_result.exploit_bonus
        < high_result.exploit_bonus
    )


# ============================================================
# Bounds
# ============================================================


def test_bonuses_are_bounded():
    result = make_result(
        uncertainty=1.0,
        confidence=1.0,
        evidence_strength=1.0,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert 0.0 <= influence.explore_bonus <= 0.25
    assert 0.0 <= influence.exploit_bonus <= 0.25


def test_strength_is_bounded():
    result = make_result(
        uncertainty=0.90,
        confidence=0.10,
        evidence_strength=0.90,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert 0.0 <= influence.strength <= 1.0


# ============================================================
# Directional Properties
# ============================================================


def test_exploration_bonus_matches_uncertainty():
    result = make_result(
        uncertainty=0.80,
        confidence=0.20,
        evidence_strength=0.50,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator(
            max_bonus=0.20,
        )
    )

    influence = integrator.assess(
        result
    )

    assert influence.explore_bonus == pytest.approx(
        0.16,
        abs=1e-6,
    )


def test_exploitation_bonus_uses_confidence_and_evidence():
    result = make_result(
        uncertainty=0.20,
        confidence=0.80,
        evidence_strength=0.50,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator(
            max_bonus=0.20,
        )
    )

    influence = integrator.assess(
        result
    )

    assert influence.exploit_bonus == pytest.approx(
        0.08,
        abs=1e-6,
    )


# ============================================================
# Strength
# ============================================================


def test_max_bonus_produces_full_strength():
    result = make_result(
        uncertainty=1.0,
        confidence=0.0,
        evidence_strength=0.0,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert influence.strength == 1.0


def test_half_max_bonus_produces_half_strength():
    result = make_result(
        uncertainty=0.50,
        confidence=0.50,
        evidence_strength=0.0,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator(
            minimum_pressure=0.01,
        )
    )

    influence = integrator.assess(
        result
    )

    assert influence.strength == pytest.approx(
        0.50,
        abs=1e-6,
    )


# ============================================================
# Properties
# ============================================================


def test_net_bonus_is_exploration_minus_exploitation():
    result = make_result(
        uncertainty=0.80,
        confidence=0.20,
        evidence_strength=0.50,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert influence.net_bonus == pytest.approx(
        influence.explore_bonus
        - influence.exploit_bonus,
        abs=1e-6,
    )


def test_strongest_bonus_is_maximum():
    result = make_result(
        uncertainty=0.20,
        confidence=0.80,
        evidence_strength=0.90,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    assert influence.strongest_bonus == max(
        influence.explore_bonus,
        influence.exploit_bonus,
    )


def test_should_explore_requires_applied():
    result = make_result(
        uncertainty=0.50,
        confidence=0.50,
        evidence_strength=1.00,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator(
            minimum_pressure=0.0,
            minimum_mode_gap=1.0,
        )
    )

    influence = integrator.assess(
        result
    )

    assert not influence.should_explore
    assert not influence.should_exploit


# ============================================================
# Threshold Configuration
# ============================================================


def test_minimum_pressure_can_disable_weak_influence():
    result = make_result(
        uncertainty=0.10,
        confidence=0.90,
        evidence_strength=0.90,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator(
            minimum_pressure=0.25,
        )
    )

    influence = integrator.assess(
        result
    )

    assert not influence.applied
    assert influence.is_neutral


def test_mode_gap_can_force_neutral_result():
    result = make_result(
        uncertainty=0.60,
        confidence=0.40,
        evidence_strength=0.50,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator(
            minimum_pressure=0.01,
            minimum_mode_gap=0.25,
        )
    )

    influence = integrator.assess(
        result
    )

    assert influence.is_neutral
    assert not influence.applied


# ============================================================
# Validation
# ============================================================


def test_none_result_is_rejected():
    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    with pytest.raises(ValueError):
        integrator.assess(None)


def test_wrong_result_type_is_rejected():
    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    with pytest.raises(ValueError):
        integrator.assess(
            object()
        )


def test_invalid_max_bonus_is_rejected():
    with pytest.raises(ValueError):
        SelfModelDecisionInfluenceIntegrator(
            max_bonus=1.5,
        )


def test_invalid_minimum_pressure_is_rejected():
    with pytest.raises(ValueError):
        SelfModelDecisionInfluenceIntegrator(
            minimum_pressure=-0.1,
        )


def test_minimum_pressure_cannot_exceed_max_bonus():
    with pytest.raises(ValueError):
        SelfModelDecisionInfluenceIntegrator(
            max_bonus=0.10,
            minimum_pressure=0.20,
        )

def test_minimum_mode_gap_can_exceed_max_bonus():
    integrator = (
        SelfModelDecisionInfluenceIntegrator(
            max_bonus=0.10,
            minimum_mode_gap=0.20,
        )
    )

    assert (
        integrator.minimum_mode_gap
        == 0.20
    )

def test_minimum_mode_gap_above_one_is_rejected():
    with pytest.raises(ValueError):
        SelfModelDecisionInfluenceIntegrator(
            minimum_mode_gap=1.01,
        )

# ============================================================
# Convenience API
# ============================================================


def test_influence_for_matches_assess():
    result = make_result(
        uncertainty=0.80,
        confidence=0.20,
        evidence_strength=0.80,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    first = integrator.assess(
        result
    )

    second = integrator.influence_for(
        result
    )

    assert first == second


def test_convenience_builder():
    result = make_result(
        uncertainty=0.90,
        confidence=0.10,
        evidence_strength=0.80,
    )

    influence = (
        build_self_model_decision_influence(
            result
        )
    )

    assert (
        influence.preferred_mode
        == SELF_MODEL_MODE_EXPLORE
    )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot_contains_all_fields():
    result = make_result(
        uncertainty=0.80,
        confidence=0.20,
        evidence_strength=0.80,
    )

    integrator = (
        SelfModelDecisionInfluenceIntegrator()
    )

    influence = integrator.assess(
        result
    )

    snapshot = influence.snapshot()

    expected_keys = {
        "preferred_mode",
        "explore_bonus",
        "exploit_bonus",
        "confidence",
        "uncertainty",
        "strength",
        "applied",
        "reason",
        "should_explore",
        "should_exploit",
        "is_neutral",
        "net_bonus",
        "strongest_bonus",
    }

    assert expected_keys.issubset(
        snapshot.keys()
    )
