from decimal import Decimal

import pytest

from agent.learning.calibrated_self_awareness import (
    CalibratedSelfAwareInfluenceApplier,
    CalibratedSelfAwareInfluenceResult,
)
from agent.learning.self_awareness_calibration import (
    SelfAwarenessCalibrator,
)
from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SelfModelDecisionInfluence,
)


def make_influence(
    *,
    confidence=0.80,
    uncertainty=0.20,
    explore_bonus=0.05,
    exploit_bonus=0.20,
    preferred_mode=SELF_MODEL_MODE_EXPLOIT,
):
    return SelfModelDecisionInfluence(
        preferred_mode=preferred_mode,
        explore_bonus=explore_bonus,
        exploit_bonus=exploit_bonus,
        confidence=confidence,
        uncertainty=uncertainty,
        strength=1.0,
        applied=True,
        reason="test influence",
    )


def make_calibration(
    *,
    raw_confidence=0.80,
    attempts=10,
    successes=8,
    failures=2,
):
    calibrator = SelfAwarenessCalibrator()

    return calibrator.calibrate(
        raw_confidence=raw_confidence,
        evidence={
            "execution": {
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
            }
        },
    )


def test_default_max_bonus():
    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    assert applier.max_bonus == 0.25


def test_invalid_max_bonus_is_rejected():
    with pytest.raises(ValueError):
        CalibratedSelfAwareInfluenceApplier(
            max_bonus=0.0
        )

    with pytest.raises(ValueError):
        CalibratedSelfAwareInfluenceApplier(
            max_bonus=1.1
        )


def test_invalid_influence_type_is_rejected():
    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    calibration = make_calibration()

    with pytest.raises(TypeError):
        applier.apply(
            influence=None,
            calibration=calibration,
        )


def test_invalid_calibration_type_is_rejected():
    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    influence = make_influence()

    with pytest.raises(TypeError):
        applier.apply(
            influence=influence,
            calibration=None,
        )


def test_matching_calibration_keeps_influence_unchanged():
    influence = make_influence(
        confidence=0.80,
        uncertainty=0.20,
        explore_bonus=0.05,
        exploit_bonus=0.20,
    )

    calibration = make_calibration(
        raw_confidence=0.80,
        attempts=10,
        successes=8,
        failures=2,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert isinstance(
        result,
        CalibratedSelfAwareInfluenceResult,
    )

    assert not result.adjusted

    assert (
        result.influence.explore_bonus
        == influence.explore_bonus
    )

    assert (
        result.influence.exploit_bonus
        == influence.exploit_bonus
    )


def test_overconfidence_reduces_exploit_bonus():
    influence = make_influence(
        confidence=0.90,
        uncertainty=0.10,
        explore_bonus=0.0,
        exploit_bonus=0.25,
    )

    calibration = make_calibration(
        raw_confidence=0.90,
        attempts=20,
        successes=4,
        failures=16,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert result.adjusted

    assert (
        result.calibrated_confidence
        < result.raw_confidence
    )

    assert (
        result.calibrated_exploit_bonus
        < result.raw_exploit_bonus
    )


def test_overconfidence_increases_explore_bonus():
    influence = make_influence(
        confidence=0.90,
        uncertainty=0.10,
        explore_bonus=0.0,
        exploit_bonus=0.25,
    )

    calibration = make_calibration(
        raw_confidence=0.90,
        attempts=20,
        successes=4,
        failures=16,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert (
        result.calibrated_explore_bonus
        > result.raw_explore_bonus
    )


def test_underconfidence_increases_exploit_bonus():
    influence = make_influence(
        confidence=0.20,
        uncertainty=0.80,
        explore_bonus=0.25,
        exploit_bonus=0.0,
        preferred_mode=SELF_MODEL_MODE_EXPLORE,
    )

    calibration = make_calibration(
        raw_confidence=0.20,
        attempts=20,
        successes=18,
        failures=2,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert (
        result.calibrated_confidence
        > result.raw_confidence
    )

    assert (
        result.calibrated_exploit_bonus
        > result.raw_exploit_bonus
    )


def test_underconfidence_reduces_explore_bonus():
    influence = make_influence(
        confidence=0.20,
        uncertainty=0.80,
        explore_bonus=0.25,
        exploit_bonus=0.0,
        preferred_mode=SELF_MODEL_MODE_EXPLORE,
    )

    calibration = make_calibration(
        raw_confidence=0.20,
        attempts=20,
        successes=18,
        failures=2,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert (
        result.calibrated_explore_bonus
        < result.raw_explore_bonus
    )


def test_confidence_drop_can_flip_preferred_mode():
    influence = make_influence(
        confidence=0.90,
        uncertainty=0.10,
        explore_bonus=0.0,
        exploit_bonus=0.25,
        preferred_mode=SELF_MODEL_MODE_EXPLOIT,
    )

    calibration = make_calibration(
        raw_confidence=0.90,
        attempts=30,
        successes=3,
        failures=27,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert (
        result.influence.preferred_mode
        == SELF_MODEL_MODE_EXPLORE
    )


def test_confidence_increase_can_flip_preferred_mode():
    influence = make_influence(
        confidence=0.20,
        uncertainty=0.80,
        explore_bonus=0.25,
        exploit_bonus=0.0,
        preferred_mode=SELF_MODEL_MODE_EXPLORE,
    )

    calibration = make_calibration(
        raw_confidence=0.20,
        attempts=30,
        successes=27,
        failures=3,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert (
        result.influence.preferred_mode
        == SELF_MODEL_MODE_EXPLOIT
    )


def test_calibrated_uncertainty_is_complement():
    influence = make_influence()

    calibration = make_calibration(
        raw_confidence=0.80,
        attempts=20,
        successes=4,
        failures=16,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert (
        result.calibrated_uncertainty
        == round(
            1.0
            - result.calibrated_confidence,
            6,
        )
    )


def test_calibrated_bonuses_are_bounded():
    influence = make_influence(
        confidence=0.0,
        uncertainty=1.0,
        explore_bonus=0.25,
        exploit_bonus=0.0,
        preferred_mode=SELF_MODEL_MODE_EXPLORE,
    )

    calibration = make_calibration(
        raw_confidence=0.0,
        attempts=100,
        successes=100,
        failures=0,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert (
        0.0
        <= result.influence.explore_bonus
        <= 0.25
    )

    assert (
        0.0
        <= result.influence.exploit_bonus
        <= 0.25
    )


def test_original_influence_is_not_mutated():
    influence = make_influence()

    original_confidence = (
        influence.confidence
    )

    original_explore_bonus = (
        influence.explore_bonus
    )

    original_exploit_bonus = (
        influence.exploit_bonus
    )

    calibration = make_calibration(
        raw_confidence=0.80,
        attempts=20,
        successes=2,
        failures=18,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    applier.apply(
        influence,
        calibration,
    )

    assert (
        influence.confidence
        == original_confidence
    )

    assert (
        influence.explore_bonus
        == original_explore_bonus
    )

    assert (
        influence.exploit_bonus
        == original_exploit_bonus
    )


def test_result_exposes_snapshot():
    influence = make_influence()

    calibration = make_calibration()

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    snapshot = result.snapshot

    assert isinstance(
        snapshot,
        dict,
    )

    assert (
        "raw_confidence"
        in snapshot
    )

    assert (
        "calibrated_confidence"
        in snapshot
    )

    assert (
        "calibrated_explore_bonus"
        in snapshot
    )

    assert (
        "calibrated_exploit_bonus"
        in snapshot
    )


def test_confidence_delta_is_reported():
    influence = make_influence()

    calibration = make_calibration(
        raw_confidence=0.80,
        attempts=20,
        successes=2,
        failures=18,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert (
        result.confidence_delta
        == round(
            result.calibrated_confidence
            - result.raw_confidence,
            6,
        )
    )


def test_reason_mentions_calibration_when_adjusted():
    influence = make_influence()

    calibration = make_calibration(
        raw_confidence=0.80,
        attempts=20,
        successes=2,
        failures=18,
    )

    applier = (
        CalibratedSelfAwareInfluenceApplier()
    )

    result = applier.apply(
        influence,
        calibration,
    )

    assert result.adjusted

    assert (
        "calibrated confidence"
        in result.influence.reason
    )


def test_decimal_max_bonus_is_accepted():
    applier = (
        CalibratedSelfAwareInfluenceApplier(
            max_bonus=Decimal("0.20")
        )
    )

    assert (
        applier.max_bonus
        == 0.20
    )
