import pytest

from agent.learning.self_awareness_calibration import (
    SelfAwarenessCalibrationResult,
    SelfAwarenessCalibrator,
)


def test_default_minimum_evidence():
    calibrator = SelfAwarenessCalibrator()

    assert calibrator.minimum_evidence == 5


def test_custom_minimum_evidence():
    calibrator = SelfAwarenessCalibrator(
        minimum_evidence=10
    )

    assert calibrator.minimum_evidence == 10


def test_minimum_evidence_must_be_positive():
    with pytest.raises(ValueError):
        SelfAwarenessCalibrator(
            minimum_evidence=0
        )


def test_empty_evidence_preserves_raw_confidence():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.8,
        evidence={},
    )

    assert isinstance(
        result,
        SelfAwarenessCalibrationResult,
    )

    assert result.raw_confidence == 0.8
    assert result.observed_success_rate == 0.8
    assert result.calibrated_confidence == 0.8
    assert result.evidence_strength == 0.0
    assert result.total_attempts == 0


def test_strong_success_evidence_increases_calibrated_confidence():
    calibrator = SelfAwarenessCalibrator(
        minimum_evidence=5
    )

    result = calibrator.calibrate(
        raw_confidence=0.40,
        evidence={
            "execution": {
                "attempts": 20,
                "successes": 20,
                "failures": 0,
            }
        },
    )

    assert (
        result.calibrated_confidence
        > result.raw_confidence
    )

    assert (
        result.observed_success_rate
        == 1.0
    )

    assert (
        result.underconfidence
        > 0.0
    )


def test_strong_failure_evidence_reduces_calibrated_confidence():
    calibrator = SelfAwarenessCalibrator(
        minimum_evidence=5
    )

    result = calibrator.calibrate(
        raw_confidence=0.90,
        evidence={
            "execution": {
                "attempts": 20,
                "successes": 0,
                "failures": 20,
            }
        },
    )

    assert (
        result.calibrated_confidence
        < result.raw_confidence
    )

    assert (
        result.observed_success_rate
        == 0.0
    )

    assert (
        result.overconfidence
        > 0.0
    )


def test_more_evidence_produces_stronger_calibration():
    calibrator = SelfAwarenessCalibrator(
        minimum_evidence=5
    )

    small = calibrator.calibrate(
        raw_confidence=0.90,
        evidence={
            "execution": {
                "attempts": 1,
                "successes": 0,
                "failures": 1,
            }
        },
    )

    large = calibrator.calibrate(
        raw_confidence=0.90,
        evidence={
            "execution": {
                "attempts": 50,
                "successes": 0,
                "failures": 50,
            }
        },
    )

    assert (
        large.calibrated_confidence
        < small.calibrated_confidence
    )

    assert (
        large.evidence_strength
        > small.evidence_strength
    )


def test_matching_confidence_is_calibrated():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.80,
        evidence={
            "execution": {
                "attempts": 20,
                "successes": 16,
                "failures": 4,
            }
        },
    )

    assert (
        result.observed_success_rate
        == 0.8
    )

    assert (
        result.calibrated_confidence
        == 0.8
    )

    assert result.calibration_gap == 0.0
    assert result.is_calibrated


def test_calibrated_uncertainty_is_complement():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.60,
        evidence={
            "execution": {
                "attempts": 10,
                "successes": 5,
                "failures": 5,
            }
        },
    )

    assert (
        result.calibrated_uncertainty
        == round(
            1.0
            - result.calibrated_confidence,
            6,
        )
    )


def test_overconfidence_is_detected():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.95,
        evidence={
            "execution": {
                "attempts": 30,
                "successes": 6,
                "failures": 24,
            }
        },
    )

    assert result.is_overconfident
    assert not result.is_underconfident


def test_underconfidence_is_detected():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.10,
        evidence={
            "execution": {
                "attempts": 30,
                "successes": 27,
                "failures": 3,
            }
        },
    )

    assert result.is_underconfident
    assert not result.is_overconfident


def test_calibrated_state_can_be_considered_consistent():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.70,
        evidence={
            "execution": {
                "attempts": 10,
                "successes": 7,
                "failures": 3,
            }
        },
    )

    assert result.confidence_consistent


def test_multiple_capabilities_are_weighted_by_attempts():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.50,
        evidence={
            "planning": {
                "attempts": 1,
                "successes": 1,
                "failures": 0,
            },
            "execution": {
                "attempts": 9,
                "successes": 0,
                "failures": 9,
            },
        },
    )

    assert (
        result.total_attempts
        == 10
    )

    assert (
        result.total_successes
        == 1
    )

    assert (
        result.total_failures
        == 9
    )

    assert (
        result.observed_success_rate
        == 0.1
    )


def test_invalid_raw_confidence_is_rejected():
    calibrator = SelfAwarenessCalibrator()

    with pytest.raises(ValueError):
        calibrator.calibrate(
            raw_confidence=1.1,
            evidence={},
        )

    with pytest.raises(ValueError):
        calibrator.calibrate(
            raw_confidence=-0.1,
            evidence={},
        )


def test_invalid_raw_confidence_type_is_rejected():
    calibrator = SelfAwarenessCalibrator()

    with pytest.raises(TypeError):
        calibrator.calibrate(
            raw_confidence="high",
            evidence={},
        )


def test_invalid_evidence_type_is_rejected():
    calibrator = SelfAwarenessCalibrator()

    with pytest.raises(TypeError):
        calibrator.calibrate(
            raw_confidence=0.5,
            evidence=[],
        )


def test_negative_evidence_is_rejected():
    calibrator = SelfAwarenessCalibrator()

    with pytest.raises(ValueError):
        calibrator.calibrate(
            raw_confidence=0.5,
            evidence={
                "execution": {
                    "attempts": -1,
                    "successes": 0,
                    "failures": 0,
                }
            },
        )


def test_success_plus_failure_cannot_exceed_attempts():
    calibrator = SelfAwarenessCalibrator()

    with pytest.raises(ValueError):
        calibrator.calibrate(
            raw_confidence=0.5,
            evidence={
                "execution": {
                    "attempts": 5,
                    "successes": 4,
                    "failures": 4,
                }
            },
        )


def test_snapshot_is_serializable_dict():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.7,
        evidence={
            "execution": {
                "attempts": 10,
                "successes": 7,
                "failures": 3,
            }
        },
    )

    snapshot = result.snapshot

    assert isinstance(
        snapshot,
        dict,
    )

    assert (
        snapshot["raw_confidence"]
        == 0.7
    )

    assert (
        "calibrated_confidence"
        in snapshot
    )

    assert (
        "evidence_strength"
        in snapshot
    )


def test_confidence_shift_matches_calibration():
    calibrator = SelfAwarenessCalibrator()

    result = calibrator.calibrate(
        raw_confidence=0.90,
        evidence={
            "execution": {
                "attempts": 10,
                "successes": 5,
                "failures": 5,
            }
        },
    )

    assert (
        result.confidence_shift
        == round(
            result.calibrated_confidence
            - result.raw_confidence,
            6,
        )
    )
