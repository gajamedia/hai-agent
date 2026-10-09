import pytest

from agent.learning.self_model_behavioral_evidence import (
    SelfModelBehavioralEvidenceModel,
    build_self_model_behavioral_evidence,
)


def test_empty_capabilities_have_no_evidence():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze({})

    assert not result.has_evidence

    assert (
        result.observed_capability_count
        == 0
    )

    assert (
        result.strongest_capability
        is None
    )

    assert (
        result.weakest_capability
        is None
    )


def test_capability_success_rate_is_calculated():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
            }
        }
    )

    capability = (
        result.get_capability(
            "planning"
        )
    )

    assert capability is not None

    assert (
        capability.success_rate
        == pytest.approx(
            0.8,
            rel=1e-6,
        )
    )


def test_capability_confidence_defaults_from_success_rate():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
            }
        }
    )

    capability = (
        result.get_capability(
            "planning"
        )
    )

    assert capability is not None

    assert (
        capability.confidence
        == 0.8
    )


def test_capability_uncertainty_defaults_from_success_rate():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
            }
        }
    )

    capability = (
        result.get_capability(
            "planning"
        )
    )

    assert capability is not None

    assert (
        capability.uncertainty
        == 0.2
    )


def test_explicit_confidence_and_uncertainty_are_preserved():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "reasoning": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
                "confidence": 0.9,
                "uncertainty": 0.1,
            }
        }
    )

    capability = (
        result.get_capability(
            "reasoning"
        )
    )

    assert capability is not None

    assert capability.confidence == 0.9
    assert capability.uncertainty == 0.1


def test_evidence_strength_increases_with_more_attempts():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    low = model.analyze(
        {
            "planning": {
                "attempts": 2,
                "successes": 2,
                "failures": 0,
            }
        }
    )

    high = model.analyze(
        {
            "planning": {
                "attempts": 20,
                "successes": 20,
                "failures": 0,
            }
        }
    )

    low_capability = (
        low.get_capability(
            "planning"
        )
    )

    high_capability = (
        high.get_capability(
            "planning"
        )
    )

    assert low_capability is not None
    assert high_capability is not None

    assert (
        high_capability.evidence_strength
        > low_capability.evidence_strength
    )


def test_multiple_capabilities_are_aggregated():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 9,
                "failures": 1,
            },
            "exploration": {
                "attempts": 10,
                "successes": 4,
                "failures": 6,
            },
        }
    )

    assert (
        result.capability_count
        == 2
    )

    assert (
        result.observed_capability_count
        == 2
    )

    assert result.has_multiple_capabilities


def test_strongest_capability_is_identified():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 9,
                "failures": 1,
            },
            "exploration": {
                "attempts": 10,
                "successes": 4,
                "failures": 6,
            },
        }
    )

    assert (
        result.strongest_capability
        == "planning"
    )


def test_weakest_capability_is_identified():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 9,
                "failures": 1,
            },
            "exploration": {
                "attempts": 10,
                "successes": 4,
                "failures": 6,
            },
        }
    )

    assert (
        result.weakest_capability
        == "exploration"
    )


def test_high_quality_capability_is_reliable():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 9,
                "failures": 1,
                "confidence": 0.9,
                "uncertainty": 0.1,
            }
        }
    )

    capability = (
        result.get_capability(
            "planning"
        )
    )

    assert capability is not None
    assert capability.reliable
    assert not capability.weak


def test_low_quality_capability_is_weak():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 3,
                "failures": 7,
                "confidence": 0.3,
                "uncertainty": 0.8,
            }
        }
    )

    capability = (
        result.get_capability(
            "planning"
        )
    )

    assert capability is not None
    assert capability.weak


def test_aggregate_confidence_is_weighted_by_attempts():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "small": {
                "attempts": 1,
                "successes": 1,
                "failures": 0,
                "confidence": 1.0,
                "uncertainty": 0.0,
            },
            "large": {
                "attempts": 9,
                "successes": 9,
                "failures": 0,
                "confidence": 0.5,
                "uncertainty": 0.5,
            },
        }
    )

    assert (
        result.average_confidence
        == pytest.approx(
            0.55,
            rel=1e-6,
        )
    )


def test_stable_self_model_evidence():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
                "confidence": 0.8,
                "uncertainty": 0.2,
            }
        }
    )

    assert result.is_stable


def test_high_uncertainty_is_detected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 3,
                "failures": 7,
                "confidence": 0.3,
                "uncertainty": 0.8,
            }
        }
    )

    assert result.uncertainty_high


def test_snapshot_contains_all_evidence():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
            }
        }
    )

    snapshot = result.snapshot()

    assert (
        "capabilities"
        in snapshot
    )

    assert (
        "average_confidence"
        in snapshot
    )

    assert (
        "average_uncertainty"
        in snapshot
    )

    assert (
        "average_success_rate"
        in snapshot
    )

    assert (
        "evidence_strength"
        in snapshot
    )

    assert (
        "strongest_capability"
        in snapshot
    )

    assert (
        "weakest_capability"
        in snapshot
    )


def test_object_capability_data_is_supported():
    class CapabilityData:
        attempts = 10
        successes = 7
        failures = 3
        confidence = 0.7
        uncertainty = 0.3

    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": CapabilityData()
        }
    )

    capability = (
        result.get_capability(
            "planning"
        )
    )

    assert capability is not None

    assert capability.attempts == 10
    assert capability.successes == 7
    assert capability.failures == 3


def test_convenience_builder():
    result = (
        build_self_model_behavioral_evidence(
            {
                "planning": {
                    "attempts": 5,
                    "successes": 4,
                    "failures": 1,
                }
            }
        )
    )

    assert result.has_evidence
    assert (
        result.strongest_capability
        == "planning"
    )


def test_none_capabilities_are_rejected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="capabilities cannot be None",
    ):
        model.analyze(
            None
        )


def test_non_mapping_capabilities_are_rejected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="capabilities must be a mapping",
    ):
        model.analyze(
            []
        )


def test_negative_counts_are_rejected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="attempts cannot be negative",
    ):
        model.analyze(
            {
                "planning": {
                    "attempts": -1,
                    "successes": 0,
                    "failures": 0,
                }
            }
        )

    with pytest.raises(
        ValueError,
        match="successes cannot be negative",
    ):
        model.analyze(
            {
                "planning": {
                    "attempts": 1,
                    "successes": -1,
                    "failures": 0,
                }
            }
        )

    with pytest.raises(
        ValueError,
        match="failures cannot be negative",
    ):
        model.analyze(
            {
                "planning": {
                    "attempts": 1,
                    "successes": 0,
                    "failures": -1,
                }
            }
        )


def test_inconsistent_counts_are_rejected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed attempts",
    ):
        model.analyze(
            {
                "planning": {
                    "attempts": 2,
                    "successes": 2,
                    "failures": 1,
                }
            }
        )


def test_invalid_confidence_is_rejected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="confidence must be between 0 and 1",
    ):
        model.analyze(
            {
                "planning": {
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "confidence": 1.5,
                }
            }
        )


def test_invalid_uncertainty_is_rejected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="uncertainty must be between 0 and 1",
    ):
        model.analyze(
            {
                "planning": {
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "uncertainty": -0.1,
                }
            }
        )


def test_invalid_minimum_evidence_is_rejected():
    with pytest.raises(
        ValueError,
        match="minimum_evidence must be >= 1",
    ):
        SelfModelBehavioralEvidenceModel(
            minimum_evidence=0
        )


def test_empty_capability_name_is_rejected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="capability cannot be empty",
    ):
        model.analyze(
            {
                "": {
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                }
            }
        )


def test_unknown_capability_data_shape_is_rejected():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="capability data must be",
    ):
        model.analyze(
            {
                "planning": 123
            }
        )
