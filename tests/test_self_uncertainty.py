import pytest

from agent.learning.capability_confidence import (
    CapabilityConfidenceIntegrator,
)
from agent.learning.self_model_behavioral_evidence import (
    SelfModelBehavioralEvidenceModel,
)
from agent.learning.self_uncertainty import (
    SelfUncertaintyIntegrator,
    build_self_uncertainty,
)


def make_summary(
    capabilities,
):
    evidence_model = (
        SelfModelBehavioralEvidenceModel()
    )

    evidence = (
        evidence_model.analyze(
            capabilities
        )
    )

    confidence_integrator = (
        CapabilityConfidenceIntegrator()
    )

    return (
        confidence_integrator.assess_evidence(
            evidence
        )
    )


def test_no_self_model_evidence_means_high_self_uncertainty():
    summary = make_summary({})

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        result.uncertainty
        == 1.0
    )

    assert (
        result.confidence
        == 0.0
    )

    assert not result.has_evidence


def test_strong_capability_evidence_reduces_self_uncertainty():
    summary = make_summary(
        {
            "planning": {
                "attempts": 20,
                "successes": 19,
                "failures": 1,
                "confidence": 0.95,
                "uncertainty": 0.05,
            }
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        result.uncertainty
        < 0.30
    )

    assert (
        result.confidence
        > 0.70
    )


def test_weak_capability_evidence_increases_self_uncertainty():
    summary = make_summary(
        {
            "planning": {
                "attempts": 10,
                "successes": 2,
                "failures": 8,
                "confidence": 0.2,
                "uncertainty": 0.8,
            }
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        result.uncertainty
        > 0.60
    )

    assert (
        result.confidence
        < 0.40
    )


def test_more_evidence_reduces_uncertainty():
    low = make_summary(
        {
            "planning": {
                "attempts": 2,
                "successes": 2,
                "failures": 0,
                "confidence": 0.9,
                "uncertainty": 0.1,
            }
        }
    )

    high = make_summary(
        {
            "planning": {
                "attempts": 20,
                "successes": 20,
                "failures": 0,
                "confidence": 0.9,
                "uncertainty": 0.1,
            }
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    low_result = integrator.assess(
        low
    )

    high_result = integrator.assess(
        high
    )

    assert (
        high_result.uncertainty
        < low_result.uncertainty
    )


def test_multiple_capabilities_are_aggregated():
    summary = make_summary(
        {
            "planning": {
                "attempts": 10,
                "successes": 9,
                "failures": 1,
                "confidence": 0.9,
                "uncertainty": 0.1,
            },
            "reasoning": {
                "attempts": 10,
                "successes": 4,
                "failures": 6,
                "confidence": 0.4,
                "uncertainty": 0.6,
            },
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        result.capability_count
        == 2
    )

    assert (
        result.observed_capability_count
        == 2
    )


def test_high_uncertainty_capabilities_are_counted():
    summary = make_summary(
        {
            "planning": {
                "attempts": 10,
                "successes": 9,
                "failures": 1,
                "confidence": 0.9,
                "uncertainty": 0.1,
            },
            "exploration": {
                "attempts": 10,
                "successes": 2,
                "failures": 8,
                "confidence": 0.2,
                "uncertainty": 0.8,
            },
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        result.high_uncertainty_count
        == 1
    )


def test_low_confidence_capabilities_are_counted():
    summary = make_summary(
        {
            "planning": {
                "attempts": 10,
                "successes": 2,
                "failures": 8,
                "confidence": 0.2,
                "uncertainty": 0.8,
            },
            "reasoning": {
                "attempts": 10,
                "successes": 3,
                "failures": 7,
                "confidence": 0.3,
                "uncertainty": 0.7,
            },
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        result.low_confidence_count
        == 2
    )


def test_strongest_and_weakest_capabilities_are_preserved():
    summary = make_summary(
        {
            "planning": {
                "attempts": 10,
                "successes": 9,
                "failures": 1,
                "confidence": 0.9,
                "uncertainty": 0.1,
            },
            "exploration": {
                "attempts": 10,
                "successes": 3,
                "failures": 7,
                "confidence": 0.3,
                "uncertainty": 0.7,
            },
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        result.strongest_capability
        == "planning"
    )

    assert (
        result.weakest_capability
        == "exploration"
    )


def test_confidence_and_uncertainty_are_bounded():
    summary = make_summary(
        {
            "planning": {
                "attempts": 100,
                "successes": 100,
                "failures": 0,
                "confidence": 1.0,
                "uncertainty": 0.0,
            }
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        0.0
        <= result.uncertainty
        <= 1.0
    )

    assert (
        0.0
        <= result.confidence
        <= 1.0
    )


def test_confidence_is_complement_of_uncertainty():
    summary = make_summary(
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

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert (
        result.confidence
        == round(
            1.0 - result.uncertainty,
            6,
        )
    )


def test_stable_self_knowledge_is_detected():
    summary = make_summary(
        {
            "planning": {
                "attempts": 20,
                "successes": 18,
                "failures": 2,
                "confidence": 0.9,
                "uncertainty": 0.1,
            }
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert result.self_knowledge_stable


def test_high_uncertainty_is_detected():
    summary = make_summary(
        {
            "planning": {
                "attempts": 10,
                "successes": 1,
                "failures": 9,
                "confidence": 0.1,
                "uncertainty": 0.9,
            }
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert result.uncertainty_high


def test_low_uncertainty_is_detected():
    summary = make_summary(
        {
            "planning": {
                "attempts": 100,
                "successes": 100,
                "failures": 0,
                "confidence": 1.0,
                "uncertainty": 0.0,
            }
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    assert result.uncertainty_low


def test_assess_capabilities_accepts_assessment_sequence():
    summary = make_summary(
        {
            "planning": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
                "confidence": 0.8,
                "uncertainty": 0.2,
            },
            "reasoning": {
                "attempts": 10,
                "successes": 6,
                "failures": 4,
                "confidence": 0.6,
                "uncertainty": 0.4,
            },
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = (
        integrator.assess_capabilities(
            summary.assessments
        )
    )

    assert (
        result.observed_capability_count
        == 2
    )


def test_empty_assessment_sequence_has_maximum_uncertainty():
    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = (
        integrator.assess_capabilities(
            tuple()
        )
    )

    assert (
        result.uncertainty
        == 1.0
    )

    assert (
        result.confidence
        == 0.0
    )


def test_convenience_builder():
    summary = make_summary(
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

    result = build_self_uncertainty(
        summary
    )

    assert result.has_evidence
    assert 0.0 <= result.uncertainty <= 1.0


def test_snapshot_contains_all_fields():
    summary = make_summary(
        {
            "planning": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
            }
        }
    )

    integrator = (
        SelfUncertaintyIntegrator()
    )

    result = integrator.assess(
        summary
    )

    snapshot = result.snapshot()

    assert "uncertainty" in snapshot
    assert "confidence" in snapshot
    assert "evidence_strength" in snapshot
    assert (
        "average_capability_uncertainty"
        in snapshot
    )
    assert (
        "average_capability_confidence"
        in snapshot
    )
    assert (
        "confidence_dispersion"
        in snapshot
    )
    assert (
        "capability_count"
        in snapshot
    )
    assert (
        "observed_capability_count"
        in snapshot
    )


def test_none_summary_is_rejected():
    integrator = (
        SelfUncertaintyIntegrator()
    )

    with pytest.raises(
        ValueError,
        match="summary cannot be None",
    ):
        integrator.assess(
            None
        )


def test_wrong_summary_type_is_rejected():
    integrator = (
        SelfUncertaintyIntegrator()
    )

    with pytest.raises(
        ValueError,
        match="CapabilityConfidenceSummary",
    ):
        integrator.assess(
            {}
        )


def test_none_assessments_are_rejected():
    integrator = (
        SelfUncertaintyIntegrator()
    )

    with pytest.raises(
        ValueError,
        match="assessments cannot be None",
    ):
        integrator.assess_capabilities(
            None
        )


def test_weights_must_sum_to_one():
    with pytest.raises(
        ValueError,
        match="weights must sum to 1",
    ):
        SelfUncertaintyIntegrator(
            capability_uncertainty_weight=0.5,
            evidence_deficit_weight=0.5,
            dispersion_weight=0.5,
        )


def test_weights_must_be_bounded():
    with pytest.raises(ValueError):
        SelfUncertaintyIntegrator(
            capability_uncertainty_weight=-0.1,
            evidence_deficit_weight=0.6,
            dispersion_weight=0.5,
        )

    with pytest.raises(ValueError):
        SelfUncertaintyIntegrator(
            capability_uncertainty_weight=1.1,
            evidence_deficit_weight=0.0,
            dispersion_weight=-0.1,
        )
