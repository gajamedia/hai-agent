import pytest

from agent.learning.capability_confidence import (
    CapabilityConfidenceIntegrator,
)
from agent.learning.self_model_behavioral_evidence import (
    SelfModelBehavioralEvidenceModel,
)


def make_evidence(
    attempts=10,
    successes=8,
    failures=2,
    confidence=0.8,
    uncertainty=0.2,
):
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    result = model.analyze(
        {
            "planning": {
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
                "confidence": confidence,
                "uncertainty": uncertainty,
            }
        }
    )

    return result.get_capability(
        "planning"
    )


def make_multi_evidence():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    return model.analyze(
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
                "successes": 4,
                "failures": 6,
                "confidence": 0.4,
                "uncertainty": 0.6,
            },
        }
    )


def test_high_quality_evidence_produces_high_confidence():
    evidence = make_evidence(
        attempts=20,
        successes=18,
        failures=2,
        confidence=0.9,
        uncertainty=0.1,
    )

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    assessment = integrator.assess(
        evidence
    )

    assert (
        assessment.confidence
        > 0.80
    )

    assert (
        assessment.uncertainty
        < 0.20
    )


def test_low_quality_evidence_produces_lower_confidence():
    evidence = make_evidence(
        attempts=10,
        successes=2,
        failures=8,
        confidence=0.2,
        uncertainty=0.8,
    )

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    assessment = integrator.assess(
        evidence
    )

    assert (
        assessment.confidence
        < 0.40
    )

    assert (
        assessment.uncertainty
        > 0.60
    )


def test_more_evidence_increases_confidence():
    low = make_evidence(
        attempts=2,
        successes=2,
        failures=0,
        confidence=0.8,
        uncertainty=0.2,
    )

    high = make_evidence(
        attempts=20,
        successes=20,
        failures=0,
        confidence=0.8,
        uncertainty=0.2,
    )

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    low_result = (
        integrator.assess(low)
    )

    high_result = (
        integrator.assess(high)
    )

    assert (
        high_result.confidence
        > low_result.confidence
    )


def test_success_rate_contributes_to_confidence():
    weak = make_evidence(
        attempts=10,
        successes=2,
        failures=8,
        confidence=0.5,
        uncertainty=0.5,
    )

    strong = make_evidence(
        attempts=10,
        successes=9,
        failures=1,
        confidence=0.5,
        uncertainty=0.5,
    )

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    weak_result = (
        integrator.assess(weak)
    )

    strong_result = (
        integrator.assess(strong)
    )

    assert (
        strong_result.confidence
        > weak_result.confidence
    )


def test_source_confidence_contributes_to_result():
    low = make_evidence(
        attempts=10,
        successes=8,
        failures=2,
        confidence=0.2,
        uncertainty=0.8,
    )

    high = make_evidence(
        attempts=10,
        successes=8,
        failures=2,
        confidence=0.9,
        uncertainty=0.1,
    )

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    low_result = (
        integrator.assess(low)
    )

    high_result = (
        integrator.assess(high)
    )

    assert (
        high_result.confidence
        > low_result.confidence
    )


def test_confidence_is_bounded():
    evidence = make_evidence(
        attempts=1000,
        successes=1000,
        failures=0,
        confidence=1.0,
        uncertainty=0.0,
    )

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = integrator.assess(
        evidence
    )

    assert (
        0.0
        <= result.confidence
        <= 1.0
    )

    assert (
        0.0
        <= result.uncertainty
        <= 1.0
    )


def test_uncertainty_is_complement_of_confidence():
    evidence = make_evidence()

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = integrator.assess(
        evidence
    )

    assert (
        result.uncertainty
        == round(
            1.0 - result.confidence,
            6,
        )
    )


def test_assessment_properties():
    evidence = make_evidence(
        attempts=20,
        successes=18,
        failures=2,
        confidence=0.9,
        uncertainty=0.1,
    )

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = integrator.assess(
        evidence
    )

    assert result.high_confidence
    assert not result.low_confidence
    assert not result.high_uncertainty
    assert result.reliable


def test_summary_contains_all_observed_capabilities():
    evidence = make_multi_evidence()

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    summary = (
        integrator.assess_evidence(
            evidence
        )
    )

    assert (
        summary.capability_count
        == 2
    )

    assert (
        summary.get("planning")
        is not None
    )

    assert (
        summary.get("exploration")
        is not None
    )


def test_highest_confidence_capability_is_identified():
    evidence = make_multi_evidence()

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = (
        integrator.assess_evidence(
            evidence
        )
    )

    assert (
        result.highest_confidence_capability
        == "planning"
    )


def test_lowest_confidence_capability_is_identified():
    evidence = make_multi_evidence()

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = (
        integrator.assess_evidence(
            evidence
        )
    )

    assert (
        result.lowest_confidence_capability
        == "exploration"
    )


def test_empty_evidence_returns_empty_summary():
    model = (
        SelfModelBehavioralEvidenceModel()
    )

    evidence = model.analyze({})

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = (
        integrator.assess_evidence(
            evidence
        )
    )

    assert result.capability_count == 0
    assert result.average_confidence == 0.0
    assert result.average_uncertainty == 0.0


def test_assess_capabilities_accepts_sequence():
    evidence = make_multi_evidence()

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = (
        integrator.assess_capabilities(
            evidence.capabilities
        )
    )

    assert (
        result.capability_count
        == 2
    )


def test_confidence_for_returns_float():
    evidence = make_evidence()

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = (
        integrator.confidence_for(
            evidence
        )
    )

    assert isinstance(
        result,
        float,
    )

    assert 0.0 <= result <= 1.0


def test_uncertainty_for_returns_float():
    evidence = make_evidence()

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = (
        integrator.uncertainty_for(
            evidence
        )
    )

    assert isinstance(
        result,
        float,
    )

    assert 0.0 <= result <= 1.0


def test_snapshot_contains_assessments():
    evidence = make_multi_evidence()

    integrator = (
        CapabilityConfidenceIntegrator()
    )

    result = (
        integrator.assess_evidence(
            evidence
        )
    )

    snapshot = result.snapshot()

    assert (
        "assessments"
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


def test_none_evidence_is_rejected():
    integrator = (
        CapabilityConfidenceIntegrator()
    )

    with pytest.raises(
        ValueError,
        match="evidence cannot be None",
    ):
        integrator.assess(
            None
        )


def test_wrong_evidence_type_is_rejected():
    integrator = (
        CapabilityConfidenceIntegrator()
    )

    with pytest.raises(
        ValueError,
        match="CapabilityBehavioralEvidence",
    ):
        integrator.assess(
            {}
        )


def test_none_summary_evidence_is_rejected():
    integrator = (
        CapabilityConfidenceIntegrator()
    )

    with pytest.raises(
        ValueError,
        match="evidence cannot be None",
    ):
        integrator.assess_evidence(
            None
        )


def test_none_capabilities_are_rejected():
    integrator = (
        CapabilityConfidenceIntegrator()
    )

    with pytest.raises(
        ValueError,
        match="capabilities cannot be None",
    ):
        integrator.assess_capabilities(
            None
        )


def test_weights_must_sum_to_one():
    with pytest.raises(
        ValueError,
        match="weights must sum to 1",
    ):
        CapabilityConfidenceIntegrator(
            source_confidence_weight=0.5,
            success_rate_weight=0.5,
            evidence_strength_weight=0.5,
        )


def test_weights_must_be_bounded():
    with pytest.raises(ValueError):
        CapabilityConfidenceIntegrator(
            source_confidence_weight=-0.1,
            success_rate_weight=0.6,
            evidence_strength_weight=0.5,
        )

    with pytest.raises(ValueError):
        CapabilityConfidenceIntegrator(
            source_confidence_weight=1.1,
            success_rate_weight=0.0,
            evidence_strength_weight=-0.1,
        )
