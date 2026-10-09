import pytest

from agent.cognition.risk import RiskAssessment


def test_risk_assessment_with_no_evidence():

    assessment = (
        RiskAssessment.from_outcomes(
            success_count=0,
            failure_count=0,
        )
    )

    assert assessment.evidence == 0

    assert (
        assessment.success_probability
        == pytest.approx(0.5)
    )

    assert (
        assessment.failure_probability
        == pytest.approx(0.5)
    )

    assert (
        assessment.uncertainty
        == pytest.approx(1.0)
    )

    assert (
        assessment.risk_score
        == pytest.approx(0.65)
    )

    assert assessment.is_high_risk


def test_successful_history_produces_lower_risk():

    assessment = (
        RiskAssessment.from_outcomes(
            success_count=9,
            failure_count=1,
        )
    )

    assert assessment.evidence == 10

    assert (
        assessment.success_probability
        > assessment.failure_probability
    )

    assert (
        assessment.risk_score
        < 0.33
    )

    assert assessment.is_low_risk


def test_failure_heavy_history_produces_higher_risk():

    assessment = (
        RiskAssessment.from_outcomes(
            success_count=2,
            failure_count=8,
        )
    )

    assert assessment.evidence == 10

    assert (
        assessment.failure_probability
        > assessment.success_probability
    )

    assert (
        assessment.risk_score
        > 0.5
    )

    assert assessment.is_medium_risk


def test_more_evidence_reduces_uncertainty():

    low_evidence = (
        RiskAssessment.from_outcomes(
            success_count=1,
            failure_count=0,
        )
    )

    high_evidence = (
        RiskAssessment.from_outcomes(
            success_count=100,
            failure_count=0,
        )
    )

    assert (
        high_evidence.uncertainty
        < low_evidence.uncertainty
    )


def test_negative_success_count_is_rejected():

    with pytest.raises(
        ValueError,
        match="success_count cannot be negative",
    ):
        RiskAssessment.from_outcomes(
            success_count=-1,
            failure_count=0,
        )


def test_negative_failure_count_is_rejected():

    with pytest.raises(
        ValueError,
        match="failure_count cannot be negative",
    ):
        RiskAssessment.from_outcomes(
            success_count=0,
            failure_count=-1,
        )


def test_risk_values_are_normalized():

    assessment = (
        RiskAssessment.from_outcomes(
            success_count=25,
            failure_count=75,
        )
    )

    assert 0.0 <= (
        assessment.success_probability
    ) <= 1.0

    assert 0.0 <= (
        assessment.failure_probability
    ) <= 1.0

    assert 0.0 <= (
        assessment.uncertainty
    ) <= 1.0

    assert 0.0 <= (
        assessment.risk_score
    ) <= 1.0
