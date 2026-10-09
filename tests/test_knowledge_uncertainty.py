import pytest

from agent.cognition.knowledge_uncertainty import (
    KnowledgeUncertainty,
    KnowledgeUncertaintyEstimator,
)
from agent.self.confidence import ConfidenceScore
from agent.self.uncertainty import Uncertainty


def test_no_knowledge_has_maximum_uncertainty():

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    result = estimator.estimate(
        evidence_count=0,
        contradiction_count=0,
    )

    assert isinstance(
        result,
        KnowledgeUncertainty,
    )

    assert result.evidence_count == 0
    assert result.contradiction_count == 0

    assert (
        result.uncertainty
        == pytest.approx(1.0)
    )

    assert result.confidence == pytest.approx(
        0.0
    )


def test_more_evidence_reduces_uncertainty():

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    weak = estimator.estimate(
        evidence_count=1,
        contradiction_count=0,
    )

    strong = estimator.estimate(
        evidence_count=25,
        contradiction_count=0,
    )

    assert (
        strong.uncertainty
        < weak.uncertainty
    )

    assert (
        strong.confidence
        > weak.confidence
    )


def test_contradictions_increase_uncertainty():

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    stable = estimator.estimate(
        evidence_count=10,
        contradiction_count=0,
    )

    unstable = estimator.estimate(
        evidence_count=10,
        contradiction_count=8,
    )

    assert (
        unstable.uncertainty
        > stable.uncertainty
    )

    assert (
        unstable.confidence
        < stable.confidence
    )


def test_contradiction_rate_is_calculated():

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    result = estimator.estimate(
        evidence_count=10,
        contradiction_count=5,
    )

    assert (
        result.contradiction_rate
        == pytest.approx(5 / 15)
    )


def test_uncertainty_can_be_converted_to_generic_uncertainty():

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    result = estimator.estimate(
        evidence_count=5,
        contradiction_count=2,
    )

    uncertainty = (
        result.as_uncertainty()
    )

    assert isinstance(
        uncertainty,
        Uncertainty,
    )

    assert (
        uncertainty.value
        == result.uncertainty
    )

    assert (
        uncertainty.certainty
        == result.confidence
    )


def test_estimator_can_use_confidence_score():

    confidence = ConfidenceScore(
        confidence=0.8,
        evidence_count=10,
    )

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    result = (
        estimator.estimate_from_confidence(
            confidence=confidence
        )
    )

    assert result.evidence_count == 10

    assert (
        result.uncertainty
        > 0.0
    )


def test_confidence_score_with_contradictions():

    confidence = ConfidenceScore(
        confidence=0.8,
        evidence_count=10,
    )

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    stable = (
        estimator.estimate_from_confidence(
            confidence=confidence,
            contradiction_count=0,
        )
    )

    unstable = (
        estimator.estimate_from_confidence(
            confidence=confidence,
            contradiction_count=10,
        )
    )

    assert (
        unstable.uncertainty
        > stable.uncertainty
    )


def test_negative_evidence_is_rejected():

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    with pytest.raises(
        ValueError,
        match="evidence_count cannot be negative",
    ):
        estimator.estimate(
            evidence_count=-1,
            contradiction_count=0,
        )


def test_negative_contradictions_are_rejected():

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    with pytest.raises(
        ValueError,
        match="contradiction_count cannot be negative",
    ):
        estimator.estimate(
            evidence_count=1,
            contradiction_count=-1,
        )


def test_values_are_normalized():

    estimator = (
        KnowledgeUncertaintyEstimator()
    )

    result = estimator.estimate(
        evidence_count=100,
        contradiction_count=100,
    )

    assert 0.0 <= (
        result.uncertainty
    ) <= 1.0

    assert 0.0 <= (
        result.confidence
    ) <= 1.0

    assert 0.0 <= (
        result.contradiction_rate
    ) <= 1.0
