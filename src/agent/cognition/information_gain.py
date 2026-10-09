from dataclasses import dataclass

from agent.cognition.knowledge_uncertainty import (
    KnowledgeUncertaintyEstimator,
)


@dataclass(frozen=True)
class InformationGainAssessment:
    """
    Describes the estimated value of gaining additional
    knowledge about a candidate.

    Values are normalized to [0.0, 1.0].
    """

    uncertainty: float
    expected_information_gain: float
    evidence: int


class InformationGainEstimator:
    """
    Estimates information gain from knowledge uncertainty.

    Information gain is currently approximated by uncertainty.

    The important distinction is:

        supportive evidence
            -> reduces uncertainty

        contradictory evidence
            -> increases uncertainty

    This allows contradictory knowledge to remain uncertain
    instead of being incorrectly treated as additional evidence.
    """

    def __init__(
        self,
        knowledge_uncertainty_estimator=None,
    ):

        self.knowledge_uncertainty_estimator = (
            knowledge_uncertainty_estimator
            if knowledge_uncertainty_estimator
            is not None
            else KnowledgeUncertaintyEstimator()
        )

    def estimate_from_evidence(
        self,
        evidence: int,
    ) -> InformationGainAssessment:

        if evidence < 0:
            raise ValueError(
                "evidence cannot be negative."
            )

        knowledge = (
            self.knowledge_uncertainty_estimator
            .estimate(
                evidence_count=evidence,
                contradiction_count=0,
            )
        )

        return InformationGainAssessment(
            uncertainty=knowledge.uncertainty,
            expected_information_gain=(
                knowledge.uncertainty
            ),
            evidence=evidence,
        )

    def estimate_from_outcomes(
        self,
        success_count: int,
        failure_count: int,
    ) -> InformationGainAssessment:

        if success_count < 0:
            raise ValueError(
                "success_count cannot be negative."
            )

        if failure_count < 0:
            raise ValueError(
                "failure_count cannot be negative."
            )

        # Outcomes are still treated as observations.
        evidence = (
            success_count
            + failure_count
        )

        return self.estimate_from_evidence(
            evidence
        )

    def estimate_from_procedural_memory(
        self,
        procedural_memory,
    ) -> InformationGainAssessment:

        if procedural_memory is None:
            raise ValueError(
                "procedural_memory cannot be None."
            )

        success_count = getattr(
            procedural_memory,
            "success_count",
            0,
        )

        failure_count = getattr(
            procedural_memory,
            "failure_count",
            0,
        )

        return self.estimate_from_outcomes(
            success_count=success_count,
            failure_count=failure_count,
        )

    def estimate_from_causal_memory(
        self,
        causal_memory,
    ) -> InformationGainAssessment:

        if causal_memory is None:
            raise ValueError(
                "causal_memory cannot be None."
            )

        evidence_count = getattr(
            causal_memory,
            "evidence_count",
            0,
        )

        contradiction_count = getattr(
            causal_memory,
            "contradiction_count",
            0,
        )

        if evidence_count < 0:
            raise ValueError(
                "evidence_count cannot be negative."
            )

        if contradiction_count < 0:
            raise ValueError(
                "contradiction_count cannot be negative."
            )

        knowledge = (
            self.knowledge_uncertainty_estimator
            .estimate(
                evidence_count=evidence_count,
                contradiction_count=(
                    contradiction_count
                ),
            )
        )

        return InformationGainAssessment(
            uncertainty=knowledge.uncertainty,
            expected_information_gain=(
                knowledge.uncertainty
            ),
            evidence=evidence_count,
        )
