from dataclasses import dataclass

from agent.self.uncertainty import Uncertainty


@dataclass(frozen=True)
class KnowledgeUncertainty:
    """
    Represents uncertainty about learned knowledge.

    Unlike generic decision uncertainty, this model considers:
        - amount of evidence
        - contradictory evidence

    All normalized values are in [0.0, 1.0].
    """

    evidence_count: int
    contradiction_count: int
    uncertainty: float
    contradiction_rate: float

    @property
    def confidence(self) -> float:
        return round(
            1.0 - self.uncertainty,
            6,
        )

    @property
    def certainty(self) -> float:
        return round(
            1.0 - self.uncertainty,
            6,
        )

    def as_uncertainty(self) -> Uncertainty:
        return Uncertainty(
            value=self.uncertainty
        )


class KnowledgeUncertaintyEstimator:
    """
    Estimates uncertainty from evidence and contradictions.

    Base uncertainty decreases with evidence:

        base_uncertainty = 1 / sqrt(evidence + 1)

    Contradictions add uncertainty:

        contradiction_rate =
            contradictions / (evidence + contradictions)

    Final uncertainty:

        base_uncertainty
        + 0.5 * contradiction_rate

    Result is clamped to [0.0, 1.0].
    """

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(1.0, value),
        )

    def estimate(
        self,
        evidence_count: int,
        contradiction_count: int,
    ) -> KnowledgeUncertainty:

        if evidence_count < 0:
            raise ValueError(
                "evidence_count cannot be negative."
            )

        if contradiction_count < 0:
            raise ValueError(
                "contradiction_count cannot be negative."
            )

        from math import sqrt

        base_uncertainty = (
            1.0
            / sqrt(evidence_count + 1)
        )

        total_observations = (
            evidence_count
            + contradiction_count
        )

        if total_observations == 0:
            contradiction_rate = 0.0
        else:
            contradiction_rate = (
                contradiction_count
                / total_observations
            )

        uncertainty = (
            base_uncertainty
            + (
                0.5
                * contradiction_rate
            )
        )

        uncertainty = self._clamp(
            uncertainty
        )

        return KnowledgeUncertainty(
            evidence_count=evidence_count,
            contradiction_count=(
                contradiction_count
            ),
            uncertainty=round(
                uncertainty,
                6,
            ),
            contradiction_rate=round(
                contradiction_rate,
                6,
            ),
        )

    def estimate_from_confidence(
        self,
        confidence,
        contradiction_count: int = 0,
    ) -> KnowledgeUncertainty:

        if contradiction_count < 0:
            raise ValueError(
                "contradiction_count cannot be negative."
            )

        evidence_count = getattr(
            confidence,
            "evidence_count",
            0,
        )

        return self.estimate(
            evidence_count=evidence_count,
            contradiction_count=(
                contradiction_count
            ),
        )
