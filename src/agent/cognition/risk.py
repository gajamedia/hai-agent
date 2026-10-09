from dataclasses import dataclass


@dataclass(frozen=True)
class RiskAssessment:
    """
    Represents the estimated risk of executing a strategy
    or procedure.

    All probabilities and scores are normalized to [0.0, 1.0].
    """

    success_probability: float
    failure_probability: float
    uncertainty: float
    evidence: int
    risk_score: float

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(1.0, value),
        )

    @classmethod
    def from_outcomes(
        cls,
        success_count: int,
        failure_count: int,
    ) -> "RiskAssessment":

        if success_count < 0:
            raise ValueError(
                "success_count cannot be negative."
            )

        if failure_count < 0:
            raise ValueError(
                "failure_count cannot be negative."
            )

        evidence = (
            success_count
            + failure_count
        )

        # Laplace smoothing prevents zero-evidence and
        # one-sided observations from producing extreme
        # probabilities.
        success_probability = (
            success_count + 1
        ) / (
            evidence + 2
        )

        failure_probability = (
            failure_count + 1
        ) / (
            evidence + 2
        )

        # Uncertainty starts at 1.0 with no evidence and
        # decreases as evidence grows.
        uncertainty = (
            1.0
            / (evidence + 1.0) ** 0.5
        )

        # Risk is currently composed of:
        #
        #   70% failure probability
        #   30% uncertainty
        #
        # This weighting intentionally makes observed
        # failures more important than lack of evidence.
        risk_score = round(
            0.7 * failure_probability
            + 0.3 * uncertainty,
            6
        )

        return cls(
            success_probability=cls._clamp(
                success_probability
            ),
            failure_probability=cls._clamp(
                failure_probability
            ),
            uncertainty=cls._clamp(
                uncertainty
            ),
            evidence=evidence,
            risk_score=cls._clamp(
                risk_score
            ),
        )

    @property
    def is_low_risk(self) -> bool:
        return self.risk_score < 0.33

    @property
    def is_medium_risk(self) -> bool:
        return (
            0.33 <= self.risk_score < 0.65
        )

    @property
    def is_high_risk(self) -> bool:
        return self.risk_score >= 0.65
