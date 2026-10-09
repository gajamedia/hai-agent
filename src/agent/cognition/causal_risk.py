from agent.cognition.risk import RiskAssessment


class CausalRiskAssessor:
    """
    Converts causal memory evidence into a RiskAssessment.

    Causal knowledge is considered less reliable when it has
    more contradictions relative to supporting evidence.
    """

    def assess(
        self,
        causal_memory,
    ) -> RiskAssessment:

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

        return RiskAssessment.from_outcomes(
            success_count=evidence_count,
            failure_count=contradiction_count,
        )
