from agent.cognition.risk import RiskAssessment


class ProceduralRiskAssessor:
    """
    Converts procedural memory outcome history
    into a RiskAssessment.
    """

    def assess(
        self,
        procedural_memory,
    ) -> RiskAssessment:

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

        return RiskAssessment.from_outcomes(
            success_count=success_count,
            failure_count=failure_count,
        )
