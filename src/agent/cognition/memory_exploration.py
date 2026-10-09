from dataclasses import dataclass
from typing import Optional

from agent.cognition.exploration import (
    ExplorationDecision,
    ExplorationDecisionMaker,
)
from agent.cognition.knowledge_uncertainty import (
    KnowledgeUncertaintyEstimator,
)


@dataclass(frozen=True)
class MemoryExplorationAssessment:
    """
    Aggregates uncertainty and information gain from
    procedural and/or causal knowledge.
    """

    uncertainty: float
    expected_information_gain: float
    evidence: int
    risk: float
    decision: ExplorationDecision


class MemoryAwareExploration:
    """
    Makes an exploration decision using actual memory.

    Conservative aggregation rule:

        combined uncertainty =
            max(procedural uncertainty,
                causal uncertainty)

    Information gain is derived from the same combined
    uncertainty.

    If only one memory source exists, that source is used.
    If no memory exists, uncertainty is treated as maximum.
    """

    def __init__(
        self,
        decision_maker: Optional[
            ExplorationDecisionMaker
        ] = None,
        uncertainty_estimator: Optional[
            KnowledgeUncertaintyEstimator
        ] = None,
    ):
        self.decision_maker = (
            decision_maker
            if decision_maker is not None
            else ExplorationDecisionMaker()
        )

        self.uncertainty_estimator = (
            uncertainty_estimator
            if uncertainty_estimator is not None
            else KnowledgeUncertaintyEstimator()
        )

    def assess(
        self,
        procedural_memory=None,
        causal_memory=None,
        risk: float = 0.0,
    ) -> MemoryExplorationAssessment:

        procedural = None
        causal = None

        if procedural_memory is not None:
            procedural = (
                self.uncertainty_estimator
                .estimate(
                    evidence_count=(
                        getattr(
                            procedural_memory,
                            "success_count",
                            0,
                        )
                        + getattr(
                            procedural_memory,
                            "failure_count",
                            0,
                        )
                    ),
                    contradiction_count=0,
                )
            )

        if causal_memory is not None:
            causal = (
                self.uncertainty_estimator
                .estimate(
                    evidence_count=(
                        getattr(
                            causal_memory,
                            "evidence_count",
                            0,
                        )
                    ),
                    contradiction_count=(
                        getattr(
                            causal_memory,
                            "contradiction_count",
                            0,
                        )
                    ),
                )
            )

        if (
            procedural is None
            and causal is None
        ):
            uncertainty = 1.0
            evidence = 0

        else:
            uncertainties = []

            if procedural is not None:
                uncertainties.append(
                    procedural.uncertainty
                )

            if causal is not None:
                uncertainties.append(
                    causal.uncertainty
                )

            uncertainty = max(
                uncertainties
            )

            procedural_evidence = (
                (
                    getattr(
                        procedural_memory,
                        "success_count",
                        0,
                    )
                    + getattr(
                        procedural_memory,
                        "failure_count",
                        0,
                    )
                )
                if procedural_memory is not None
                else 0
            )

            causal_evidence = (
                getattr(
                    causal_memory,
                    "evidence_count",
                    0,
                )
                if causal_memory is not None
                else 0
            )

            evidence = (
                procedural_evidence
                + causal_evidence
            )

        information_gain = uncertainty

        decision = (
            self.decision_maker.decide(
                uncertainty=uncertainty,
                expected_information_gain=(
                    information_gain
                ),
                risk=risk,
            )
        )

        return MemoryExplorationAssessment(
            uncertainty=round(
                uncertainty,
                6,
            ),
            expected_information_gain=round(
                information_gain,
                6,
            ),
            evidence=evidence,
            risk=max(
                0.0,
                min(1.0, risk),
            ),
            decision=decision,
        )

    def decide(
        self,
        procedural_memory=None,
        causal_memory=None,
        risk: float = 0.0,
    ) -> ExplorationDecision:

        return self.assess(
            procedural_memory=procedural_memory,
            causal_memory=causal_memory,
            risk=risk,
        ).decision
