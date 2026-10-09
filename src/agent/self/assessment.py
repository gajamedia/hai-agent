from dataclasses import dataclass

from agent.self.capability import Capability
from agent.learning.strategy import Strategy


@dataclass
class SelfAssessment:
    """
    Assessment of the agent's confidence,
    uncertainty, and risk for a capability or strategy.
    """

    confidence: float
    uncertainty: float
    risk: float


class SelfAssessmentEngine:

    def assess_capability(
        self,
        capability: Capability,
    ) -> SelfAssessment:

        confidence = capability.confidence

        uncertainty = round(
            1.0 - confidence,
            6,
        )

        risk = self._calculate_risk(
            confidence=confidence,
            failure_count=(
                capability.failure_count
            ),
            attempts=capability.attempts,
        )

        return SelfAssessment(
            confidence=confidence,
            uncertainty=uncertainty,
            risk=risk,
        )

    def assess_strategy(
        self,
        strategy: Strategy,
    ) -> SelfAssessment:

        confidence = self._strategy_confidence(
            strategy
        )

        uncertainty = round(
            1.0 - confidence,
            6,
        )

        risk = self._calculate_risk(
            confidence=confidence,
            failure_count=(
                strategy.failure_count
            ),
            attempts=strategy.attempts,
        )

        return SelfAssessment(
            confidence=confidence,
            uncertainty=uncertainty,
            risk=risk,
        )

    def _strategy_confidence(
        self,
        strategy: Strategy,
    ) -> float:

        if strategy.attempts == 0:
            return 0.0

        return strategy.success_rate

    def _calculate_risk(
        self,
        confidence: float,
        failure_count: int,
        attempts: int,
    ) -> float:

        if attempts == 0:
            return 1.0

        failure_rate = (
            failure_count / attempts
        )

        risk = (
            0.5 * (1.0 - confidence)
            + 0.5 * failure_rate
        )

        return round(
            min(
                1.0,
                max(
                    0.0,
                    risk,
                ),
            ),
            6,
        )
