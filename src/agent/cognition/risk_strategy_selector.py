from __future__ import annotations

from agent.cognition.risk import RiskAssessment
from agent.learning.strategy import Strategy


class RiskAwareStrategySelector:
    """
    Selects strategies using:

    - success probability
    - risk
    - uncertainty
    - average action cost
    - reflection penalty
    - goal progress
    """

    def __init__(
        self,
        risk_weight: float = 0.5,
        uncertainty_weight: float = 0.2,
        action_cost_weight: float = 0.01,
        reflection_weight: float = 0.2,
        progress_weight: float = 0.3,
    ):
        self.risk_weight = risk_weight
        self.uncertainty_weight = (
            uncertainty_weight
        )
        self.action_cost_weight = (
            action_cost_weight
        )
        self.reflection_weight = (
            reflection_weight
        )
        self.progress_weight = progress_weight

    # =================================================
    # Risk assessment
    # =================================================

    def assess(
        self,
        strategy: Strategy,
    ) -> RiskAssessment:

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        return RiskAssessment.from_outcomes(
            success_count=(
                strategy.success_count
            ),
            failure_count=(
                strategy.failure_count
            ),
        )

    # =================================================
    # Strategy score
    # =================================================

    def score(
        self,
        strategy: Strategy,
    ) -> float:

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        assessment = self.assess(
            strategy
        )

        # -------------------------------------------------
        # Goal progress
        # -------------------------------------------------

        progress = max(
            -1.0,
            min(
                1.0,
                strategy.average_progress,
            ),
        )

        # -------------------------------------------------
        # Final score
        # -------------------------------------------------

        score = (
            assessment.success_probability
            - (
                self.risk_weight
                * assessment.risk_score
            )
            - (
                self.uncertainty_weight
                * assessment.uncertainty
            )
            - (
                self.action_cost_weight
                * strategy.average_actions
            )
            - (
                self.reflection_weight
                * strategy.reflection_penalty
            )
            + (
                self.progress_weight
                * progress
            )
        )

        return round(
            score,
            6,
        )

    # =================================================
    # Ranking
    # =================================================

    def rank(
        self,
        strategies: list[Strategy],
    ) -> list[tuple[Strategy, float]]:
        """
        Return strategies ranked by score.

        The existing API returns:

            [(strategy, score), ...]

        rather than:

            [strategy, ...]
        """

        if strategies is None:
            raise ValueError(
                "strategies cannot be None."
            )

        scored = [
            (
                strategy,
                self.score(strategy),
            )
            for strategy in strategies
        ]

        return sorted(
            scored,
            key=lambda item: item[1],
            reverse=True,
        )

    # =================================================
    # Selection
    # =================================================

    def select(
        self,
        strategies: list[Strategy],
    ) -> Strategy | None:

        if strategies is None:
            raise ValueError(
                "strategies cannot be None."
            )

        if not strategies:
            return None

        ranked = self.rank(
            strategies
        )

        return ranked[0][0]
