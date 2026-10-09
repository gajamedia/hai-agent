from typing import List, Optional

from agent.learning.strategy import Strategy


class StrategySelector:

    def select(
        self,
        strategies: List[Strategy],
        goal: Optional[str] = None,
    ) -> Optional[Strategy]:

        candidates = strategies

        if goal is not None:

            candidates = [
                strategy
                for strategy in strategies
                if strategy.goal == goal
            ]

        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=self._score,
            reverse=True,
        )

        return ranked[0]

    def _score(
        self,
        strategy: Strategy,
    ) -> float:

        if strategy.attempts == 0:
            return 0.0

        success_score = (
            strategy.success_rate
        )

        action_penalty = (
            strategy.average_actions
            * 0.01
        )

        reflection_penalty = (
            strategy.reflection_penalty
        )

        return round(
            success_score
            - action_penalty
            - reflection_penalty,
            6,
        )
