from __future__ import annotations

from dataclasses import dataclass

from agent.learning.strategy import Strategy


@dataclass(frozen=True)
class StrategyConsolidationResult:
    """
    Result of consolidating equivalent strategies.
    """

    strategies: list[Strategy]
    merged_count: int


class StrategyConsolidator:
    """
    Consolidates equivalent strategies.

    Two strategies are equivalent when:
        - goal is equal
        - action sequence is equal

    Evidence is merged into the first strategy.
    """

    @staticmethod
    def _key(
        strategy: Strategy,
    ) -> tuple[str, tuple[str, ...]]:

        return (
            strategy.goal.strip().lower(),
            tuple(strategy.steps),
        )

    def equivalent(
        self,
        first: Strategy,
        second: Strategy,
    ) -> bool:

        if first is None or second is None:
            return False

        return (
            self._key(first)
            == self._key(second)
        )

    def merge(
        self,
        target: Strategy,
        source: Strategy,
    ) -> Strategy:

        if target is None:
            raise ValueError(
                "target strategy cannot be None."
            )

        if source is None:
            raise ValueError(
                "source strategy cannot be None."
            )

        if target is source:
            return target

        if not self.equivalent(
            target,
            source,
        ):
            raise ValueError(
                "Strategies are not equivalent."
            )

        target.success_count += (
            source.success_count
        )

        target.failure_count += (
            source.failure_count
        )

        target.total_actions += (
            source.total_actions
        )

        target.total_progress += (
            source.total_progress
        )

        target.progress_count += (
            source.progress_count
        )

        target.positive_progress_count += (
            source.positive_progress_count
        )

        target.negative_progress_count += (
            source.negative_progress_count
        )

        target.reflection_penalty = min(
            1.0,
            round(
                max(
                    target.reflection_penalty,
                    source.reflection_penalty,
                ),
                6,
            ),
        )

        for experience_id in (
            source.source_experiences
        ):
            target._record_experience(
                experience_id
            )

        source.retire()

        return target

    def consolidate(
        self,
        strategies: list[Strategy],
    ) -> StrategyConsolidationResult:

        if strategies is None:
            raise ValueError(
                "strategies cannot be None."
            )

        representatives: dict[
            tuple[str, tuple[str, ...]],
            Strategy,
        ] = {}

        merged_count = 0

        for strategy in strategies:

            key = self._key(
                strategy
            )

            if key not in representatives:

                representatives[key] = (
                    strategy
                )

                continue

            representative = (
                representatives[key]
            )

            self.merge(
                target=representative,
                source=strategy,
            )

            merged_count += 1

        return StrategyConsolidationResult(
            strategies=list(
                representatives.values()
            ),
            merged_count=merged_count,
        )


class StrategyLifecycleManager:
    """
    Decides whether a strategy should remain active,
    be paused, or be retired.

    Lifecycle is intentionally conservative.
    """

    def __init__(
        self,
        minimum_attempts: int = 3,
        pause_success_rate: float = 0.4,
        pause_progress: float = 0.0,
        retire_success_rate: float = 0.2,
        retire_progress: float = -0.2,
    ):
        self.minimum_attempts = (
            minimum_attempts
        )

        self.pause_success_rate = (
            pause_success_rate
        )

        self.pause_progress = (
            pause_progress
        )

        self.retire_success_rate = (
            retire_success_rate
        )

        self.retire_progress = (
            retire_progress
        )

    def evaluate(
        self,
        strategy: Strategy,
    ) -> str:

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        if strategy.is_retired:
            return strategy.status

        if (
            strategy.attempts
            < self.minimum_attempts
        ):
            return strategy.status

        success_rate = (
            strategy.success_rate
        )

        progress = (
            strategy.average_progress
        )

        # -------------------------------------------------
        # Retire
        # -------------------------------------------------

        if (
            success_rate
            <= self.retire_success_rate
            and progress
            <= self.retire_progress
        ):
            strategy.retire()

            return strategy.status

        # -------------------------------------------------
        # Pause
        # -------------------------------------------------

        if (
            success_rate
            <= self.pause_success_rate
            or progress
            <= self.pause_progress
        ):
            strategy.pause()

            return strategy.status

        # -------------------------------------------------
        # Keep active
        # -------------------------------------------------

        strategy.activate()

        return strategy.status

    def evaluate_all(
        self,
        strategies: list[Strategy],
    ) -> list[Strategy]:

        if strategies is None:
            raise ValueError(
                "strategies cannot be None."
            )

        for strategy in strategies:
            self.evaluate(
                strategy
            )

        return strategies
