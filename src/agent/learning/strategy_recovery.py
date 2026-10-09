"""
Strategy recovery subsystem.

Responsible for recovering paused strategies when
sufficient positive evidence appears.

Retired strategies are terminal and cannot be recovered
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from agent.learning.strategy import Strategy


@dataclass(frozen=True)
class StrategyRecoveryResult:
    strategy: Strategy
    recovered: bool
    reason: str

    @property
    def strategy_name(self) -> str:
        return self.strategy.name


class StrategyRecoveryManager:
    """
    Recover paused strategies when sufficient positive
    evidence indicates that the strategy is useful again.

    Recovery rules:

    - strategy must exist
    - strategy must currently be PAUSED
    - strategy must not be RETIRED
    - minimum number of attempts must be reached
    - success rate must meet the recovery threshold
    - average progress must be positive

    RETIRED strategies are terminal and cannot be
    automatically recovered.
    """

    def __init__(
        self,
        min_attempts: int = 3,
        min_success_rate: float = 0.50,
        min_average_progress: float = 0.0,
    ) -> None:

        if min_attempts < 1:
            raise ValueError(
                "min_attempts must be >= 1."
            )

        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError(
                "min_success_rate must be between 0.0 and 1.0."
            )

        self.min_attempts = min_attempts
        self.min_success_rate = min_success_rate
        self.min_average_progress = min_average_progress

    def can_recover(
        self,
        strategy: Optional[Strategy],
    ) -> bool:
        """
        Determine whether a paused strategy currently
        satisfies the recovery requirements.
        """

        if strategy is None:
            return False

        if strategy.is_retired:
            return False

        if not strategy.is_paused:
            return False

        if strategy.attempts < self.min_attempts:
            return False

        if strategy.success_rate < self.min_success_rate:
            return False

        if strategy.average_progress <= self.min_average_progress:
            return False

        return True

    def recover(
        self,
        strategy: Strategy,
    ) -> StrategyRecoveryResult:
        """
        Attempt to recover a single strategy.
        """

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        # -------------------------------------------------
        # Already retired
        # -------------------------------------------------

        if strategy.is_retired:
            return StrategyRecoveryResult(
                strategy=strategy,
                recovered=False,
                reason=(
                    "retired strategies are terminal"
                ),
            )

        # -------------------------------------------------
        # Already active
        # -------------------------------------------------

        if strategy.is_active:
            return StrategyRecoveryResult(
                strategy=strategy,
                recovered=False,
                reason=(
                    "strategy is already active"
                ),
            )

        # -------------------------------------------------
        # Only PAUSED strategies can recover
        # -------------------------------------------------

        if not strategy.is_paused:
            return StrategyRecoveryResult(
                strategy=strategy,
                recovered=False,
                reason=(
                    "strategy is not paused"
                ),
            )

        # -------------------------------------------------
        # Minimum evidence
        # -------------------------------------------------

        if strategy.attempts < self.min_attempts:
            return StrategyRecoveryResult(
                strategy=strategy,
                recovered=False,
                reason=(
                    "insufficient attempts"
                ),
            )

        # -------------------------------------------------
        # Success rate
        # -------------------------------------------------

        if (
            strategy.success_rate
            < self.min_success_rate
        ):
            return StrategyRecoveryResult(
                strategy=strategy,
                recovered=False,
                reason=(
                    "success rate below "
                    "recovery threshold"
                ),
            )

        # -------------------------------------------------
        # Average progress
        # -------------------------------------------------

        if (
            strategy.average_progress
            <= self.min_average_progress
        ):
            return StrategyRecoveryResult(
                strategy=strategy,
                recovered=False,
                reason=(
                    "average progress below "
                    "recovery threshold"
                ),
            )

        # -------------------------------------------------
        # Recovery
        # -------------------------------------------------

        strategy.activate()

        return StrategyRecoveryResult(
            strategy=strategy,
            recovered=True,
            reason=(
                "strategy recovered from "
                "positive evidence"
            ),
        )

    def recover_all(
        self,
        strategies: List[Strategy],
    ) -> List[StrategyRecoveryResult]:
        """
        Attempt recovery for all supplied strategies.
        """

        if strategies is None:
            raise ValueError(
                "strategies cannot be None."
            )

        return [
            self.recover(strategy)
            for strategy in strategies
        ]
