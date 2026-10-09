"""
Strategy evolution candidate selection subsystem.

Selects the best validated strategy candidate using the
existing RiskAwareStrategySelector.

The selector does not register or mutate candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.learning.strategy import Strategy
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)


@dataclass(frozen=True)
class StrategySelectionResult:
    """
    Result of evolution candidate selection.
    """

    strategy: Optional[Strategy]
    score: Optional[float]
    selected: bool
    reason: str

    @property
    def strategy_name(self) -> Optional[str]:
        if self.strategy is None:
            return None

        return self.strategy.name


class StrategyEvolutionSelector:
    """
    Select the strongest validated evolution candidate.

    Primary ranking:
        RiskAwareStrategySelector.score()

    Tie-breakers:
        1. fewer actions / steps
        2. original candidate order

    Candidates are never modified or persisted here.
    """

    def __init__(
        self,
        risk_selector: Optional[
            RiskAwareStrategySelector
        ] = None,
    ) -> None:

        self.risk_selector = (
            risk_selector
            if risk_selector is not None
            else RiskAwareStrategySelector()
        )

    # =====================================================
    # Ranking
    # =====================================================

    def rank(
        self,
        candidates: List[Strategy],
    ) -> List[
        Tuple[Strategy, float]
    ]:
        """
        Rank candidates by risk-aware score.

        Ties are resolved by fewer steps.
        Python's stable sorting preserves original order
        when both score and step count are equal.
        """

        if candidates is None:
            raise ValueError(
                "candidates cannot be None."
            )

        if not candidates:
            return []

        for candidate in candidates:
            if candidate is None:
                raise ValueError(
                    "candidate cannot be None."
                )

        ranked = self.risk_selector.rank(
            candidates
        )

        ranked.sort(
            key=lambda item: (
                -item[1],
                len(item[0].steps),
            )
        )

        return ranked

    # =====================================================
    # Selection
    # =====================================================

    def select(
        self,
        candidates: List[Strategy],
    ) -> StrategySelectionResult:
        """
        Select the strongest candidate.
        """

        if candidates is None:
            raise ValueError(
                "candidates cannot be None."
            )

        if not candidates:
            return StrategySelectionResult(
                strategy=None,
                score=None,
                selected=False,
                reason=(
                    "no candidates available"
                ),
            )

        ranked = self.rank(
            candidates
        )

        strategy, score = ranked[0]

        return StrategySelectionResult(
            strategy=strategy,
            score=score,
            selected=True,
            reason=(
                "best candidate selected "
                "by risk-aware score"
            ),
        )

@dataclass(frozen=True)
class StrategyEvolutionSelectionResult:
    """
    Combined validation + selection result.
    """

    valid_candidates: List[Strategy]
    rejected_candidates: List[Strategy]
    selection: StrategySelectionResult


class StrategyEvolutionPipeline:
    """
    Validate and select evolution candidates.

    Registration is intentionally outside this pipeline.
    """

    def __init__(
        self,
        validator,
        selector: Optional[
            StrategyEvolutionSelector
        ] = None,
    ) -> None:

        self.validator = validator

        self.selector = (
            selector
            if selector is not None
            else StrategyEvolutionSelector()
        )

    def select(
        self,
        source_strategy: Optional[Strategy],
        candidates: List[Strategy],
        world_state,
        goal_state,
    ) -> StrategyEvolutionSelectionResult:

        if candidates is None:
            raise ValueError(
                "candidates cannot be None."
            )

        validation_results = (
            self.validator.validate_all(
                source_strategy=source_strategy,
                candidates=candidates,
                world_state=world_state,
                goal_state=goal_state,
            )
        )

        valid_candidates = [
            result.strategy
            for result in validation_results
            if result.valid
        ]

        rejected_candidates = [
            result.strategy
            for result in validation_results
            if not result.valid
        ]

        selection = self.selector.select(
            valid_candidates
        )

        return StrategyEvolutionSelectionResult(
            valid_candidates=valid_candidates,
            rejected_candidates=rejected_candidates,
            selection=selection,
        )
