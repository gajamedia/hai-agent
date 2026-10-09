"""
Unified strategy evolution subsystem.

Generates strategy candidates from failed experiences by
using two complementary mutation mechanisms:

1. Remove a failed action.
2. Replace a failed action with an alternative action.

This module only generates candidates.

Candidate validation and selection remain responsibilities
of their dedicated components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from agent.core.action import Action
from agent.core.experience import Experience
from agent.learning.strategy import Strategy
from agent.learning.strategy_action_replacement import (
    StrategyActionReplacement,
)


@dataclass(frozen=True)
class StrategyEvolutionResult:
    """
    Result of a strategy evolution attempt.
    """

    source_strategy: Strategy
    candidate: Optional[Strategy]
    evolved: bool
    reason: str

    @property
    def candidate_strategy(
        self,
    ) -> Optional[Strategy]:
        return self.candidate


class StrategyEvolution:
    """
    Unified strategy evolution engine.

    Candidate generation order:

        1. Remove failed action.
        2. Replace failed action.

    Removal candidates are preserved as the first generation
    mechanism for backward compatibility.

    Replacement candidates are generated only when
    registered actions are supplied.
    """

    def __init__(
        self,
        actions: Optional[
            Iterable[Action]
        ] = None,
        max_candidates: int = 3,
    ) -> None:

        if max_candidates < 1:
            raise ValueError(
                "max_candidates must be >= 1."
            )

        self.max_candidates = (
            max_candidates
        )

        self.actions = {}

        if actions is not None:

            for action in actions:

                if action is None:
                    continue

                self.actions[
                    action.name
                ] = action

        self.action_replacement = (
            StrategyActionReplacement(
                actions=self.actions.values(),
                max_candidates=max_candidates,
            )
        )

    # =====================================================
    # Failed action extraction
    # =====================================================

    def _failed_actions(
        self,
        experience: Experience,
    ) -> List[str]:
        """
        Return unique failed action names in experience.
        """

        failed_actions: List[str] = []

        for episode in experience.episodes:

            action = getattr(
                episode,
                "action",
                None,
            )

            if action is None:
                continue

            result = (
                getattr(
                    episode,
                    "result",
                    None,
                )
                or {}
            )

            if not result.get(
                "success",
                False,
            ):

                if action not in failed_actions:
                    failed_actions.append(
                        action
                    )

        return failed_actions

    # =====================================================
    # Removal
    # =====================================================

    def _remove_action(
        self,
        strategy: Strategy,
        action: str,
    ) -> Optional[Strategy]:
        """
        Create a candidate by removing a failed action.
        """

        new_steps = [
            step
            for step in strategy.steps
            if step != action
        ]

        if new_steps == strategy.steps:
            return None

        if not new_steps:
            return None

        return Strategy(
            name=(
                f"{strategy.name}_"
                f"without_{action}"
            ),
            goal=strategy.goal,
            steps=new_steps,
        )

    def evolve_removal_candidates(
        self,
        strategy: Strategy,
        experience: Experience,
    ) -> List[Strategy]:
        """
        Generate candidates by removing failed actions.
        """

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        if experience is None:
            raise ValueError(
                "experience cannot be None."
            )

        if not experience.failed:
            return []

        candidates: List[Strategy] = []

        failed_actions = (
            self._failed_actions(
                experience
            )
        )

        for action in failed_actions:

            candidate = (
                self._remove_action(
                    strategy=strategy,
                    action=action,
                )
            )

            if candidate is None:
                continue

            candidates.append(
                candidate
            )

            if (
                len(candidates)
                >= self.max_candidates
            ):
                break

        return candidates

    # =====================================================
    # Unified candidate generation
    # =====================================================

    def evolve_candidates(
        self,
        strategy: Strategy,
        experience: Experience,
    ) -> List[Strategy]:
        """
        Generate all evolution candidates.

        Order:

            removal candidates
            replacement candidates

        Duplicate candidates are removed while preserving
        generation order.
        """

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        if experience is None:
            raise ValueError(
                "experience cannot be None."
            )

        if not experience.failed:
            return []

        candidates: List[Strategy] = []

        # -------------------------------------------------
        # 1. Removal candidates
        # -------------------------------------------------

        removal_candidates = (
            self.evolve_removal_candidates(
                strategy=strategy,
                experience=experience,
            )
        )

        for candidate in removal_candidates:

            if candidate not in candidates:
                candidates.append(
                    candidate
                )

        # -------------------------------------------------
        # 2. Replacement candidates
        # -------------------------------------------------

        remaining_slots = (
            self.max_candidates
            - len(candidates)
        )

        if remaining_slots > 0:

            replacement_candidates = (
                self.action_replacement
                .evolve_candidates(
                    strategy=strategy,
                    experience=experience,
                )
            )

            for candidate in (
                replacement_candidates
            ):

                if candidate in candidates:
                    continue

                candidates.append(
                    candidate
                )

                if (
                    len(candidates)
                    >= self.max_candidates
                ):
                    break

        return candidates

    # =====================================================
    # Public evolution API
    # =====================================================

    def evolve(
        self,
        strategy: Strategy,
        experience: Experience,
    ) -> StrategyEvolutionResult:
        """
        Generate and return the first evolution candidate.

        Evolution is only attempted for failed experiences
        that actually contain at least one failed action.

        Candidate generation order:

            1. Remove failed action.
            2. Replace failed action.

        Removal candidates keep priority for backward
        compatibility.
        """

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        if experience is None:
            raise ValueError(
                "experience cannot be None."
            )

        # -------------------------------------------------
        # Successful experience does not require evolution.
        # -------------------------------------------------

        if experience.completed:
            return StrategyEvolutionResult(
                source_strategy=strategy,
                candidate=None,
                evolved=False,
                reason=(
                    "successful experiences do not "
                    "require evolution"
                ),
            )

        # -------------------------------------------------
        # Experience must explicitly be failed.
        # -------------------------------------------------

        if not experience.failed:
            return StrategyEvolutionResult(
                source_strategy=strategy,
                candidate=None,
                evolved=False,
                reason=(
                    "experience is not marked as failed"
                ),
            )

        # -------------------------------------------------
        # A failed experience must actually contain
        # at least one failed action.
        # -------------------------------------------------

        failed_actions = self._failed_actions(
            experience
        )

        if not failed_actions:
            return StrategyEvolutionResult(
                source_strategy=strategy,
                candidate=None,
                evolved=False,
                reason=(
                    "no failed actions found"
                ),
            )

        # -------------------------------------------------
        # Generate unified candidates.
        # -------------------------------------------------

        candidates = self.evolve_candidates(
            strategy=strategy,
            experience=experience,
        )

        if not candidates:
            return StrategyEvolutionResult(
                source_strategy=strategy,
                candidate=None,
                evolved=False,
                reason=(
                    "no valid evolution candidate "
                    "could be generated"
                ),
            )

        # -------------------------------------------------
        # First candidate preserves the existing
        # removal-first behavior.
        # -------------------------------------------------

        first_candidate = candidates[0]

        removal_candidate_names = {
            candidate.name
            for candidate in (
                self.evolve_removal_candidates(
                    strategy=strategy,
                    experience=experience,
                )
            )
        }

        if (
            first_candidate.name
            in removal_candidate_names
        ):
            reason = (
                "candidate generated by removing "
                "a failed action"
            )

        else:
            reason = (
                "candidate generated by replacing "
                "a failed action"
            )

        return StrategyEvolutionResult(
            source_strategy=strategy,
            candidate=first_candidate,
            evolved=True,
            reason=reason,
        )
