"""
Strategy action replacement subsystem.

Finds alternative registered actions that can replace
a failed action in an existing strategy.

Replacement is based on action effects rather than
hard-coded action names.

This module only generates candidates.
Candidate validation remains the responsibility of
StrategyCandidateValidator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from agent.core.action import Action
from agent.core.experience import Experience
from agent.learning.strategy import Strategy


@dataclass(frozen=True)
class StrategyActionReplacementResult:
    """
    Result of a replacement attempt.
    """

    source_strategy: Strategy
    failed_action: Optional[str]
    replacement_action: Optional[str]
    candidate: Optional[Strategy]
    replaced: bool
    reason: str


class StrategyActionReplacement:
    """
    Generate strategy candidates by replacing a failed
    action with another registered action that provides
    equivalent effects.

    Example:

        take_key
        effects = {"key": "agent"}

    can potentially be replaced by:

        find_spare_key
        effects = {"key": "agent"}

    The resulting candidate is still validated separately.
    """

    def __init__(
        self,
        actions: Optional[Iterable[Action]] = None,
        max_candidates: int = 3,
    ) -> None:

        if max_candidates < 1:
            raise ValueError(
                "max_candidates must be >= 1."
            )

        self.actions = {}

        if actions is not None:
            for action in actions:
                if action is None:
                    continue

                self.actions[action.name] = action

        self.max_candidates = max_candidates

    # =====================================================
    # Failed action extraction
    # =====================================================

    def _failed_actions(
        self,
        experience: Experience,
    ) -> List[str]:
        """
        Extract unique failed action names from experience.
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

            result = getattr(
                episode,
                "result",
                None,
            ) or {}

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
    # Effect matching
    # =====================================================

    def _effects_match(
        self,
        failed_action: Action,
        alternative: Action,
    ) -> bool:
        """
        Determine whether an alternative action provides
        all effects required from the failed action.

        The alternative may have additional effects.
        """

        if failed_action is None:
            return False

        if alternative is None:
            return False

        if (
            failed_action.name
            == alternative.name
        ):
            return False

        failed_effects = (
            failed_action.effects
        )

        alternative_effects = (
            alternative.effects
        )

        if not failed_effects:
            return False

        for key, expected_value in (
            failed_effects.items()
        ):

            if (
                alternative_effects.get(key)
                != expected_value
            ):
                return False

        return True

    # =====================================================
    # Candidate creation
    # =====================================================

    def _replace_action(
        self,
        strategy: Strategy,
        failed_action: str,
        replacement_action: str,
    ) -> Optional[Strategy]:
        """
        Create a candidate by replacing one strategy step.
        """

        if (
            failed_action
            not in strategy.steps
        ):
            return None

        new_steps = [
            (
                replacement_action
                if step == failed_action
                else step
            )
            for step in strategy.steps
        ]

        if new_steps == strategy.steps:
            return None

        return Strategy(
            name=(
                f"{strategy.name}_"
                f"replace_{failed_action}_"
                f"with_{replacement_action}"
            ),
            goal=strategy.goal,
            steps=new_steps,
        )

    # =====================================================
    # Replacement search
    # =====================================================

    def find_replacements(
        self,
        failed_action: str,
    ) -> List[Action]:
        """
        Find registered actions that can potentially
        replace a failed action.

        This method does not validate runtime preconditions.
        """

        failed = self.actions.get(
            failed_action
        )

        if failed is None:
            return []

        alternatives: List[Action] = []

        for action in self.actions.values():

            if self._effects_match(
                failed,
                action,
            ):
                alternatives.append(
                    action
                )

            if (
                len(alternatives)
                >= self.max_candidates
            ):
                break

        return alternatives

    # =====================================================
    # Public API
    # =====================================================

    def evolve(
        self,
        strategy: Strategy,
        experience: Experience,
    ) -> StrategyActionReplacementResult:

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        if experience is None:
            raise ValueError(
                "experience cannot be None."
            )

        if not experience.failed:
            return StrategyActionReplacementResult(
                source_strategy=strategy,
                failed_action=None,
                replacement_action=None,
                candidate=None,
                replaced=False,
                reason=(
                    "experience is not marked as failed"
                ),
            )

        failed_actions = (
            self._failed_actions(
                experience
            )
        )

        if not failed_actions:
            return StrategyActionReplacementResult(
                source_strategy=strategy,
                failed_action=None,
                replacement_action=None,
                candidate=None,
                replaced=False,
                reason=(
                    "no failed actions found"
                ),
            )

        for failed_action in failed_actions:

            # Failed action must actually belong
            # to the source strategy.
            if (
                failed_action
                not in strategy.steps
            ):
                continue

            replacements = (
                self.find_replacements(
                    failed_action
                )
            )

            if not replacements:
                continue

            replacement = replacements[0]

            candidate = (
                self._replace_action(
                    strategy=strategy,
                    failed_action=failed_action,
                    replacement_action=(
                        replacement.name
                    ),
                )
            )

            if candidate is None:
                continue

            return StrategyActionReplacementResult(
                source_strategy=strategy,
                failed_action=failed_action,
                replacement_action=(
                    replacement.name
                ),
                candidate=candidate,
                replaced=True,
                reason=(
                    "candidate generated by "
                    "action replacement"
                ),
            )

        return StrategyActionReplacementResult(
            source_strategy=strategy,
            failed_action=(
                failed_actions[0]
                if failed_actions
                else None
            ),
            replacement_action=None,
            candidate=None,
            replaced=False,
            reason=(
                "no compatible replacement action found"
            ),
        )

    def evolve_candidates(
        self,
        strategy: Strategy,
        experience: Experience,
    ) -> List[Strategy]:
        """
        Generate all replacement candidates.
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

        for failed_action in failed_actions:

            if (
                failed_action
                not in strategy.steps
            ):
                continue

            replacements = (
                self.find_replacements(
                    failed_action
                )
            )

            for replacement in replacements:

                candidate = (
                    self._replace_action(
                        strategy=strategy,
                        failed_action=failed_action,
                        replacement_action=(
                            replacement.name
                        ),
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
                    return candidates

        return candidates
