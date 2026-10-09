from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.core.experience import Experience
from agent.learning.strategy import Strategy


@dataclass(frozen=True)
class StrategyAttributionResult:
    """
    Result of attributing an experience to a strategy.
    """

    strategy: Optional[Strategy]
    matched: bool
    reason: str

    @property
    def strategy_name(
        self,
    ) -> Optional[str]:

        if self.strategy is None:
            return None

        return self.strategy.name


class StrategyAttributor:
    """
    Attributes an experience to the strategy that actually
    matches its goal and executed action sequence.
    """

    @staticmethod
    def _normalize_goal(
        goal: Optional[str],
    ) -> str:

        if goal is None:
            return ""

        return (
            goal
            .strip()
            .lower()
        )

    @staticmethod
    def _experience_actions(
        experience: Experience,
    ) -> list[str]:

        actions: list[str] = []

        for episode in experience.episodes:

            if episode.action is not None:
                actions.append(
                    episode.action
                )

        return actions

    def matches(
        self,
        strategy: Strategy,
        experience: Experience,
    ) -> bool:

        if strategy is None:
            return False

        if experience is None:
            return False

        # -------------------------------------------------
        # Goal must match
        # -------------------------------------------------

        strategy_goal = (
            self._normalize_goal(
                strategy.goal
            )
        )

        experience_goal = (
            self._normalize_goal(
                experience.goal
            )
        )

        if (
            strategy_goal
            != experience_goal
        ):
            return False

        # -------------------------------------------------
        # Action sequence must match
        # -------------------------------------------------

        experience_actions = (
            self._experience_actions(
                experience
            )
        )

        return (
            experience_actions
            == list(strategy.steps)
        )

    def attribute(
        self,
        experience: Experience,
        strategies: list[Strategy],
    ) -> StrategyAttributionResult:

        if experience is None:
            return StrategyAttributionResult(
                strategy=None,
                matched=False,
                reason="experience is None",
            )

        if not strategies:
            return StrategyAttributionResult(
                strategy=None,
                matched=False,
                reason="no strategies available",
            )

        for strategy in strategies:

            if self.matches(
                strategy,
                experience,
            ):
                return StrategyAttributionResult(
                    strategy=strategy,
                    matched=True,
                    reason="goal and action sequence matched",
                )

        return StrategyAttributionResult(
            strategy=None,
            matched=False,
            reason="no strategy matched the experience",
        )
