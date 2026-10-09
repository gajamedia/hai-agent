from __future__ import annotations

from typing import List, Optional
from agent.learning.strategy import Strategy
from agent.memory.strategy_repository import (
    StrategyRepository,
)


class StrategyMemoryStore:
    """
    In-memory strategy registry backed by an optional
    persistent SQLite repository.
    """

    def __init__(
        self,
        repository: StrategyRepository | None = None,
    ):
        self.repository = repository

        self._strategies: list[
            Strategy
        ] = []

        self.load()

    # =================================================
    # Basic operations
    # =================================================

    def add(
        self,
        strategy: Strategy,
    ) -> Strategy:

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        existing = self.find_matching(
            strategy
        )

        if existing is not None:
            return existing

        self._strategies.append(
            strategy
        )

        self._persist(
            strategy
        )

        return strategy

    def all(
        self,
    ) -> list[Strategy]:

        return list(
            self._strategies
        )

    def count(
        self,
    ) -> int:

        return len(
            self._strategies
        )

    def find_matching(
        self,
        strategy: Strategy,
    ) -> Strategy | None:
        """
        Find the same strategy identity.

        IMPORTANT:

        Identity includes the strategy name.

        Two strategies can intentionally have the same
        goal and action sequence while still being distinct
        strategy records.

        Semantic equivalence is handled separately by
        StrategyConsolidator.
        """

        if strategy is None:
            return None

        for existing in self._strategies:

            if (
                existing.name
                == strategy.name
                and existing.goal
                == strategy.goal
                and existing.steps
                == strategy.steps
            ):
                return existing

        return None

    # =================================================
    # Persistence
    # =================================================

    def save(
        self,
        strategy: Strategy,
    ) -> None:

        if self.repository is None:
            return

        self.repository.save(
            strategy
        )

    def save_all(
        self,
    ) -> None:

        for strategy in self._strategies:
            self.save(
                strategy
            )

    def load(
        self,
    ) -> None:

        if self.repository is None:
            return

        self._strategies = (
            self.repository.load_all()
        )

    def clear(
        self,
    ) -> None:

        self._strategies.clear()

        if self.repository is not None:
            self.repository.clear()

    # =================================================
    # Internal
    # =================================================

    def _persist(
        self,
        strategy: Strategy,
    ) -> None:

        self.save(
            strategy
        )

    def find_exact(self, goal: str, steps: List[str]) -> Optional[Strategy]:
        """
        Find a strategy by exact goal and exact ordered steps.

        This method preserves the legacy StrategyStore API.
        """
        for strategy in self._strategies:
            if strategy.goal != goal:
                continue

            if strategy.steps != steps:
                continue

            return strategy

        return None


    def get_or_create(self, goal: str, steps: List[str]) -> Strategy:
        """
        Return an existing strategy or create a new one.

        Preserves the legacy StrategyStore behavior while using
        StrategyMemoryStore as the canonical implementation.
        """
        existing = self.find_exact(
            goal=goal,
            steps=steps,
        )

        if existing is not None:
            return existing

        strategy = Strategy(
            name=goal.strip().lower().replace(" ", "_"),
            goal=goal,
            steps=list(steps),
        )

        self.add(strategy)

        return strategy


    def for_goal(self, goal: str) -> List[Strategy]:
        """
        Return all strategies belonging to a goal.
        """
        return [
            strategy
            for strategy in self._strategies
            if strategy.goal == goal
        ]

