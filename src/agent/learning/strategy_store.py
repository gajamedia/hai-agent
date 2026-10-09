# from typing import List, Optional

# from agent.learning.strategy import Strategy


# class StrategyStore:

#     def __init__(self):
#         self.strategies: List[Strategy] = []

#     def add(
#         self,
#         strategy: Strategy,
#     ) -> None:
#         self.strategies.append(
#             strategy
#         )

#     def all(self) -> List[Strategy]:
#         return list(self.strategies)

#     def count(self) -> int:
#         return len(self.strategies)

#     def find_exact(
#         self,
#         goal: str,
#         steps: List[str],
#     ) -> Optional[Strategy]:

#         for strategy in self.strategies:

#             if strategy.goal != goal:
#                 continue

#             if strategy.steps != steps:
#                 continue

#             return strategy

#         return None

#     def get_or_create(
#         self,
#         goal: str,
#         steps: List[str],
#     ) -> Strategy:

#         existing = self.find_exact(
#             goal=goal,
#             steps=steps,
#         )

#         if existing is not None:
#             return existing

#         strategy = Strategy(
#             name=(
#                 goal.lower()
#                 .replace(" ", "_")
#             ),
#             goal=goal,
#             steps=list(steps),
#         )

#         self.add(strategy)

#         return strategy

#     def for_goal(
#         self,
#         goal: str,
#     ) -> List[Strategy]:

#         return [
#             strategy
#             for strategy in self.strategies
#             if strategy.goal == goal
#         ]

#     def clear(self) -> None:
#         self.strategies.clear()
"""
Backward-compatible strategy store API.

The canonical implementation lives in
agent.memory.strategy_store.StrategyMemoryStore.

This module exists only so older learning code and tests
can continue importing StrategyStore from this location.
"""

from agent.memory.strategy_store import StrategyMemoryStore


StrategyStore = StrategyMemoryStore


__all__ = [
    "StrategyStore",
]