from agent.learning.strategy import Strategy
from agent.memory.reflection import ReflectionMemory


class ReflectionStrategyUpdater:

    def apply(
        self,
        strategy: Strategy,
        reflection: ReflectionMemory,
        penalty: float = 0.2,
    ) -> None:

        if reflection.action is None:
            return

        if reflection.action not in strategy.steps:
            return

        strategy.apply_reflection_penalty(
            amount=penalty
        )

        reflection.applied = True
