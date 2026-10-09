from typing import Optional

from agent.core.action import Action
from agent.learning.failure_analyzer import (
    FailureAnalysis,
    FailureAnalyzer,
)
from agent.memory.episodic import EpisodicMemory
from agent.memory.reflection import ReflectionMemory
from agent.memory.reflection_store import (
    ReflectionMemoryStore,
)


class ReflectionEngine:

    def __init__(
        self,
        reflection_store: ReflectionMemoryStore,
    ):
        self.reflection_store = reflection_store
        self.failure_analyzer = FailureAnalyzer()

    def reflect(
        self,
        episode: EpisodicMemory,
        action: Action,
    ) -> Optional[ReflectionMemory]:

        if episode.result is None:
            return None

        if episode.result.get(
            "success",
            False,
        ):
            return None

        if episode.observation_before is None:
            return None

        analysis: Optional[
            FailureAnalysis
        ] = self.failure_analyzer.analyze(
            action=action,
            world_state=episode.observation_before,
            result=episode.result,
        )

        if analysis is None:
            return None

        memory = ReflectionMemory(
            content=(
                f"Reflection after failed "
                f"action '{action.name}'."
            ),
            goal=episode.goal,
            action=analysis.action,
            problem=analysis.problem,
            cause=analysis.cause,
            lesson=analysis.lesson,
            condition=analysis.condition,
            confidence=0.7,
            source="reflection",
            source_episodes=[
                episode.memory_id
            ],
            tags=[
                "reflection",
                "failure",
                "learned",
            ],
        )

        self.reflection_store.add(
            memory
        )

        return memory
