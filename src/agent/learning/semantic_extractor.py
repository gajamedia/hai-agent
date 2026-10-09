from typing import List

from agent.learning.change_detector import (
    ChangeDetector,
)

from agent.learning.context_extractor import (
    ContextExtractor,
)

from agent.memory.episodic import (
    EpisodicMemory,
)

from agent.memory.semantic import (
    SemanticMemory,
)


class SemanticExtractor:

    def __init__(self):
        self.change_detector = ChangeDetector()
        self.context_extractor = (
            ContextExtractor()
        )

    def extract(
        self,
        episode: EpisodicMemory,
    ) -> List[SemanticMemory]:

        if (
            episode.observation_before is None
            or episode.observation_after is None
        ):
            return []

        changes = self.change_detector.detect(
            episode.observation_before,
            episode.observation_after,
        )

        context = (
            self.context_extractor.extract(
                episode
            )
        )

        memories: List[SemanticMemory] = []

        for change in changes:

            memory = SemanticMemory(
                content=(
                    f"{change.key} changed "
                    f"from {change.before} "
                    f"to {change.after}."
                ),
                subject=change.key,
                predicate="changes_from",
                object={
                    "before": change.before,
                    "after": change.after,
                },
                context=context,
                confidence=1.0,
                evidence_count=1,
                source_episodes=[
                    episode.memory_id
                ],
                source="episodic_memory",
                tags=[
                    "semantic",
                    "learned",
                    change.key,
                ],
            )

            memories.append(memory)

        return memories
