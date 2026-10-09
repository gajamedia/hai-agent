from typing import List

from agent.learning.contradiction import (
    Contradiction,
)

from agent.learning.contradiction_detector import (
    ContradictionDetector,
)

from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_store import (
    SemanticMemoryStore,
)


class ContradictionLearning:

    def __init__(
        self,
        semantic_store: SemanticMemoryStore,
    ):
        self.semantic_store = semantic_store

        self.detector = (
            ContradictionDetector()
        )

    def check_episode(
        self,
        episode: EpisodicMemory,
    ) -> List[Contradiction]:

        contradictions = []

        for memory in self.semantic_store.all():

            contradiction = (
                self.detector.detect_semantic(
                    memory=memory,
                    episode=episode,
                )
            )

            if contradiction is None:
                continue

            memory.register_contradiction(
                episode_id=(
                    episode.memory_id
                )
            )

            contradictions.append(
                contradiction
            )

        return contradictions
