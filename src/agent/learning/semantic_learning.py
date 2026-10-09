from typing import List

from agent.learning.semantic_extractor import SemanticExtractor
from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_store import SemanticMemoryStore

class SemanticLearning:

    def __init__(
        self,
        semantic_store: SemanticMemoryStore,
    ):
        self.semantic_store = semantic_store
        self.extractor = SemanticExtractor()

    def learn_from_episode(
        self,
        episode: EpisodicMemory,
    ) -> List[SemanticMemory]:

        extracted_memories = (
            self.extractor.extract(
                episode
            )
        )

        learned_memories = []

        for memory in extracted_memories:

            learned_memory = (
                self.semantic_store.reinforce(
                    subject=memory.subject,
                    predicate=memory.predicate,
                    object_value=memory.object,
                    episode_id=episode.memory_id,
                    context=memory.context,
                )
            )

            learned_memories.append(
                learned_memory
            )

        return learned_memories
