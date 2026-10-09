from typing import Dict, List

from agent.core.action import Action
from agent.learning.causal_extractor import CausalExtractor
from agent.memory.causal import CausalMemory
from agent.memory.causal_store import CausalMemoryStore
from agent.memory.episodic import EpisodicMemory


class CausalLearning:
    def __init__(
        self,
        causal_store: CausalMemoryStore,
    ):
        self.causal_store = causal_store
        self.extractor = CausalExtractor()

    def learn_from_episode_pair(
        self,
        previous_episode: EpisodicMemory,
        next_episode: EpisodicMemory,
        actions: Dict[str, Action],
    ) -> List[CausalMemory]:

        relationships = self.extractor.extract(
            previous_episode=previous_episode,
            next_episode=next_episode,
            actions=actions,
        )

        learned_memories = []

        for relationship in relationships:

            memory = self.causal_store.reinforce(
                cause=relationship["cause"],
                relation=relationship["relation"],
                effect=relationship["effect"],
                condition=relationship["condition"],
                episode_ids=[
                    previous_episode.memory_id,
                    next_episode.memory_id,
                ],
            )

            learned_memories.append(memory)

        return learned_memories


    def learn_from_episodes(
        self,
        episodes: List[EpisodicMemory],
        actions: Dict[str, Action],
    ) -> List[CausalMemory]:

        learned_memories = []

        if len(episodes) < 2:
            return learned_memories

        for index in range(
            len(episodes) - 1
        ):

            previous_episode = episodes[index]
            next_episode = episodes[index + 1]

            relationships = self.extractor.extract(
                previous_episode=previous_episode,
                next_episode=next_episode,
                actions=actions,
            )

            for relationship in relationships:

                memory = self.causal_store.reinforce(
                    cause=relationship["cause"],
                    relation=relationship["relation"],
                    effect=relationship["effect"],
                    condition=relationship["condition"],
                    episode_ids=[
                        previous_episode.memory_id,
                        next_episode.memory_id,
                    ],
                )

                learned_memories.append(memory)

        return learned_memories
