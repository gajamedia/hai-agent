from typing import Dict, Optional

from agent.core.action import Action
from agent.learning.causal_learning import CausalLearning
from agent.memory.causal import CausalMemory
from agent.memory.episodic import EpisodicMemory


class OnlineCausalLearning:
    """
    Learns causal relationships as soon as a new episode becomes available.
    """

    def __init__(self, causal_learning: CausalLearning):
        self.causal_learning = causal_learning
        self.previous_episode: Optional[EpisodicMemory] = None

    def observe_episode(
        self,
        episode: EpisodicMemory,
        actions: Dict[str, Action],
    ) -> list[CausalMemory]:
        learned_memories = []

        # Replaying the same episode must not create a self-pair or
        # reprocess the preceding pair.
        if (
            self.previous_episode is not None
            and self.previous_episode.memory_id == episode.memory_id
        ):
            return learned_memories

        if self.previous_episode is not None:
            learned_memories = (
                self.causal_learning.learn_from_episode_pair(
                    previous_episode=self.previous_episode,
                    next_episode=episode,
                    actions=actions,
                )
            )

        self.previous_episode = episode
        return learned_memories

    def reset(self) -> None:
        self.previous_episode = None
