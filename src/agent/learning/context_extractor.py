from typing import Any, Dict

from agent.memory.episodic import EpisodicMemory


class ContextExtractor:

    def extract(
        self,
        episode: EpisodicMemory,
    ) -> Dict[str, Any]:

        if episode.observation_before is None:
            return {}

        return dict(
            episode.observation_before
        )
