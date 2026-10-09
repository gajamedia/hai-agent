# from typing import List, Optional

# from agent.memory.episodic import EpisodicMemory
# from agent.memory.procedural import ProceduralMemory
# from agent.memory.procedural_store import (
#     ProceduralMemoryStore,
# )


class ProceduralLearning:

    def __init__(
        self,
        procedural_store,
    ):
        self.procedural_store = (
            procedural_store
        )

    def learn_from_episode_sequence(
        self,
        goal,
        episodes,
    ):

        if not episodes:
            return None

        successful = [
            episode
            for episode in episodes
            if (
                episode.result is not None
                and episode.result.get(
                    "success",
                    False,
                )
            )
        ]

        if len(successful) != len(
            episodes
        ):
            return None

        steps = [
            episode.action
            for episode in episodes
            if episode.action is not None
        ]

        if not steps:
            return None

        skill = (
            goal
            .strip()
            .lower()
            .replace(" ", "_")
        )

        memory = (
            self.procedural_store.get_or_create(
                skill,
                steps,
            )
        )

        memory.record_success(
            [
                episode.memory_id
                for episode in episodes
            ]
        )

        self.procedural_store.save_all()

        return memory
