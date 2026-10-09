from agent.core.experience import Experience
from agent.core.experience_store import ExperienceStore
from agent.memory.episodic import EpisodicMemory


class ExperienceRecorder:

    def __init__(
        self,
        store: ExperienceStore,
    ):
        self.store = store

        self.current_experience = None

    def start(
        self,
        experience_id: str,
        goal: str,
    ) -> Experience:

        experience = Experience(
            experience_id=experience_id,
            goal=goal,
        )

        self.current_experience = experience

        self.store.add(
            experience
        )

        return experience

    def add_episode(
        self,
        episode: EpisodicMemory,
    ) -> None:

        if self.current_experience is None:
            raise RuntimeError(
                "No active experience."
            )

        self.current_experience.add_episode(
            episode
        )

    def complete(self) -> None:

        if self.current_experience is None:
            raise RuntimeError(
                "No active experience."
            )

        self.current_experience.complete()

    def fail(self) -> None:

        if self.current_experience is None:
            raise RuntimeError(
                "No active experience."
            )

        self.current_experience.fail()

    def close(self) -> None:

        self.current_experience = None
