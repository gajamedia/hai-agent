from typing import List, Optional

from agent.memory.episodic import EpisodicMemory


class EpisodicMemoryStore:

    def __init__(
        self,
        repository=None,
    ):
        self.memories: List[
            EpisodicMemory
        ] = []

        self.repository = repository

        if self.repository is not None:
            self.load()

    def add(
        self,
        memory: EpisodicMemory,
    ) -> None:

        self.memories.append(memory)

        self._persist(memory)

    def all(
        self,
    ) -> List[EpisodicMemory]:

        return list(self.memories)

    def latest(
        self,
    ) -> Optional[EpisodicMemory]:

        if not self.memories:
            return None

        return self.memories[-1]

    def count(self) -> int:
        return len(self.memories)

    def clear(self) -> None:

        self.memories.clear()

        if self.repository is not None:
            self.repository.clear()

    def load(self) -> None:

        if self.repository is None:
            return

        self.memories = (
            self.repository.all()
        )

    def save_all(self) -> None:

        if self.repository is None:
            return

        self.repository.save_many(
            self.memories
        )

    def _persist(
        self,
        memory: EpisodicMemory,
    ) -> None:

        if self.repository is None:
            return

        self.repository.save(memory)
