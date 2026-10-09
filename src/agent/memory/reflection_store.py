from typing import List, Optional

from agent.memory.reflection import ReflectionMemory


class ReflectionMemoryStore:

    def __init__(
        self,
        repository=None,
    ):
        self.memories: List[
            ReflectionMemory
        ] = []

        self.repository = repository

        if self.repository is not None:
            self.load()

    def add(
        self,
        memory: ReflectionMemory,
    ) -> None:

        self.memories.append(memory)

        self._persist(memory)

    def all(
        self,
    ) -> List[ReflectionMemory]:

        return list(self.memories)

    def latest(
        self,
    ) -> Optional[ReflectionMemory]:

        if not self.memories:
            return None

        return self.memories[-1]

    def count(self) -> int:
        return len(self.memories)

    def find_by_action(
        self,
        action: str,
    ) -> List[ReflectionMemory]:

        return [
            memory
            for memory in self.memories
            if memory.action == action
        ]

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
        memory: ReflectionMemory,
    ) -> None:

        if self.repository is None:
            return

        self.repository.save(memory)
