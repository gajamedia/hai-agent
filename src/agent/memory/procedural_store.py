from typing import List, Optional

from agent.memory.procedural import (
    ProceduralMemory,
)


class ProceduralMemoryStore:

    def __init__(
        self,
        repository=None,
    ):
        self.memories: List[
            ProceduralMemory
        ] = []

        self.repository = repository

        if self.repository is not None:
            self.load()

    def add(
        self,
        memory: ProceduralMemory,
    ) -> None:

        self.memories.append(memory)

        self._persist(memory)

    def all(self) -> List[ProceduralMemory]:
        return list(self.memories)

    def latest(
        self,
    ) -> Optional[ProceduralMemory]:

        if not self.memories:
            return None

        return self.memories[-1]

    def count(self) -> int:
        return len(self.memories)

    def find_exact(
        self,
        skill,
        steps,
    ):

        for memory in self.memories:

            if memory.skill != skill:
                continue

            if memory.steps != steps:
                continue

            return memory

        return None

    def get_or_create(
        self,
        skill,
        steps,
    ):

        memory = self.find_exact(
            skill,
            steps,
        )

        if memory is not None:
            return memory

        memory = ProceduralMemory(
            content=(
                f"Procedure for skill "
                f"'{skill}'."
            ),
            skill=skill,
            steps=list(steps),
            confidence=0.5,
            source="procedural_learning",
            tags=[
                "procedural",
                "learned",
            ],
        )

        self.memories.append(memory)

        self._persist(memory)

        return memory

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
        memory: ProceduralMemory,
    ) -> None:

        if self.repository is None:
            return

        self.repository.save(
            memory
        )
