from typing import List, Optional

from agent.memory.causal import CausalMemory


class CausalMemoryStore:

    def __init__(
        self,
        repository=None,
    ):
        self.memories: List[
            CausalMemory
        ] = []

        self.repository = repository

        if self.repository is not None:
            self.load()

    def add(
        self,
        memory: CausalMemory,
    ) -> None:
        self.memories.append(memory)
        self._persist(memory)

    def all(self) -> List[CausalMemory]:
        return list(self.memories)

    def latest(
        self,
    ) -> Optional[CausalMemory]:
        if not self.memories:
            return None

        return self.memories[-1]

    def count(self) -> int:
        return len(self.memories)

    def find_exact(
        self,
        cause,
        relation,
        effect,
        condition,
    ):
        for memory in self.memories:
            if memory.cause != cause:
                continue

            if memory.relation != relation:
                continue

            if memory.effect != effect:
                continue

            if memory.condition != condition:
                continue

            return memory

        return None

    def reinforce(
        self,
        cause,
        relation,
        effect,
        condition,
        episode_ids,
    ):
        memory = self.find_exact(
            cause,
            relation,
            effect,
            condition,
        )

        if memory is None:
            memory = CausalMemory(
                content=(
                    f"{cause} "
                    f"{relation} "
                    f"{effect}"
                ),
                cause=cause,
                relation=relation,
                effect=effect,
                condition=condition,
                confidence=0.5,
                evidence_count=0,
                source="causal_learning",
                tags=[
                    "causal",
                    "learned",
                ],
            )

            self.memories.append(memory)

        memory.reinforce(
            episode_ids=episode_ids
        )

        self._persist(memory)

        return memory

    def clear(self) -> None:
        self.memories.clear()

        if self.repository is not None:
            self.repository.clear()

    def load(self) -> None:
        if self.repository is None:
            return

        self.memories = self.repository.all()

    def save_all(self) -> None:
        if self.repository is None:
            return

        self.repository.save_many(
            self.memories
        )

    def _persist(
        self,
        memory: CausalMemory,
    ) -> None:
        if self.repository is None:
            return

        self.repository.save(memory)
