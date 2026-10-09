from typing import Any, List, Optional

from agent.memory.semantic import SemanticMemory


class SemanticMemoryStore:

    def __init__(
        self,
        repository=None,
    ):
        self.memories: List[SemanticMemory] = []
        self.repository = repository

        if self.repository is not None:
            self.load()

    def add(
        self,
        memory: SemanticMemory,
    ) -> None:
        self.memories.append(memory)

        self._persist(memory)

    def all(self) -> List[SemanticMemory]:
        return list(self.memories)

    def latest(
        self,
    ) -> Optional[SemanticMemory]:
        if not self.memories:
            return None

        return self.memories[-1]

    def count(self) -> int:
        return len(self.memories)

    def find(
        self,
        subject: str,
        predicate: Optional[str] = None,
    ) -> List[SemanticMemory]:

        results = []

        for memory in self.memories:

            if memory.subject != subject:
                continue

            if (
                predicate is not None
                and memory.predicate != predicate
            ):
                continue

            results.append(memory)

        return results

    def find_exact(
        self,
        subject: str,
        predicate: str,
        object_value: Any,
        context=None,
    ) -> Optional[SemanticMemory]:

        target_context = (
            context
            if context is not None
            else {}
        )

        for memory in self.memories:

            if memory.subject != subject:
                continue

            if memory.predicate != predicate:
                continue

            if memory.object != object_value:
                continue

            if memory.context != target_context:
                continue

            return memory

        return None

    def reinforce(
        self,
        subject: str,
        predicate: str,
        object_value: Any,
        episode_id: str,
        context=None,
    ) -> SemanticMemory:

        target_context = (
            context
            if context is not None
            else {}
        )

        memory = self.find_exact(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            context=target_context,
        )

        if memory is None:

            memory = SemanticMemory(
                content=(
                    f"{subject} "
                    f"{predicate} "
                    f"{object_value}"
                ),
                subject=subject,
                predicate=predicate,
                object=object_value,
                context=dict(target_context),
                confidence=0.5,
                evidence_count=0,
                source="semantic_learning",
                tags=[
                    "semantic",
                    "learned",
                ],
            )

            self.memories.append(memory)

        memory.reinforce(
            episode_id=episode_id,
        )

        self._persist(memory)

        return memory

    def find_by_context(
        self,
        context,
    ) -> List[SemanticMemory]:

        return [
            memory
            for memory in self.memories
            if memory.matches_context(context)
        ]

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
        memory: SemanticMemory,
    ) -> None:

        if self.repository is None:
            return

        self.repository.save(memory)
