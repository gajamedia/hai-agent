from typing import List, Optional

from agent.memory.causal import CausalMemory


class CausalMemoryStore:
    def __init__(self, repository=None):
        self.memories: List[CausalMemory] = []
        self.repository = repository

        if self.repository is not None:
            self.load()

    def add(self, memory: CausalMemory) -> None:
        # Publish locally only after persistence succeeds.
        self._persist(memory)
        self.memories.append(memory)

    def all(self) -> List[CausalMemory]:
        return list(self.memories)

    def latest(self) -> Optional[CausalMemory]:
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

        is_new = memory is None

        if is_new:
            memory = CausalMemory(
                content=f"{cause} {relation} {effect}",
                cause=cause,
                relation=relation,
                effect=effect,
                condition=condition,
                confidence=0.5,
                evidence_count=0,
                source="causal_learning",
                tags=["causal", "learned"],
            )

        # Snapshot all attributes changed by reinforce().
        old_confidence = memory.confidence
        old_evidence_count = memory.evidence_count
        old_source_episodes = list(memory.source_episodes)

        memory.reinforce(episode_ids=episode_ids)

        # A duplicate replay is a no-op; avoid unnecessary persistence.
        changed = (
            memory.confidence != old_confidence
            or memory.evidence_count != old_evidence_count
            or memory.source_episodes != old_source_episodes
        )

        if not changed and not is_new:
            return memory

        try:
            self._persist(memory)
        except Exception:
            memory.confidence = old_confidence
            memory.evidence_count = old_evidence_count
            memory.source_episodes = old_source_episodes
            raise

        if is_new:
            self.memories.append(memory)

        return memory

    def clear(self) -> None:
        # Do not discard local state if persistent clearing fails.
        if self.repository is not None:
            self.repository.clear()

        self.memories.clear()

    def load(self) -> None:
        if self.repository is None:
            return

        self.memories = self.repository.all()

    def save_all(self) -> None:
        if self.repository is None:
            return

        self.repository.save_many(self.memories)

    def _persist(self, memory: CausalMemory) -> None:
        if self.repository is None:
            return

        self.repository.save(memory)
