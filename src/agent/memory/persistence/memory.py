from pathlib import Path

from agent.memory.persistence.database import (
    SQLiteDatabase,
)
from agent.memory.persistence.semantic_repository import (
    SemanticMemoryRepository,
)
from agent.memory.persistence.episodic_repository import (
    EpisodicMemoryRepository,
)
from agent.memory.persistence.causal_repository import (
    CausalMemoryRepository,
)
from agent.memory.persistence.reflection_repository import (
    ReflectionMemoryRepository,
)
from agent.memory.persistence.self_model_repository import (
    SelfModelRepository,
)
from agent.memory.persistence.procedural_repository import ProceduralMemoryRepository


class PersistentMemory:

    def __init__(
        self,
        database_path: str | Path,
    ):
        self.database = SQLiteDatabase(
            database_path
        )

        self.semantic = (
            SemanticMemoryRepository(
                self.database
            )
        )

        self.episodic = (
            EpisodicMemoryRepository(
                self.database
            )
        )

        self.causal = (
            CausalMemoryRepository(
                self.database
            )
        )

        self.reflection = (
            ReflectionMemoryRepository(
                self.database
            )
        )

        self.procedural = (
            ProceduralMemoryRepository(
                self.database
            )
        )

        self.self_model = (
            SelfModelRepository(
                self.database
            )
        )

    def clear(self) -> None:
        self.semantic.clear()
        self.episodic.clear()
        self.causal.clear()
        self.reflection.clear()

    def counts(self) -> dict[str, int]:
        return {
            "semantic": self.semantic.count(),
            "episodic": self.episodic.count(),
            "causal": self.causal.count(),
            "reflection": self.reflection.count(),
            "procedural": self.procedural.count(),
            "self_model": (
                1
                if self.self_model.exists()
                else 0
            ),
        }
