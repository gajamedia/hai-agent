from pathlib import Path

from agent.memory.persistence import (
    SQLiteDatabase,
    SemanticMemoryRepository,
)
from agent.memory.semantic import SemanticMemory


DATABASE_PATH = (
    Path("data") / "experiment_020.db"
)


def create_memory() -> SemanticMemory:
    return SemanticMemory(
        memory_id="persistent-box-memory",
        content=(
            "box changes from closed to open "
            "under normal condition"
        ),
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
        confidence=0.85,
        evidence_count=8,
        contradiction_count=0,
        source="experiment_020",
        tags=[
            "semantic",
            "learned",
            "persistent",
        ],
    )


def main():
    print("=" * 40)
    print("PERSISTENT SEMANTIC MEMORY")
    print("=" * 40)

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    print()
    print("SESSION 1")
    print("-" * 40)

    database_1 = SQLiteDatabase(
        DATABASE_PATH
    )

    repository_1 = SemanticMemoryRepository(
        database_1
    )

    memory = create_memory()

    repository_1.save(memory)

    print(
        "Saved memory:",
        memory.memory_id,
    )

    print(
        "Database count:",
        repository_1.count(),
    )

    del repository_1
    del database_1

    print()
    print("SESSION 2")
    print("-" * 40)

    database_2 = SQLiteDatabase(
        DATABASE_PATH
    )

    repository_2 = SemanticMemoryRepository(
        database_2
    )

    restored = repository_2.get(
        "persistent-box-memory"
    )

    if restored is None:
        print("Memory was NOT restored.")
        return

    print(
        "Restored memory:",
        restored.memory_id,
    )

    print(
        "Subject:",
        restored.subject,
    )

    print(
        "Predicate:",
        restored.predicate,
    )

    print(
        "Object:",
        restored.object,
    )

    print(
        "Context:",
        restored.context,
    )

    print(
        "Confidence:",
        restored.confidence,
    )

    print(
        "Evidence:",
        restored.evidence_count,
    )

    print()
    print("RESULT")
    print("-" * 40)

    if (
        restored.subject == "box"
        and restored.predicate == "changes_from"
        and restored.context
        == {"condition": "normal"}
        and restored.object
        == {
            "before": "closed",
            "after": "open",
        }
    ):
        print(
            "SUCCESS: memory survived restart."
        )
    else:
        print(
            "FAILURE: restored memory is incorrect."
        )


if __name__ == "__main__":
    main()
