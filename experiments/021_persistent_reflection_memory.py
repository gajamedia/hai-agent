from pathlib import Path

from agent.memory.persistence import (
    ReflectionMemoryRepository,
    SQLiteDatabase,
)
from agent.memory.reflection import ReflectionMemory


DATABASE_PATH = (
    Path("data")
    / "experiment_021.db"
)


def create_reflection() -> ReflectionMemory:

    return ReflectionMemory(
        memory_id="reflection-restart-001",
        content=(
            "Reflection after failed "
            "find_spare_key."
        ),
        goal="open the door",
        action="find_spare_key",
        problem=(
            "spare_key was unavailable."
        ),
        cause=(
            "The planner did not verify "
            "spare_key availability."
        ),
        lesson=(
            "Check spare_key before "
            "executing find_spare_key."
        ),
        condition={
            "spare_key": "unavailable",
        },
        source_episodes=[
            "episode-001",
        ],
        confidence=0.7,
        importance=1.0,
        source="reflection",
        tags=[
            "reflection",
            "failure",
        ],
        applied=False,
    )


def main():

    print("=" * 40)
    print("PERSISTENT REFLECTION MEMORY")
    print("=" * 40)

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    print()
    print("SESSION 1")
    print("-" * 40)

    database_1 = SQLiteDatabase(
        DATABASE_PATH
    )

    repository_1 = (
        ReflectionMemoryRepository(
            database_1
        )
    )

    reflection = create_reflection()

    repository_1.save(reflection)

    print(
        "Saved reflection:",
        reflection.memory_id,
    )

    print(
        "Action:",
        reflection.action,
    )

    print(
        "Lesson:",
        reflection.lesson,
    )

    del repository_1
    del database_1

    print()
    print("SESSION 2")
    print("-" * 40)

    database_2 = SQLiteDatabase(
        DATABASE_PATH
    )

    repository_2 = (
        ReflectionMemoryRepository(
            database_2
        )
    )

    restored = repository_2.get(
        "reflection-restart-001"
    )

    if restored is None:
        print(
            "FAILURE: reflection was not restored."
        )
        return

    print(
        "Restored reflection:",
        restored.memory_id,
    )

    print(
        "Action:",
        restored.action,
    )

    print(
        "Problem:",
        restored.problem,
    )

    print(
        "Cause:",
        restored.cause,
    )

    print(
        "Lesson:",
        restored.lesson,
    )

    print(
        "Condition:",
        restored.condition,
    )

    print()
    print("RESULT")
    print("-" * 40)

    if (
        restored.action == "find_spare_key"
        and restored.condition
        == {"spare_key": "unavailable"}
        and restored.applied is False
    ):
        print(
            "SUCCESS: reflection survived restart."
        )
    else:
        print(
            "FAILURE: restored reflection "
            "is incorrect."
        )


if __name__ == "__main__":
    main()
