from agent.memory.persistence import (
    ReflectionMemoryRepository,
    SQLiteDatabase,
)
from agent.memory.reflection import ReflectionMemory
from agent.memory.reflection_store import (
    ReflectionMemoryStore,
)


def create_reflection() -> ReflectionMemory:

    return ReflectionMemory(
        memory_id="reflection-001",
        content=(
            "Reflection after failed "
            "action 'find_spare_key'."
        ),
        importance=1.0,
        confidence=0.7,
        source="reflection",
        tags=[
            "reflection",
            "failure",
            "learned",
        ],
        goal="open the door",
        action="find_spare_key",
        problem=(
            "Action 'find_spare_key' failed "
            "because precondition 'spare_key' "
            "was not satisfied."
        ),
        cause=(
            "spare_key was 'unavailable', "
            "expected 'available'."
        ),
        lesson=(
            "Before executing 'find_spare_key', "
            "verify that spare_key == 'available'."
        ),
        condition={
            "key": "spare_key",
            "expected": "available",
            "actual": "unavailable",
        },
        source_episodes=[
            "episode-001",
        ],
        applied=False,
    )


def test_reflection_can_be_persisted(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = ReflectionMemoryRepository(
        database
    )

    reflection = create_reflection()

    repository.save(reflection)

    assert repository.count() == 1

    loaded = repository.get(
        "reflection-001"
    )

    assert loaded is not None

    assert loaded.action == (
        "find_spare_key"
    )

    assert loaded.goal == (
        "open the door"
    )

    assert loaded.problem == (
        reflection.problem
    )

    assert loaded.cause == (
        reflection.cause
    )

    assert loaded.lesson == (
        reflection.lesson
    )

    assert loaded.condition == {
        "key": "spare_key",
        "expected": "available",
        "actual": "unavailable",
    }

    assert loaded.source_episodes == [
        "episode-001"
    ]

    assert loaded.applied is False


def test_reflection_survives_restart(
    tmp_path,
):

    database_path = (
        tmp_path / "memory.db"
    )

    database_1 = SQLiteDatabase(
        database_path
    )

    repository_1 = ReflectionMemoryRepository(
        database_1
    )

    repository_1.save(
        create_reflection()
    )

    del repository_1
    del database_1

    database_2 = SQLiteDatabase(
        database_path
    )

    repository_2 = ReflectionMemoryRepository(
        database_2
    )

    loaded = repository_2.get(
        "reflection-001"
    )

    assert loaded is not None

    assert loaded.action == (
        "find_spare_key"
    )

    assert loaded.lesson == (
        create_reflection().lesson
    )

    assert loaded.applied is False


def test_reflection_store_loads_persistent_memory(
    tmp_path,
):

    database_path = (
        tmp_path / "memory.db"
    )

    database = SQLiteDatabase(
        database_path
    )

    repository = ReflectionMemoryRepository(
        database
    )

    store_1 = ReflectionMemoryStore(
        repository=repository
    )

    store_1.add(
        create_reflection()
    )

    assert store_1.count() == 1

    del store_1

    store_2 = ReflectionMemoryStore(
        repository=repository
    )

    assert store_2.count() == 1

    reflection = store_2.latest()

    assert reflection is not None

    assert reflection.memory_id == (
        "reflection-001"
    )

    assert reflection.action == (
        "find_spare_key"
    )

    results = store_2.find_by_action(
        "find_spare_key"
    )

    assert len(results) == 1


def test_reflection_store_persists_updates(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = ReflectionMemoryRepository(
        database
    )

    store = ReflectionMemoryStore(
        repository=repository
    )

    reflection = create_reflection()

    store.add(reflection)

    assert reflection.applied is False

    reflection.applied = True

    reflection.confidence = 0.9

    store.save_all()

    loaded = repository.get(
        "reflection-001"
    )

    assert loaded is not None

    assert loaded.applied is True
    assert loaded.confidence == 0.9
