from agent.memory.persistence import (
    SQLiteDatabase,
    SemanticMemoryRepository,
)
from agent.memory.semantic import SemanticMemory


def create_memory() -> SemanticMemory:
    return SemanticMemory(
        memory_id="memory-001",
        content="box changes from closed to open",
        importance=0.9,
        confidence=0.85,
        source="test",
        tags=[
            "semantic",
            "learned",
            "box",
        ],
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        evidence_count=8,
        contradiction_count=1,
        source_episodes=[
            "episode-001",
            "episode-002",
        ],
        context={
            "condition": "normal",
        },
    )


def test_semantic_memory_can_be_saved_and_loaded(
    tmp_path,
):
    database_path = (
        tmp_path / "memory.db"
    )

    database = SQLiteDatabase(
        database_path
    )

    repository = SemanticMemoryRepository(
        database
    )

    memory = create_memory()

    repository.save(memory)

    assert repository.count() == 1

    loaded = repository.get(
        "memory-001"
    )

    assert loaded is not None

    assert loaded.memory_id == memory.memory_id
    assert loaded.content == memory.content
    assert loaded.importance == memory.importance
    assert loaded.confidence == memory.confidence
    assert loaded.source == memory.source
    assert loaded.tags == memory.tags

    assert loaded.subject == memory.subject
    assert loaded.predicate == memory.predicate
    assert loaded.object == memory.object

    assert (
        loaded.evidence_count
        == memory.evidence_count
    )

    assert (
        loaded.contradiction_count
        == memory.contradiction_count
    )

    assert (
        loaded.source_episodes
        == memory.source_episodes
    )

    assert loaded.context == memory.context


def test_semantic_memory_survives_repository_restart(
    tmp_path,
):
    database_path = (
        tmp_path / "memory.db"
    )

    # Session 1
    database_1 = SQLiteDatabase(
        database_path
    )

    repository_1 = SemanticMemoryRepository(
        database_1
    )

    memory = create_memory()

    repository_1.save(memory)

    assert repository_1.count() == 1

    # Simulasi proses Python/session pertama selesai.
    del repository_1
    del database_1

    # Session 2
    database_2 = SQLiteDatabase(
        database_path
    )

    repository_2 = SemanticMemoryRepository(
        database_2
    )

    loaded = repository_2.get(
        "memory-001"
    )

    assert loaded is not None
    assert loaded.subject == "box"
    assert loaded.predicate == "changes_from"

    assert loaded.object == {
        "before": "closed",
        "after": "open",
    }

    assert loaded.context == {
        "condition": "normal",
    }

    assert loaded.confidence == 0.85
    assert loaded.evidence_count == 8
    assert loaded.contradiction_count == 1


def test_semantic_memory_can_be_updated(
    tmp_path,
):
    database_path = (
        tmp_path / "memory.db"
    )

    database = SQLiteDatabase(
        database_path
    )

    repository = SemanticMemoryRepository(
        database
    )

    memory = create_memory()

    repository.save(memory)

    memory.confidence = 0.95
    memory.evidence_count = 10

    repository.save(memory)

    loaded = repository.get(
        "memory-001"
    )

    assert loaded is not None
    assert loaded.confidence == 0.95
    assert loaded.evidence_count == 10
    assert repository.count() == 1


def test_semantic_memory_can_be_deleted(
    tmp_path,
):
    database_path = (
        tmp_path / "memory.db"
    )

    database = SQLiteDatabase(
        database_path
    )

    repository = SemanticMemoryRepository(
        database
    )

    repository.save(
        create_memory()
    )

    assert repository.count() == 1

    deleted = repository.delete(
        "memory-001"
    )

    assert deleted is True
    assert repository.count() == 0

    loaded = repository.get(
        "memory-001"
    )

    assert loaded is None
