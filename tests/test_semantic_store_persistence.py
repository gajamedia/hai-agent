from agent.memory.persistence import (
    SQLiteDatabase,
    SemanticMemoryRepository,
)
from agent.memory.semantic_store import (
    SemanticMemoryStore,
)


def test_store_persists_new_memory(
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

    store = SemanticMemoryStore(
        repository=repository
    )

    memory = store.reinforce(
        subject="box",
        predicate="changes_from",
        object_value={
            "before": "closed",
            "after": "open",
        },
        episode_id="episode-001",
        context={
            "condition": "normal",
        },
    )

    assert store.count() == 1
    assert repository.count() == 1

    persisted = repository.get(
        memory.memory_id
    )

    assert persisted is not None
    assert persisted.subject == "box"
    assert persisted.confidence == 0.6
    assert persisted.evidence_count == 1


def test_store_loads_existing_memories(
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

    store_1 = SemanticMemoryStore(
        repository=repository
    )

    store_1.reinforce(
        subject="box",
        predicate="changes_from",
        object_value={
            "before": "closed",
            "after": "open",
        },
        episode_id="episode-001",
        context={
            "condition": "normal",
        },
    )

    del store_1

    store_2 = SemanticMemoryStore(
        repository=repository
    )

    assert store_2.count() == 1

    memory = store_2.find_exact(
        subject="box",
        predicate="changes_from",
        object_value={
            "before": "closed",
            "after": "open",
        },
        context={
            "condition": "normal",
        },
    )

    assert memory is not None
    assert memory.evidence_count == 1
    assert memory.confidence == 0.6


def test_store_persists_reinforcement(
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

    store = SemanticMemoryStore(
        repository=repository
    )

    first = store.reinforce(
        subject="box",
        predicate="changes_from",
        object_value={
            "before": "closed",
            "after": "open",
        },
        episode_id="episode-001",
    )

    assert first.evidence_count == 1
    assert first.confidence == 0.6

    second = store.reinforce(
        subject="box",
        predicate="changes_from",
        object_value={
            "before": "closed",
            "after": "open",
        },
        episode_id="episode-002",
    )

    assert second.evidence_count == 2
    assert second.confidence == 0.7

    persisted = repository.get(
        first.memory_id
    )

    assert persisted is not None
    assert persisted.evidence_count == 2
    assert persisted.confidence == 0.7
    assert persisted.source_episodes == [
        "episode-001",
        "episode-002",
    ]


def test_store_clear_removes_persistent_memory(
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

    store = SemanticMemoryStore(
        repository=repository
    )

    store.reinforce(
        subject="box",
        predicate="changes_from",
        object_value={
            "before": "closed",
            "after": "open",
        },
        episode_id="episode-001",
    )

    assert store.count() == 1
    assert repository.count() == 1

    store.clear()

    assert store.count() == 0
    assert repository.count() == 0
