from agent.memory.causal import CausalMemory
from agent.memory.causal_store import CausalMemoryStore
from agent.memory.persistence import (
    CausalMemoryRepository,
    SQLiteDatabase,
)


def create_memory():
    return CausalMemory(
        memory_id="causal-001",
        content="open_box enables take_key",
        importance=1.0,
        confidence=0.8,
        source="causal_learning",
        tags=[
            "causal",
            "learned",
        ],
        cause="open_box",
        relation="enables",
        effect="take_key",
        condition={
            "key": "box",
            "value": "open",
        },
        evidence_count=3,
        contradiction_count=0,
        source_episodes=[
            "episode-001",
            "episode-002",
        ],
    )


def test_causal_memory_can_be_persisted(
    tmp_path,
):
    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = CausalMemoryRepository(
        database
    )

    memory = create_memory()

    repository.save(memory)

    assert repository.count() == 1

    loaded = repository.get(
        "causal-001"
    )

    assert loaded is not None
    assert loaded.cause == "open_box"
    assert loaded.relation == "enables"
    assert loaded.effect == "take_key"

    assert loaded.condition == {
        "key": "box",
        "value": "open",
    }

    assert loaded.evidence_count == 3
    assert loaded.confidence == 0.8


def test_causal_memory_survives_restart(
    tmp_path,
):
    database_path = tmp_path / "memory.db"

    database_1 = SQLiteDatabase(
        database_path
    )

    repository_1 = CausalMemoryRepository(
        database_1
    )

    repository_1.save(
        create_memory()
    )

    del repository_1
    del database_1

    database_2 = SQLiteDatabase(
        database_path
    )

    repository_2 = CausalMemoryRepository(
        database_2
    )

    loaded = repository_2.get(
        "causal-001"
    )

    assert loaded is not None
    assert loaded.cause == "open_box"
    assert loaded.effect == "take_key"
    assert loaded.evidence_count == 3


def test_causal_store_loads_persistent_memory(
    tmp_path,
):
    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = CausalMemoryRepository(
        database
    )

    store_1 = CausalMemoryStore(
        repository=repository
    )

    store_1.add(
        create_memory()
    )

    del store_1

    store_2 = CausalMemoryStore(
        repository=repository
    )

    assert store_2.count() == 1

    memory = store_2.latest()

    assert memory is not None
    assert memory.cause == "open_box"
    assert memory.effect == "take_key"


def test_causal_store_persists_reinforcement(
    tmp_path,
):
    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = CausalMemoryRepository(
        database
    )

    store = CausalMemoryStore(
        repository=repository
    )

    memory = store.reinforce(
        cause="open_box",
        relation="enables",
        effect="take_key",
        condition={
            "key": "box",
            "value": "open",
        },
        episode_ids=[
            "episode-001",
        ],
    )

    assert memory.evidence_count == 1
    assert memory.confidence == 0.6

    store.reinforce(
        cause="open_box",
        relation="enables",
        effect="take_key",
        condition={
            "key": "box",
            "value": "open",
        },
        episode_ids=[
            "episode-002",
        ],
    )

    loaded = repository.get(
        memory.memory_id
    )

    assert loaded is not None
    assert loaded.evidence_count == 2
    assert loaded.confidence == 0.7
    assert loaded.source_episodes == [
        "episode-001",
        "episode-002",
    ]
