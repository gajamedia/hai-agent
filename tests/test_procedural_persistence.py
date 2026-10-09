from agent.memory.persistence import (
    ProceduralMemoryRepository,
    SQLiteDatabase,
)
from agent.memory.procedural import (
    ProceduralMemory,
)
from agent.memory.procedural_store import (
    ProceduralMemoryStore,
)


def create_memory():

    return ProceduralMemory(
        memory_id="procedure-001",
        content="Procedure for opening the door.",
        importance=1.0,
        confidence=0.8,
        source="procedural_learning",
        tags=[
            "procedural",
            "learned",
        ],
        skill="open_the_door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=3,
        failure_count=0,
        source_episodes=[
            "episode-001",
            "episode-002",
            "episode-003",
        ],
    )


def test_procedural_memory_can_be_persisted(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = (
        ProceduralMemoryRepository(
            database
        )
    )

    memory = create_memory()

    repository.save(memory)

    loaded = repository.get(
        "procedure-001"
    )

    assert loaded is not None

    assert loaded.skill == (
        "open_the_door"
    )

    assert loaded.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    assert loaded.success_count == 3
    assert loaded.failure_count == 0

    assert loaded.source_episodes == [
        "episode-001",
        "episode-002",
        "episode-003",
    ]


def test_procedural_memory_survives_restart(
    tmp_path,
):

    database_path = (
        tmp_path / "memory.db"
    )

    database_1 = SQLiteDatabase(
        database_path
    )

    repository_1 = (
        ProceduralMemoryRepository(
            database_1
        )
    )

    repository_1.save(
        create_memory()
    )

    del repository_1
    del database_1

    database_2 = SQLiteDatabase(
        database_path
    )

    repository_2 = (
        ProceduralMemoryRepository(
            database_2
        )
    )

    loaded = repository_2.get(
        "procedure-001"
    )

    assert loaded is not None
    assert loaded.skill == (
        "open_the_door"
    )

    assert loaded.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]


def test_procedural_store_loads_persistent_memory(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = (
        ProceduralMemoryRepository(
            database
        )
    )

    store_1 = ProceduralMemoryStore(
        repository=repository
    )

    store_1.add(
        create_memory()
    )

    del store_1

    store_2 = ProceduralMemoryStore(
        repository=repository
    )

    assert store_2.count() == 1

    memory = store_2.latest()

    assert memory is not None
    assert memory.skill == (
        "open_the_door"
    )


def test_procedural_store_persists_success_update(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = (
        ProceduralMemoryRepository(
            database
        )
    )

    store = ProceduralMemoryStore(
        repository=repository
    )

    memory = store.get_or_create(
        skill="open_the_door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    memory.record_success(
        [
            "episode-001",
            "episode-002",
            "episode-003",
        ]
    )

    store.save_all()

    loaded = repository.get(
        memory.memory_id
    )

    assert loaded is not None
    assert loaded.success_count == 1
    assert loaded.confidence == 0.6
    assert loaded.source_episodes == [
        "episode-001",
        "episode-002",
        "episode-003",
    ]
