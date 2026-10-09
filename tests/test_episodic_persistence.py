from agent.memory.episodic import EpisodicMemory
from agent.memory.episodic_store import (
    EpisodicMemoryStore,
)
from agent.memory.persistence import (
    EpisodicMemoryRepository,
    SQLiteDatabase,
)


def create_episode() -> EpisodicMemory:

    return EpisodicMemory(
        memory_id="episode-001",
        content="Agent opened the box.",
        importance=1.0,
        confidence=1.0,
        source="agent",
        tags=[
            "experience",
            "open_box",
            "success",
        ],
        goal="open the door",
        action="open_box",
        observation_before={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        observation_after={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
        result={
            "success": True,
            "message": (
                "The box is now open."
            ),
        },
        cycle=1,
    )


def test_episode_can_be_persisted(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = (
        EpisodicMemoryRepository(
            database
        )
    )

    episode = create_episode()

    repository.save(episode)

    assert repository.count() == 1

    loaded = repository.get(
        "episode-001"
    )

    assert loaded is not None

    assert loaded.goal == (
        "open the door"
    )

    assert loaded.action == (
        "open_box"
    )

    assert loaded.observation_before == {
        "door": "locked",
        "box": "closed",
        "key": "inside_box",
    }

    assert loaded.observation_after == {
        "door": "locked",
        "box": "open",
        "key": "inside_box",
    }

    assert loaded.result["success"] is True
    assert loaded.cycle == 1


def test_episode_survives_restart(
    tmp_path,
):

    database_path = (
        tmp_path / "memory.db"
    )

    database_1 = SQLiteDatabase(
        database_path
    )

    repository_1 = (
        EpisodicMemoryRepository(
            database_1
        )
    )

    repository_1.save(
        create_episode()
    )

    del repository_1
    del database_1

    database_2 = SQLiteDatabase(
        database_path
    )

    repository_2 = (
        EpisodicMemoryRepository(
            database_2
        )
    )

    restored = repository_2.get(
        "episode-001"
    )

    assert restored is not None
    assert restored.action == "open_box"
    assert restored.result["success"] is True
    assert restored.cycle == 1


def test_episodic_store_loads_persistent_memory(
    tmp_path,
):

    database_path = (
        tmp_path / "memory.db"
    )

    database = SQLiteDatabase(
        database_path
    )

    repository = (
        EpisodicMemoryRepository(
            database
        )
    )

    store_1 = EpisodicMemoryStore(
        repository=repository
    )

    store_1.add(
        create_episode()
    )

    assert store_1.count() == 1

    del store_1

    store_2 = EpisodicMemoryStore(
        repository=repository
    )

    assert store_2.count() == 1

    episode = store_2.latest()

    assert episode is not None
    assert episode.memory_id == (
        "episode-001"
    )
    assert episode.action == (
        "open_box"
    )


def test_episodic_store_clear_removes_memory(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = (
        EpisodicMemoryRepository(
            database
        )
    )

    store = EpisodicMemoryStore(
        repository=repository
    )

    store.add(
        create_episode()
    )

    assert store.count() == 1
    assert repository.count() == 1

    store.clear()

    assert store.count() == 0
    assert repository.count() == 0
