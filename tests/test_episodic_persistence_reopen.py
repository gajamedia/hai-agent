from agent.memory.episodic_store import EpisodicMemoryStore
from agent.memory.persistence import (
    EpisodicMemoryRepository,
    SQLiteDatabase,
)


def test_reopening_database_does_not_duplicate_episodes(
    tmp_path,
):
    database_path = tmp_path / "reopen_memory.db"

    database = SQLiteDatabase(database_path)
    repository = EpisodicMemoryRepository(database)
    store = EpisodicMemoryStore(repository=repository)

    # Persist one episode through the existing store API.
    from agent.memory import EpisodicMemory

    episode = EpisodicMemory(
        memory_id="stable-episode-id",
        content="Agent took the key and failed",
        importance=0.8,
        confidence=1.0,
        source="persistence-test",
        tags=["experience", "take_key", "failure"],
        goal="open the door",
        action="take_key",
        observation_before={"door": "locked"},
        observation_after=None,
        result={"success": False},
        cycle=1,
    )
    store.add(episode)

    assert repository.count() == 1

    # Reopen the same database repeatedly with fresh objects.
    for _ in range(3):
        reopened_database = SQLiteDatabase(database_path)
        reopened_repository = EpisodicMemoryRepository(
            reopened_database
        )
        reopened_store = EpisodicMemoryStore(
            repository=reopened_repository
        )

        assert reopened_repository.count() == 1
        assert reopened_store.count() == 1

        restored = reopened_store.latest()
        assert restored.memory_id == "stable-episode-id"
        assert restored.action == "take_key"
        assert restored.result["success"] is False
