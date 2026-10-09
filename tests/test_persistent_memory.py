from agent.memory.persistence import (
    PersistentMemory,
)


def test_persistent_memory_creates_all_repositories(
    tmp_path,
):
    memory = PersistentMemory(
        tmp_path / "agent.db"
    )

    assert memory.semantic is not None
    assert memory.episodic is not None
    assert memory.causal is not None
    assert memory.reflection is not None


def test_persistent_memory_counts_all_memory_types(
    tmp_path,
):
    memory = PersistentMemory(
        tmp_path / "agent.db"
    )

    assert memory.counts() == {
        "semantic": 0,
        "episodic": 0,
        "causal": 0,
        "reflection": 0,
        "procedural": 0,
        "self_model": 0,
    }

def test_persistent_memory_uses_one_database(
    tmp_path,
):
    database_path = (
        tmp_path / "agent.db"
    )

    memory = PersistentMemory(
        database_path
    )

    assert (
        memory.database.database_path
        == database_path
    )


def test_persistent_memory_can_clear_all(
    tmp_path,
):
    memory = PersistentMemory(
        tmp_path / "agent.db"
    )

    assert memory.counts() == {
        "semantic": 0,
        "episodic": 0,
        "causal": 0,
        "reflection": 0,
        "procedural": 0,
        "self_model": 0,
    }

    memory.clear()

    assert memory.counts() == {
        "semantic": 0,
        "episodic": 0,
        "causal": 0,
        "reflection": 0,
        "procedural": 0,
        "self_model": 0,
    }
