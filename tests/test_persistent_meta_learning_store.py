from agent.learning.meta_learning import (
    MetaLearningProfile,
)
from agent.learning.meta_learning_repository import (
    MetaLearningRepository,
)
from agent.learning.persistent_meta_learning_store import (
    PersistentMetaLearningStore,
)
import pytest


def create_store(
    tmp_path,
):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    return PersistentMetaLearningStore(
        repository
    )


def test_store_requires_repository():
    with pytest.raises(
        ValueError,
        match="repository cannot be None",
    ):
        PersistentMetaLearningStore(
            None
        )


def test_get_or_create_creates_profile(tmp_path):
    store = create_store(
        tmp_path
    )

    profile = store.get_or_create(
        "agent-a"
    )

    assert isinstance(
        profile,
        MetaLearningProfile,
    )

    assert store.get(
        "agent-a"
    ) is profile


def test_get_or_create_returns_same_memory_profile(
    tmp_path,
):
    store = create_store(
        tmp_path
    )

    first = store.get_or_create(
        "agent-a"
    )

    second = store.get_or_create(
        "agent-a"
    )

    assert first is second
    assert store.count() == 1


def test_save_persists_profile(tmp_path):
    store = create_store(
        tmp_path
    )

    profile = store.get_or_create(
        "agent-a"
    )

    profile.record_exploration(
        success=True,
        event_id="event-1",
    )

    store.save(
        "agent-a"
    )

    assert (
        store.persistent_count()
        == 1
    )

    loaded = (
        store.repository.get(
            "agent-a"
        )
    )

    assert loaded is not None
    assert (
        loaded.exploration_count
        == 1
    )


def test_get_loads_profile_from_repository(
    tmp_path,
):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    profile = MetaLearningProfile()

    profile.record_exploitation(
        success=True,
        event_id="persistent-event",
    )

    repository.save(
        "agent-a",
        profile,
    )

    store = PersistentMetaLearningStore(
        repository
    )

    loaded = store.get(
        "agent-a"
    )

    assert loaded is not None
    assert (
        loaded.exploitation_count
        == 1
    )

    assert (
        loaded.learning_events
        == ["persistent-event"]
    )


def test_loaded_profile_is_cached(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="event-1",
    )

    repository.save(
        "agent-a",
        profile,
    )

    store = PersistentMetaLearningStore(
        repository
    )

    first = store.get(
        "agent-a"
    )

    second = store.get(
        "agent-a"
    )

    assert first is second


def test_add_replaces_memory_profile(tmp_path):
    store = create_store(
        tmp_path
    )

    first = MetaLearningProfile()

    store.add(
        "agent-a",
        first,
    )

    second = MetaLearningProfile()

    second.record_exploration(
        success=True,
        event_id="replacement",
    )

    result = store.add(
        "agent-a",
        second,
    )

    assert result is second
    assert (
        store.get("agent-a")
        is second
    )


def test_save_accepts_explicit_profile(tmp_path):
    store = create_store(
        tmp_path
    )

    profile = MetaLearningProfile()

    profile.record_evolution(
        success=False,
        event_id="evolution-1",
    )

    saved = store.save(
        "agent-a",
        profile,
    )

    assert saved is profile

    loaded = store.repository.get(
        "agent-a"
    )

    assert loaded is not None
    assert (
        loaded.evolution_count
        == 1
    )


def test_save_without_existing_profile_raises(
    tmp_path,
):
    store = create_store(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="profile cannot be None",
    ):
        store.save(
            "missing"
        )


def test_save_all_persists_all_cached_profiles(
    tmp_path,
):
    store = create_store(
        tmp_path
    )

    profile_a = store.get_or_create(
        "agent-a"
    )

    profile_a.record_exploration(
        success=True,
        event_id="a-1",
    )

    profile_b = store.get_or_create(
        "agent-b"
    )

    profile_b.record_exploitation(
        success=True,
        event_id="b-1",
    )

    store.save_all()

    assert (
        store.persistent_count()
        == 2
    )


def test_remove_only_removes_memory_cache(
    tmp_path,
):
    store = create_store(
        tmp_path
    )

    profile = store.get_or_create(
        "agent-a"
    )

    profile.record_exploration(
        success=True,
        event_id="event-1",
    )

    store.save(
        "agent-a"
    )

    assert store.remove(
        "agent-a"
    )

    assert store.get(
        "agent-a"
    ) is not None

    assert (
        store.persistent_count()
        == 1
    )


def test_remove_missing_profile_returns_false(
    tmp_path,
):
    store = create_store(
        tmp_path
    )

    assert not store.remove(
        "missing"
    )


def test_delete_removes_memory_and_persistence(
    tmp_path,
):
    store = create_store(
        tmp_path
    )

    store.get_or_create(
        "agent-a"
    )

    store.save(
        "agent-a"
    )

    assert store.delete(
        "agent-a"
    )

    assert store.get(
        "agent-a"
    ) is None

    assert (
        store.persistent_count()
        == 0
    )


def test_clear_cache_does_not_delete_database(
    tmp_path,
):
    store = create_store(
        tmp_path
    )

    profile = store.get_or_create(
        "agent-a"
    )

    profile.record_exploration(
        success=True,
        event_id="event-1",
    )

    store.save(
        "agent-a"
    )

    store.clear_cache()

    assert store.count() == 0
    assert store.persistent_count() == 1

    loaded = store.get(
        "agent-a"
    )

    assert loaded is not None
    assert (
        loaded.exploration_count
        == 1
    )


def test_clear_removes_cache_and_database(
    tmp_path,
):
    store = create_store(
        tmp_path
    )

    store.get_or_create(
        "agent-a"
    )

    store.get_or_create(
        "agent-b"
    )

    store.save_all()

    store.clear()

    assert store.count() == 0
    assert store.persistent_count() == 0


def test_empty_agent_name_is_rejected(tmp_path):
    store = create_store(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        store.get("")

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        store.get_or_create("")

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        store.add(
            "",
            MetaLearningProfile(),
        )

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        store.save("")

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        store.remove("")

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        store.delete("")


def test_restart_recovery():
    """
    Simulate a process restart by creating a new repository
    and a completely new store using the same SQLite database.
    """

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:

        database = (
            Path(directory)
            / "meta_learning.db"
        )

        # ---------------------------------------------
        # Process 1
        # ---------------------------------------------

        repository_1 = (
            MetaLearningRepository(
                database
            )
        )

        store_1 = (
            PersistentMetaLearningStore(
                repository_1
            )
        )

        profile_1 = store_1.get_or_create(
            "agent-restart"
        )

        profile_1.record_exploration(
            success=True,
            event_id="restart-1",
        )

        profile_1.record_progress(
            progress_delta=0.5,
            event_id="restart-progress",
        )

        store_1.save(
            "agent-restart"
        )

        # ---------------------------------------------
        # Process 2
        # ---------------------------------------------

        repository_2 = (
            MetaLearningRepository(
                database
            )
        )

        store_2 = (
            PersistentMetaLearningStore(
                repository_2
            )
        )

        restored = store_2.get(
            "agent-restart"
        )

        assert restored is not None

        assert (
            restored.exploration_count
            == 1
        )

        assert (
            restored.total_progress
            == 0.5
        )

        assert (
            restored.learning_events
            == [
                "restart-1",
                "restart-progress",
            ]
        )
