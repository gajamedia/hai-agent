from pathlib import Path

import pytest

from agent.learning.meta_learning import (
    MetaLearningProfile,
)
from agent.learning.meta_learning_repository import (
    MetaLearningRepository,
)


def create_profile() -> MetaLearningProfile:
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="explore-1",
    )

    profile.record_exploration(
        success=False,
        event_id="explore-2",
    )

    profile.record_exploitation(
        success=True,
        event_id="exploit-1",
    )

    profile.record_evolution(
        success=True,
        event_id="evolution-1",
    )

    profile.record_recovery(
        success=False,
        event_id="recovery-1",
    )

    profile.record_progress(
        progress_delta=0.5,
        event_id="progress-1",
    )

    return profile


def test_repository_creates_database(tmp_path):
    database = (
        tmp_path / "meta_learning.db"
    )

    repository = MetaLearningRepository(
        database
    )

    assert database.exists()
    assert repository.count() == 0


def test_save_and_get_profile(tmp_path):
    database = (
        tmp_path / "meta_learning.db"
    )

    repository = MetaLearningRepository(
        database
    )

    profile = create_profile()

    repository.save(
        agent_name="agent-a",
        profile=profile,
    )

    loaded = repository.get(
        "agent-a"
    )

    assert loaded is not None

    assert (
        loaded.exploration_count
        == profile.exploration_count
    )

    assert (
        loaded.exploitation_count
        == profile.exploitation_count
    )

    assert (
        loaded.evolution_count
        == profile.evolution_count
    )

    assert (
        loaded.recovery_count
        == profile.recovery_count
    )

    assert (
        loaded.total_progress
        == profile.total_progress
    )

    assert (
        loaded.progress_observation_count
        == profile.progress_observation_count
    )

    assert (
        loaded.learning_events
        == profile.learning_events
    )


def test_profile_rates_survive_round_trip(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    profile = create_profile()

    repository.save(
        "agent-a",
        profile,
    )

    loaded = repository.get(
        "agent-a"
    )

    assert loaded is not None

    assert (
        loaded.exploration_success_rate
        == profile.exploration_success_rate
    )

    assert (
        loaded.exploitation_success_rate
        == profile.exploitation_success_rate
    )

    assert (
        loaded.evolution_success_rate
        == profile.evolution_success_rate
    )

    assert (
        loaded.recovery_success_rate
        == profile.recovery_success_rate
    )

    assert (
        loaded.average_progress
        == profile.average_progress
    )


def test_save_updates_existing_profile(tmp_path):
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

    profile.record_exploration(
        success=True,
        event_id="event-2",
    )

    repository.save(
        "agent-a",
        profile,
    )

    assert repository.count() == 1

    loaded = repository.get(
        "agent-a"
    )

    assert loaded is not None
    assert loaded.exploration_count == 2


def test_exists(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    assert not repository.exists(
        "agent-a"
    )

    repository.save(
        "agent-a",
        MetaLearningProfile(),
    )

    assert repository.exists(
        "agent-a"
    )


def test_get_missing_profile_returns_none(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    assert (
        repository.get("missing")
        is None
    )


def test_delete_profile(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    repository.save(
        "agent-a",
        MetaLearningProfile(),
    )

    assert repository.delete(
        "agent-a"
    )

    assert not repository.exists(
        "agent-a"
    )

    assert repository.get(
        "agent-a"
    ) is None


def test_delete_missing_profile_returns_false(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    assert not repository.delete(
        "missing"
    )


def test_clear_repository(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    repository.save(
        "agent-a",
        MetaLearningProfile(),
    )

    repository.save(
        "agent-b",
        MetaLearningProfile(),
    )

    assert repository.count() == 2

    repository.clear()

    assert repository.count() == 0


def test_multiple_agents_are_isolated(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    profile_a = MetaLearningProfile()

    profile_a.record_exploration(
        success=True,
        event_id="a-1",
    )

    profile_b = MetaLearningProfile()

    profile_b.record_exploitation(
        success=True,
        event_id="b-1",
    )

    repository.save(
        "agent-a",
        profile_a,
    )

    repository.save(
        "agent-b",
        profile_b,
    )

    loaded_a = repository.get(
        "agent-a"
    )

    loaded_b = repository.get(
        "agent-b"
    )

    assert loaded_a is not None
    assert loaded_b is not None

    assert loaded_a.exploration_count == 1
    assert loaded_a.exploitation_count == 0

    assert loaded_b.exploration_count == 0
    assert loaded_b.exploitation_count == 1


def test_invalid_database_path_is_rejected():
    with pytest.raises(
        ValueError,
        match="database_path cannot be None",
    ):
        MetaLearningRepository(None)


def test_invalid_agent_name_is_rejected(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        repository.get("")

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        repository.exists("")

    with pytest.raises(
        ValueError,
        match="agent_name cannot be empty",
    ):
        repository.delete("")


def test_invalid_profile_is_rejected(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    with pytest.raises(
        ValueError,
        match="profile cannot be None",
    ):
        repository.save(
            "agent-a",
            None,
        )


def test_learning_event_order_is_preserved(tmp_path):
    repository = MetaLearningRepository(
        tmp_path / "meta_learning.db"
    )

    profile = MetaLearningProfile()

    events = [
        "event-1",
        "event-2",
        "event-3",
    ]

    for event in events:
        profile.record_exploration(
            success=True,
            event_id=event,
        )

    repository.save(
        "agent-a",
        profile,
    )

    loaded = repository.get(
        "agent-a"
    )

    assert loaded is not None
    assert (
        loaded.learning_events
        == events
    )


def test_repository_is_persistent_across_instances(
    tmp_path,
):
    database = (
        Path(tmp_path)
        / "meta_learning.db"
    )

    repository_a = (
        MetaLearningRepository(
            database
        )
    )

    profile = MetaLearningProfile()

    profile.record_exploitation(
        success=True,
        event_id="persistent-1",
    )

    repository_a.save(
        "agent-a",
        profile,
    )

    repository_b = (
        MetaLearningRepository(
            database
        )
    )

    loaded = repository_b.get(
        "agent-a"
    )

    assert loaded is not None
    assert loaded.exploitation_count == 1
    assert (
        loaded.learning_events
        == ["persistent-1"]
    )
