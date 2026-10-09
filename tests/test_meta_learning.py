from agent.learning.meta_learning import (
    MetaLearningProfile,
    MetaLearningStore,
)


def test_new_profile_starts_empty():
    profile = MetaLearningProfile()

    assert profile.exploration_count == 0
    assert profile.exploitation_count == 0
    assert profile.evolution_count == 0
    assert profile.recovery_count == 0

    assert profile.total_progress == 0.0
    assert profile.progress_observation_count == 0

    assert profile.total_decisions == 0
    assert profile.total_adaptation_events == 0
    assert profile.total_learning_events == 0


def test_record_exploration_success():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="explore-1",
    )

    assert profile.exploration_count == 1
    assert profile.exploration_success_count == 1
    assert profile.exploration_failure_count == 0

    assert profile.exploration_success_rate == 1.0
    assert profile.learning_events == [
        "explore-1"
    ]


def test_record_exploration_failure():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=False,
        event_id="explore-1",
    )

    assert profile.exploration_count == 1
    assert profile.exploration_success_count == 0
    assert profile.exploration_failure_count == 1

    assert profile.exploration_success_rate == 0.0


def test_record_exploitation():
    profile = MetaLearningProfile()

    profile.record_exploitation(
        success=True,
        event_id="exploit-1",
    )

    profile.record_exploitation(
        success=False,
        event_id="exploit-2",
    )

    assert profile.exploitation_count == 2
    assert profile.exploitation_success_count == 1
    assert profile.exploitation_failure_count == 1

    assert profile.exploitation_success_rate == 0.5


def test_record_evolution():
    profile = MetaLearningProfile()

    profile.record_evolution(
        success=True,
        event_id="evolution-1",
    )

    profile.record_evolution(
        success=False,
        event_id="evolution-2",
    )

    assert profile.evolution_count == 2
    assert profile.evolution_success_count == 1
    assert profile.evolution_failure_count == 1

    assert profile.evolution_success_rate == 0.5


def test_record_recovery():
    profile = MetaLearningProfile()

    profile.record_recovery(
        success=True,
        event_id="recovery-1",
    )

    assert profile.recovery_count == 1
    assert profile.recovery_success_count == 1
    assert profile.recovery_failure_count == 0

    assert profile.recovery_success_rate == 1.0


def test_record_progress():
    profile = MetaLearningProfile()

    profile.record_progress(
        progress_delta=0.25,
        event_id="progress-1",
    )

    profile.record_progress(
        progress_delta=0.75,
        event_id="progress-2",
    )

    assert profile.total_progress == 1.0
    assert profile.progress_observation_count == 2
    assert profile.average_progress == 0.5


def test_duplicate_event_id_is_not_stored_twice():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="event-1",
    )

    profile.record_progress(
        progress_delta=0.5,
        event_id="event-1",
    )

    assert profile.learning_events == [
        "event-1"
    ]

    assert profile.total_learning_events == 1


def test_total_decisions():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
    )

    profile.record_exploitation(
        success=True,
    )

    profile.record_exploitation(
        success=False,
    )

    assert profile.total_decisions == 3


def test_total_adaptation_events():
    profile = MetaLearningProfile()

    profile.record_evolution(
        success=True,
    )

    profile.record_recovery(
        success=True,
    )

    profile.record_recovery(
        success=False,
    )

    assert profile.total_adaptation_events == 3


def test_overall_adaptation_success_rate():
    profile = MetaLearningProfile()

    profile.record_evolution(
        success=True,
    )

    profile.record_evolution(
        success=False,
    )

    profile.record_recovery(
        success=True,
    )

    assert (
        profile.overall_adaptation_success_rate
        == round(2 / 3, 6)
    )


def test_snapshot_contains_learning_metrics():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
    )

    profile.record_exploitation(
        success=False,
    )

    profile.record_evolution(
        success=True,
    )

    profile.record_recovery(
        success=True,
    )

    profile.record_progress(
        progress_delta=0.5,
    )

    snapshot = profile.snapshot()

    assert snapshot[
        "exploration_count"
    ] == 1

    assert snapshot[
        "exploration_success_rate"
    ] == 1.0

    assert snapshot[
        "exploitation_count"
    ] == 1

    assert snapshot[
        "exploitation_success_rate"
    ] == 0.0

    assert snapshot[
        "evolution_count"
    ] == 1

    assert snapshot[
        "evolution_success_rate"
    ] == 1.0

    assert snapshot[
        "recovery_count"
    ] == 1

    assert snapshot[
        "recovery_success_rate"
    ] == 1.0

    assert snapshot[
        "average_progress"
    ] == 0.5


def test_store_get_or_create():
    store = MetaLearningStore()

    first = store.get_or_create(
        "test-agent"
    )

    second = store.get_or_create(
        "test-agent"
    )

    assert first is second
    assert store.count() == 1


def test_store_add_returns_existing_profile():
    store = MetaLearningStore()

    first = MetaLearningProfile()
    second = MetaLearningProfile()

    result_first = store.add(
        "agent",
        first,
    )

    result_second = store.add(
        "agent",
        second,
    )

    assert result_first is first
    assert result_second is first
    assert store.count() == 1


def test_store_get():
    store = MetaLearningStore()

    profile = MetaLearningProfile()

    store.add(
        "agent",
        profile,
    )

    assert store.get(
        "agent"
    ) is profile


def test_store_remove():
    store = MetaLearningStore()

    profile = store.get_or_create(
        "agent"
    )

    removed = store.remove(
        "agent"
    )

    assert removed is profile
    assert store.count() == 0


def test_store_clear():
    store = MetaLearningStore()

    store.get_or_create(
        "agent"
    )

    store.get_or_create(
        "another-agent"
    )

    assert store.count() == 2

    store.clear()

    assert store.count() == 0


def test_none_profile_is_rejected():
    store = MetaLearningStore()

    try:
        store.add(
            "agent",
            None,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "profile cannot be None."
        )


def test_empty_agent_name_is_rejected():
    store = MetaLearningStore()

    try:
        store.add(
            "",
            MetaLearningProfile(),
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "agent_name cannot be empty."
        )
