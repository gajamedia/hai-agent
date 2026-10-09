from agent.learning.meta_learning import (
    MetaLearningProfile,
)
from agent.learning.meta_learning_attribution import (
    MetaLearningEventAttributor,
)


def test_record_exploration_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_exploration(
        profile=profile,
        success=True,
        event_id="explore-1",
    )

    assert profile.exploration_count == 1
    assert profile.exploration_success_count == 1
    assert profile.exploration_failure_count == 0

    assert profile.learning_events == [
        "explore-1"
    ]


def test_record_exploitation_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_exploitation(
        profile=profile,
        success=False,
        event_id="exploit-1",
    )

    assert profile.exploitation_count == 1
    assert profile.exploitation_success_count == 0
    assert profile.exploitation_failure_count == 1


def test_record_evolution_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_evolution(
        profile=profile,
        success=True,
        event_id="evolution-1",
    )

    assert profile.evolution_count == 1
    assert profile.evolution_success_count == 1


def test_record_recovery_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_recovery(
        profile=profile,
        success=True,
        event_id="recovery-1",
    )

    assert profile.recovery_count == 1
    assert profile.recovery_success_count == 1


def test_record_progress_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_progress(
        profile=profile,
        progress_delta=0.75,
        event_id="progress-1",
    )

    assert profile.progress_observation_count == 1
    assert profile.total_progress == 0.75


def test_generic_exploration_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_event(
        profile=profile,
        event_type="exploration",
        success=True,
        event_id="event-1",
    )

    assert profile.exploration_count == 1


def test_generic_exploitation_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_event(
        profile=profile,
        event_type=" exploitation ",
        success=False,
        event_id="event-2",
    )

    assert profile.exploitation_count == 1
    assert profile.exploitation_failure_count == 1


def test_generic_evolution_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_event(
        profile=profile,
        event_type="EVOLUTION",
        success=True,
        event_id="event-3",
    )

    assert profile.evolution_count == 1


def test_generic_recovery_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_event(
        profile=profile,
        event_type="RECOVERY",
        success=False,
        event_id="event-4",
    )

    assert profile.recovery_count == 1
    assert profile.recovery_failure_count == 1


def test_generic_progress_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_event(
        profile=profile,
        event_type="progress",
        progress_delta=-0.25,
        event_id="event-5",
    )

    assert profile.progress_observation_count == 1
    assert profile.total_progress == -0.25


def test_multiple_event_types_update_same_profile():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_event(
        profile=profile,
        event_type="exploration",
        success=True,
        event_id="explore-1",
    )

    attributor.record_event(
        profile=profile,
        event_type="exploitation",
        success=True,
        event_id="exploit-1",
    )

    attributor.record_event(
        profile=profile,
        event_type="evolution",
        success=False,
        event_id="evolution-1",
    )

    attributor.record_event(
        profile=profile,
        event_type="recovery",
        success=True,
        event_id="recovery-1",
    )

    attributor.record_event(
        profile=profile,
        event_type="progress",
        progress_delta=0.5,
        event_id="progress-1",
    )

    assert profile.exploration_count == 1
    assert profile.exploitation_count == 1
    assert profile.evolution_count == 1
    assert profile.recovery_count == 1

    assert profile.progress_observation_count == 1

    assert profile.total_learning_events == 5


def test_duplicate_event_id_remains_single_event():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    attributor.record_exploration(
        profile=profile,
        success=True,
        event_id="same-event",
    )

    attributor.record_evolution(
        profile=profile,
        success=True,
        event_id="same-event",
    )

    assert profile.learning_events == [
        "same-event"
    ]

    assert profile.total_learning_events == 1


def test_missing_success_is_rejected():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    try:
        attributor.record_event(
            profile=profile,
            event_type="exploration",
            event_id="event-1",
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "success is required for exploration."
        )


def test_missing_progress_is_rejected():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    try:
        attributor.record_event(
            profile=profile,
            event_type="progress",
            event_id="event-1",
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "progress_delta is required for progress."
        )


def test_unsupported_event_type_is_rejected():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    try:
        attributor.record_event(
            profile=profile,
            event_type="unknown",
            success=True,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "unsupported event type: unknown."
        )


def test_none_profile_is_rejected():
    attributor = (
        MetaLearningEventAttributor()
    )

    try:
        attributor.record_exploration(
            profile=None,
            success=True,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "profile cannot be None."
        )


def test_empty_event_type_is_rejected():
    profile = MetaLearningProfile()

    attributor = (
        MetaLearningEventAttributor()
    )

    try:
        attributor.record_event(
            profile=profile,
            event_type="",
            success=True,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "event_type cannot be empty."
        )
