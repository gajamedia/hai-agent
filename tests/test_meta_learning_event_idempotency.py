from agent.learning.meta_learning import MetaLearningProfile


def test_duplicate_exploration_event_is_counted_once():
    profile = MetaLearningProfile()

    profile.record_exploration(True, event_id="decision-1")
    profile.record_exploration(False, event_id="decision-1")

    assert profile.exploration_count == 1
    assert profile.exploration_success_count == 1
    assert profile.exploration_failure_count == 0
    assert profile.total_learning_events == 1


def test_duplicate_exploitation_event_is_counted_once():
    profile = MetaLearningProfile()

    profile.record_exploitation(False, event_id="decision-2")
    profile.record_exploitation(True, event_id="decision-2")

    assert profile.exploitation_count == 1
    assert profile.exploitation_failure_count == 1
    assert profile.exploitation_success_count == 0


def test_duplicate_evolution_and_recovery_events_are_counted_once():
    profile = MetaLearningProfile()

    profile.record_evolution(False, event_id="evolution-1")
    profile.record_evolution(True, event_id="evolution-1")
    profile.record_recovery(True, event_id="recovery-1")
    profile.record_recovery(False, event_id="recovery-1")

    assert profile.evolution_count == 1
    assert profile.evolution_failure_count == 1
    assert profile.evolution_success_count == 0
    assert profile.recovery_count == 1
    assert profile.recovery_success_count == 1
    assert profile.recovery_failure_count == 0
    assert profile.total_adaptation_events == 2


def test_duplicate_progress_event_does_not_double_count_progress():
    profile = MetaLearningProfile()

    profile.record_progress(0.4, event_id="progress-1")
    profile.record_progress(0.9, event_id="progress-1")

    assert profile.total_progress == 0.4
    assert profile.progress_observation_count == 1
    assert profile.average_progress == 0.4


def test_calls_without_event_id_preserve_existing_behavior():
    profile = MetaLearningProfile()

    profile.record_exploration(True)
    profile.record_exploration(False)
    profile.record_progress(0.2)
    profile.record_progress(0.3)

    assert profile.exploration_count == 2
    assert profile.total_progress == 0.5
    assert profile.progress_observation_count == 2


def test_different_event_ids_are_counted_independently():
    profile = MetaLearningProfile()

    profile.record_exploration(True, event_id="decision-a")
    profile.record_exploration(False, event_id="decision-b")

    assert profile.exploration_count == 2
    assert profile.exploration_success_count == 1
    assert profile.exploration_failure_count == 1
    assert profile.total_learning_events == 2
