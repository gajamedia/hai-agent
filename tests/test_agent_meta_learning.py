from agent.core.agent import Agent


def test_agent_creates_meta_learning_profile():
    agent = Agent("meta-agent")

    profile = agent.get_meta_learning_profile()

    assert profile is not None
    assert (
        agent.meta_learning_store.get("meta-agent")
        is profile
    )


def test_agent_meta_learning_snapshot_starts_empty():
    agent = Agent("meta-agent")

    snapshot = agent.get_meta_learning_snapshot()

    assert snapshot["exploration_count"] == 0
    assert snapshot["exploration_success_rate"] == 0.0

    assert snapshot["exploitation_count"] == 0
    assert snapshot["exploitation_success_rate"] == 0.0

    assert snapshot["evolution_count"] == 0
    assert snapshot["evolution_success_rate"] == 0.0

    assert snapshot["recovery_count"] == 0
    assert snapshot["recovery_success_rate"] == 0.0

    assert snapshot["average_progress"] == 0.0
    assert snapshot["total_decisions"] == 0
    assert snapshot["total_adaptation_events"] == 0
    assert snapshot["total_learning_events"] == 0
    assert snapshot["overall_adaptation_success_rate"] == 0.0


def test_agent_records_exploration():
    agent = Agent("meta-agent")

    agent._record_meta_exploration(
        success=True
    )

    profile = agent.get_meta_learning_profile()

    assert profile.exploration_count == 1
    assert profile.exploration_success_count == 1
    assert profile.exploration_failure_count == 0

    assert profile.total_decisions == 1
    assert profile.total_learning_events == 1


def test_agent_records_exploitation():
    agent = Agent("meta-agent")

    agent._record_meta_exploitation(
        success=False
    )

    profile = agent.get_meta_learning_profile()

    assert profile.exploitation_count == 1
    assert profile.exploitation_success_count == 0
    assert profile.exploitation_failure_count == 1

    assert profile.total_decisions == 1
    assert profile.total_learning_events == 1


def test_agent_records_evolution():
    agent = Agent("meta-agent")

    agent._record_meta_evolution(
        success=True
    )

    agent._record_meta_evolution(
        success=False
    )

    profile = agent.get_meta_learning_profile()

    assert profile.evolution_count == 2
    assert profile.evolution_success_count == 1
    assert profile.evolution_failure_count == 1

    assert profile.total_adaptation_events == 2
    assert profile.total_learning_events == 2


def test_agent_records_recovery():
    agent = Agent("meta-agent")

    agent._record_meta_recovery(
        success=True
    )

    agent._record_meta_recovery(
        success=False
    )

    profile = agent.get_meta_learning_profile()

    assert profile.recovery_count == 2
    assert profile.recovery_success_count == 1
    assert profile.recovery_failure_count == 1

    assert profile.total_adaptation_events == 2
    assert profile.total_learning_events == 2


def test_agent_records_progress():
    agent = Agent("meta-agent")

    agent._record_meta_progress(
        progress_delta=0.5,
        event_id="experience-001",
    )

    profile = agent.get_meta_learning_profile()

    assert profile.total_progress == 0.5
    assert profile.progress_observation_count == 1
    assert profile.average_progress == 0.5

    assert "experience-001" in profile.learning_events
    assert profile.total_learning_events == 1


def test_agent_progress_event_is_not_duplicated():
    agent = Agent("meta-agent")

    agent._record_meta_progress(
        progress_delta=0.5,
        event_id="experience-001",
    )

    agent._record_meta_progress(
        progress_delta=0.5,
        event_id="experience-001",
    )

    profile = agent.get_meta_learning_profile()

    # Replayed event IDs must not update the learning profile twice.
    assert profile.progress_observation_count == 1
    assert profile.total_progress == 0.5

    assert (
        profile.learning_events.count(
            "experience-001"
        )
        == 1
    )


def test_agent_generates_unique_meta_event_ids():
    agent = Agent("meta-agent")

    event_1 = (
        agent._next_meta_learning_event_id(
            "test"
        )
    )

    agent._record_meta_progress(
        progress_delta=0.1,
        event_id=event_1,
    )

    event_2 = (
        agent._next_meta_learning_event_id(
            "test"
        )
    )

    assert event_1 != event_2


def test_agent_meta_learning_decision_recording():
    agent = Agent("meta-agent")

    agent._record_meta_exploration(
        success=True
    )

    agent._record_meta_exploitation(
        success=False
    )

    profile = agent.get_meta_learning_profile()

    assert profile.exploration_count == 1
    assert profile.exploitation_count == 1

    assert profile.exploration_success_count == 1
    assert profile.exploitation_failure_count == 1

    assert profile.total_decisions == 2


def test_agent_meta_learning_profile_accumulates_all_event_types():
    agent = Agent("meta-agent")

    agent._record_meta_exploration(
        success=True
    )

    agent._record_meta_exploitation(
        success=True
    )

    agent._record_meta_evolution(
        success=True
    )

    agent._record_meta_recovery(
        success=True
    )

    agent._record_meta_progress(
        progress_delta=0.25,
        event_id="progress-001",
    )

    profile = agent.get_meta_learning_profile()

    assert profile.exploration_count == 1
    assert profile.exploitation_count == 1
    assert profile.evolution_count == 1
    assert profile.recovery_count == 1

    assert profile.total_decisions == 2
    assert profile.total_adaptation_events == 2

    assert profile.progress_observation_count == 1
    assert profile.average_progress == 0.25

    assert profile.total_learning_events == 5
