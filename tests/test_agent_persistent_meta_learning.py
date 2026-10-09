from agent.core.agent import Agent


def test_agent_without_memory_path_uses_in_memory_meta_learning():
    agent = Agent(
        name="agent-memory-test"
    )

    assert agent.memory_path is None
    assert agent.persistent_memory is None

    assert agent.meta_learning_repository is None

    assert agent.meta_learning_profile is not None


def test_agent_with_memory_path_uses_persistent_meta_learning(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-persistent",
        memory_path=database,
    )

    assert agent.memory_path == database
    assert agent.persistent_memory is not None

    assert (
        agent.meta_learning_repository
        is not None
    )

    assert (
        agent.meta_learning_store
        .repository
        is agent.meta_learning_repository
    )


def test_meta_learning_profile_is_persisted_after_exploration(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-exploration",
        memory_path=database,
    )

    agent._record_meta_exploration(
        success=True
    )

    assert (
        agent.meta_learning_profile
        .exploration_count
        == 1
    )

    repository = (
        agent.meta_learning_repository
    )

    loaded = repository.get(
        "agent-exploration"
    )

    assert loaded is not None

    assert (
        loaded.exploration_count
        == 1
    )

    assert (
        loaded.exploration_success_count
        == 1
    )


def test_meta_learning_profile_is_persisted_after_exploitation(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-exploitation",
        memory_path=database,
    )

    agent._record_meta_exploitation(
        success=True
    )

    loaded = (
        agent.meta_learning_repository.get(
            "agent-exploitation"
        )
    )

    assert loaded is not None

    assert (
        loaded.exploitation_count
        == 1
    )

    assert (
        loaded.exploitation_success_count
        == 1
    )


def test_meta_learning_profile_is_persisted_after_evolution(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-evolution",
        memory_path=database,
    )

    agent._record_meta_evolution(
        success=True
    )

    loaded = (
        agent.meta_learning_repository.get(
            "agent-evolution"
        )
    )

    assert loaded is not None

    assert (
        loaded.evolution_count
        == 1
    )

    assert (
        loaded.evolution_success_count
        == 1
    )


def test_meta_learning_profile_is_persisted_after_recovery(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-recovery",
        memory_path=database,
    )

    agent._record_meta_recovery(
        success=False
    )

    loaded = (
        agent.meta_learning_repository.get(
            "agent-recovery"
        )
    )

    assert loaded is not None

    assert (
        loaded.recovery_count
        == 1
    )

    assert (
        loaded.recovery_failure_count
        == 1
    )


def test_meta_learning_profile_is_persisted_after_progress(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-progress",
        memory_path=database,
    )

    agent._record_meta_progress(
        progress_delta=0.5,
        event_id="progress-1",
    )

    loaded = (
        agent.meta_learning_repository.get(
            "agent-progress"
        )
    )

    assert loaded is not None

    assert (
        loaded.total_progress
        == 0.5
    )

    assert (
        loaded.progress_observation_count
        == 1
    )

    assert (
        loaded.learning_events
        == ["progress-1"]
    )


def test_new_agent_restores_meta_learning_profile(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    first_agent = Agent(
        name="agent-restart",
        memory_path=database,
    )

    first_agent._record_meta_exploration(
        success=True
    )

    first_agent._record_meta_exploration(
        success=False
    )

    first_agent._record_meta_progress(
        progress_delta=0.75,
        event_id="restart-progress",
    )

    second_agent = Agent(
        name="agent-restart",
        memory_path=database,
    )

    profile = (
        second_agent.get_meta_learning_profile()
    )

    assert (
        profile.exploration_count
        == 2
    )

    assert (
        profile.exploration_success_count
        == 1
    )

    assert (
        profile.exploration_failure_count
        == 1
    )

    assert (
        profile.total_progress
        == 0.75
    )

    assert (
        profile.learning_events
        == [
            "agent-restart:meta:exploration:1",
            "agent-restart:meta:exploration:2",
            "restart-progress",
        ]
    )


def test_meta_learning_runtime_uses_restored_profile(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    first_agent = Agent(
        name="agent-runtime-restart",
        memory_path=database,
    )

    first_agent._record_meta_exploitation(
        success=True
    )

    first_agent._record_meta_exploitation(
        success=True
    )

    first_agent._record_meta_exploitation(
        success=True
    )

    second_agent = Agent(
        name="agent-runtime-restart",
        memory_path=database,
    )

    assert (
        second_agent.meta_learning_profile
        .exploitation_count
        == 3
    )

    runtime = (
        second_agent.evaluate_meta_learning_runtime(
            uncertainty=0.1,
            information_gain=0.1,
            risk=0.1,
        )
    )

    assert runtime is not None
    assert runtime.decision is not None

    assert (
        second_agent.meta_learning_profile
        .exploitation_success_rate
        == 1.0
    )


def test_meta_learning_profiles_are_isolated_by_agent_name(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent_a = Agent(
        name="agent-a",
        memory_path=database,
    )

    agent_b = Agent(
        name="agent-b",
        memory_path=database,
    )

    agent_a._record_meta_exploration(
        success=True
    )

    agent_b._record_meta_exploitation(
        success=True
    )

    profile_a = (
        agent_a.meta_learning_repository.get(
            "agent-a"
        )
    )

    profile_b = (
        agent_b.meta_learning_repository.get(
            "agent-b"
        )
    )

    assert profile_a is not None
    assert profile_b is not None

    assert (
        profile_a.exploration_count
        == 1
    )

    assert (
        profile_a.exploitation_count
        == 0
    )

    assert (
        profile_b.exploration_count
        == 0
    )

    assert (
        profile_b.exploitation_count
        == 1
    )


def test_meta_learning_snapshot_survives_restart(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-snapshot",
        memory_path=database,
    )

    agent._record_meta_exploration(
        success=True
    )

    agent._record_meta_exploitation(
        success=False
    )

    before = (
        agent.get_meta_learning_snapshot()
    )

    restored_agent = Agent(
        name="agent-snapshot",
        memory_path=database,
    )

    after = (
        restored_agent.get_meta_learning_snapshot()
    )

    assert after == before
