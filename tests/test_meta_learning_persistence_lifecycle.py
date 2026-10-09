from agent.core.agent import Agent


def test_explicit_meta_learning_save_persists_profile(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-save",
        memory_path=database,
    )

    agent.meta_learning_profile.record_exploration(
        success=True,
        event_id="manual-save-event",
    )

    # Bypass the automatic helper intentionally.
    # The lifecycle method must still persist it.
    agent.save_meta_learning()

    restored = Agent(
        name="agent-save",
        memory_path=database,
    )

    assert (
        restored.meta_learning_profile
        .exploration_count
        == 1
    )

    assert (
        restored.meta_learning_profile
        .learning_events
        == [
            "manual-save-event"
        ]
    )


def test_non_persistent_agent_save_is_safe():
    agent = Agent(
        name="agent-memory-only",
    )

    agent.meta_learning_profile.record_exploration(
        success=True,
        event_id="memory-event",
    )

    agent.save_meta_learning()

    assert (
        agent.meta_learning_profile
        .exploration_count
        == 1
    )


def test_meta_learning_continues_after_restart(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    # -------------------------------------------------
    # Process 1
    # -------------------------------------------------

    first = Agent(
        name="agent-continuity",
        memory_path=database,
    )

    first._record_meta_exploration(
        success=True
    )

    # -------------------------------------------------
    # Process 2
    # -------------------------------------------------

    second = Agent(
        name="agent-continuity",
        memory_path=database,
    )

    assert (
        second.meta_learning_profile
        .exploration_count
        == 1
    )

    second._record_meta_exploration(
        success=True
    )

    # -------------------------------------------------
    # Process 3
    # -------------------------------------------------

    third = Agent(
        name="agent-continuity",
        memory_path=database,
    )

    assert (
        third.meta_learning_profile
        .exploration_count
        == 2
    )

    assert (
        third.meta_learning_profile
        .exploration_success_count
        == 2
    )

    assert (
        third.meta_learning_profile
        .learning_events
        == [
            "agent-continuity:meta:exploration:1",
            "agent-continuity:meta:exploration:2",
        ]
    )


def test_meta_learning_progress_continues_after_restart(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    first = Agent(
        name="agent-progress-continuity",
        memory_path=database,
    )

    first._record_meta_progress(
        progress_delta=0.25,
        event_id="progress-1",
    )

    second = Agent(
        name="agent-progress-continuity",
        memory_path=database,
    )

    assert (
        second.meta_learning_profile
        .total_progress
        == 0.25
    )

    assert (
        second.meta_learning_profile
        .progress_observation_count
        == 1
    )

    second._record_meta_progress(
        progress_delta=0.75,
        event_id="progress-2",
    )

    third = Agent(
        name="agent-progress-continuity",
        memory_path=database,
    )

    assert (
        third.meta_learning_profile
        .total_progress
        == 1.0
    )

    assert (
        third.meta_learning_profile
        .progress_observation_count
        == 2
    )

    assert (
        third.meta_learning_profile
        .average_progress
        == 0.5
    )


def test_meta_learning_runtime_uses_continued_evidence(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    first = Agent(
        name="agent-runtime-continuity",
        memory_path=database,
    )

    first._record_meta_exploitation(
        success=True
    )

    first._record_meta_exploitation(
        success=True
    )

    first._record_meta_exploitation(
        success=True
    )

    second = Agent(
        name="agent-runtime-continuity",
        memory_path=database,
    )

    profile = (
        second.get_meta_learning_profile()
    )

    assert (
        profile.exploitation_count
        == 3
    )

    assert (
        profile.exploitation_success_rate
        == 1.0
    )

    runtime = (
        second.evaluate_meta_learning_runtime(
            uncertainty=0.1,
            information_gain=0.1,
            risk=0.1,
        )
    )

    assert runtime is not None

    assert (
        runtime.decision
        is not None
    )

    assert (
        runtime.decision.base_decision
        is not None
    )


def test_other_agent_profile_is_not_affected_by_restart(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    first = Agent(
        name="agent-a",
        memory_path=database,
    )

    first._record_meta_exploration(
        success=True
    )

    first._record_meta_exploitation(
        success=False
    )

    second = Agent(
        name="agent-b",
        memory_path=database,
    )

    second._record_meta_recovery(
        success=True
    )

    restored_a = Agent(
        name="agent-a",
        memory_path=database,
    )

    restored_b = Agent(
        name="agent-b",
        memory_path=database,
    )

    assert (
        restored_a.meta_learning_profile
        .exploration_count
        == 1
    )

    assert (
        restored_a.meta_learning_profile
        .exploitation_count
        == 1
    )

    assert (
        restored_a.meta_learning_profile
        .recovery_count
        == 0
    )

    assert (
        restored_b.meta_learning_profile
        .exploration_count
        == 0
    )

    assert (
        restored_b.meta_learning_profile
        .exploitation_count
        == 0
    )

    assert (
        restored_b.meta_learning_profile
        .recovery_count
        == 1
    )


def test_meta_learning_profile_survives_regular_memory_operations(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-memory-lifecycle",
        memory_path=database,
    )

    agent._record_meta_exploration(
        success=True
    )

    agent.clear_memory()

    restored = Agent(
        name="agent-memory-lifecycle",
        memory_path=database,
    )

    assert (
        restored.meta_learning_profile
        .exploration_count
        == 1
    )
