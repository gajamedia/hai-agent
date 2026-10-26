from agent.core.agent import Agent


def test_agent_restores_recorded_episodes_after_restart(tmp_path):
    database_path = tmp_path / "agent_memory.db"

    first_agent = Agent(
        name="cross-session-agent",
        memory_path=database_path,
    )
    first_agent.record_experience(
        action="take_key",
        observation_before={
            "door": "locked",
            "key": "inside_box",
        },
        observation_after={
            "door": "locked",
            "key": "inside_box",
        },
        result={"success": False},
    )

    assert first_agent.episodic_memory.count() == 1
    assert first_agent.persistent_memory.episodic.count() == 1

    del first_agent

    second_agent = Agent(
        name="cross-session-agent",
        memory_path=database_path,
    )

    assert second_agent.episodic_memory.count() == 1

    episode = second_agent.episodic_memory.latest()
    assert episode.action == "take_key"
    assert episode.result["success"] is False
