from agent.core.agent import Agent
from agent.core.goal import Goal


def test_agent_can_use_persistent_memory(
    tmp_path,
):
    database_path = (
        tmp_path / "agent.db"
    )

    agent = Agent(
        "hAI",
        memory_path=database_path,
    )

    assert agent.is_persistent is True

    assert agent.memory_counts() == {
        "episodic": 0,
        "semantic": 0,
        "causal": 0,
        "reflection": 0,
        "procedural": 0,
        "self_model": 0,
    }


def test_agent_record_experience_is_persistent(
    tmp_path,
):
    database_path = (
        tmp_path / "agent.db"
    )

    agent = Agent(
        "hAI",
        memory_path=database_path,
    )

    agent.set_goal(
        Goal("open the door")
    )

    episode = agent.record_experience(
        action="open_box",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "open",
        },
        result={
            "success": True,
            "message": (
                "The box is now open."
            ),
        },
    )

    assert episode.action == (
        "open_box"
    )

    assert agent.memory_counts() == {
        "episodic": 1,
        "semantic": 1,
        "causal": 0,
        "reflection": 0,
        "procedural": 0,
        "self_model": 0,
    }


def test_agent_memory_survives_restart(
    tmp_path,
):
    database_path = (
        tmp_path / "agent.db"
    )

    # =============================================
    # SESSION 1
    # =============================================

    agent_1 = Agent(
        "hAI",
        memory_path=database_path,
    )

    agent_1.set_goal(
        Goal("open the door")
    )

    agent_1.record_experience(
        action="open_box",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "open",
        },
        result={
            "success": True,
        },
    )

    assert agent_1.memory_counts() == {
        "episodic": 1,
        "semantic": 1,
        "causal": 0,
        "reflection": 0,
        "procedural": 0,
        "self_model": 0,
    }

    del agent_1

    # =============================================
    # SESSION 2
    # =============================================

    agent_2 = Agent(
        "hAI",
        memory_path=database_path,
    )

    assert agent_2.is_persistent is True

    assert agent_2.memory_counts() == {
        "episodic": 1,
        "semantic": 1,
        "causal": 0,
        "reflection": 0,
        "procedural": 0,
        "self_model": 0,
    }

    episode = (
        agent_2.episodic_memory.latest()
    )

    assert episode is not None

    assert episode.action == (
        "open_box"
    )

    semantic_memories = (
        agent_2.semantic_memory.find(
            subject="box",
            predicate="changes_from",
        )
    )

    assert len(
        semantic_memories
    ) == 1

    semantic = semantic_memories[0]

    assert semantic.subject == "box"

    assert semantic.predicate == (
        "changes_from"
    )

    assert semantic.object == {
        "before": "closed",
        "after": "open",
    }


def test_agent_without_persistence_keeps_memory_in_ram():
    agent = Agent("hAI")

    assert agent.is_persistent is False

    agent.set_goal(
        Goal("open the door")
    )

    agent.record_experience(
        action="open_box",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "open",
        },
        result={
            "success": True,
        },
    )

    assert agent.memory_counts() == {
        "episodic": 1,
        "semantic": 1,
        "causal": 0,
        "reflection": 0,
        "procedural": 0,
        "self_model": 0,
    }
