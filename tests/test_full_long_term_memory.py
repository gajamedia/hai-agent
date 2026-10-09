from agent.core.action import Action
from agent.core.agent import Agent


def create_actions():

    return [
        Action(
            name="open_box",
            description="Open the box",
            handler=lambda: {
                "success": True,
                "message": "Box opened.",
            },
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),
        Action(
            name="take_key",
            description="Take the key",
            handler=lambda: {
                "success": True,
                "message": "Key taken.",
            },
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock the door",
            handler=lambda: {
                "success": False,
                "message": "The agent does not have the key.",
            },
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={
                "door": "unlocked",
            },
        ),
    ]


def test_all_long_term_memory_survives_agent_restart(
    tmp_path,
):

    database_path = (
        tmp_path / "agent_memory.db"
    )

    # =================================================
    # SESSION 1
    # =================================================

    agent_1 = Agent(
        "hAI",
        memory_path=database_path,
    )

    agent_1.register_actions(
        create_actions()
    )

    # -------------------------------------------------
    # Successful experience
    # -------------------------------------------------

    experience_1 = (
        agent_1.start_experience(
            experience_id="experience-001",
            goal="get the key",
        )
    )

    agent_1.record_experience(
        action="open_box",
        observation_before={
            "box": "closed",
            "key": "inside_box",
        },
        observation_after={
            "box": "open",
            "key": "inside_box",
        },
        result={
            "success": True,
        },
    )

    agent_1.record_experience(
        action="take_key",
        observation_before={
            "box": "open",
            "key": "inside_box",
        },
        observation_after={
            "box": "open",
            "key": "agent",
        },
        result={
            "success": True,
        },
    )

    agent_1.complete_experience()

    assert experience_1.completed is True

    # -------------------------------------------------
    # Failed experience
    # -------------------------------------------------

    experience_2 = (
        agent_1.start_experience(
            experience_id="experience-002",
            goal="unlock the door",
        )
    )

    agent_1.record_experience(
        action="unlock_door",
        observation_before={
            "door": "locked",
            "key": "missing",
        },
        observation_after={
            "door": "locked",
            "key": "missing",
        },
        result={
            "success": False,
            "message": (
                "The agent does not have the key."
            ),
        },
    )

    agent_1.fail_experience()

    assert experience_2.failed is True

    # -------------------------------------------------
    # Verify Session 1
    # -------------------------------------------------

    counts_1 = agent_1.memory_counts()

    assert counts_1 == {
        "episodic": 3,
        "semantic": 2,
        "causal": 1,
        "reflection": 1,
        "procedural": 1,
        "self_model": 1,
    }

    assert (
        agent_1.procedural_memory.count()
        == 1
    )

    self_model_1 = (
        agent_1.self_model_store.get()
    )

    assert (
        self_model_1.get_capability(
            "open_box"
        )
        is not None
    )

    assert (
        self_model_1.get_capability(
            "take_key"
        )
        is not None
    )

    assert (
        self_model_1.get_capability(
            "unlock_door"
        )
        is not None
    )

    del agent_1

    # =================================================
    # SESSION 2
    # =================================================

    agent_2 = Agent(
        "hAI",
        memory_path=database_path,
    )

    agent_2.register_actions(
        create_actions()
    )

    # -------------------------------------------------
    # All memory counts restored
    # -------------------------------------------------

    counts_2 = agent_2.memory_counts()

    assert counts_2 == {
        "episodic": 3,
        "semantic": 2,
        "causal": 1,
        "reflection": 1,
        "procedural": 1,
        "self_model": 1,
    }

    # -------------------------------------------------
    # Episodic
    # -------------------------------------------------

    episodes = (
        agent_2.episodic_memory.all()
    )

    assert len(episodes) == 3

    assert episodes[0].action == (
        "open_box"
    )

    assert episodes[1].action == (
        "take_key"
    )

    assert episodes[2].action == (
        "unlock_door"
    )

    # -------------------------------------------------
    # Semantic
    # -------------------------------------------------

    semantic = (
        agent_2.semantic_memory.find(
            subject="box",
            predicate="changes_from",
        )
    )

    assert len(semantic) == 1

    assert semantic[0].object == {
        "before": "closed",
        "after": "open",
    }

    key_semantic = (
        agent_2.semantic_memory.find(
            subject="key",
            predicate="changes_from",
        )
    )

    assert len(key_semantic) == 1

    assert key_semantic[0].object == {
        "before": "inside_box",
        "after": "agent",
    }

    # -------------------------------------------------
    # Causal
    # -------------------------------------------------

    causal = (
        agent_2.causal_memory.all()
    )

    assert len(causal) == 1

    assert causal[0].cause == (
        "open_box"
    )

    assert causal[0].relation == (
        "enables"
    )

    assert causal[0].effect == (
        "take_key"
    )

    # -------------------------------------------------
    # Reflection
    # -------------------------------------------------

    reflections = (
        agent_2.reflection_memory.find_by_action(
            "unlock_door"
        )
    )

    assert len(reflections) == 1

    reflection = reflections[0]

    assert reflection.action == (
        "unlock_door"
    )

    assert reflection.applied is False

    # -------------------------------------------------
    # Procedural
    # -------------------------------------------------

    procedures = (
        agent_2.procedural_memory.all()
    )

    assert len(procedures) == 1

    procedure = procedures[0]

    assert procedure.skill == (
        "get_the_key"
    )

    assert procedure.steps == [
        "open_box",
        "take_key",
    ]

    assert procedure.success_count == 1

    # -------------------------------------------------
    # Self Model
    # -------------------------------------------------

    self_model = (
        agent_2.self_model_store.get()
    )

    open_box_capability = (
        self_model.get_capability(
            "open_box"
        )
    )

    assert open_box_capability is not None

    assert (
        open_box_capability.success_count
        == 1
    )

    take_key_capability = (
        self_model.get_capability(
            "take_key"
        )
    )

    assert take_key_capability is not None

    assert (
        take_key_capability.success_count
        == 1
    )

    unlock_capability = (
        self_model.get_capability(
            "unlock_door"
        )
    )

    assert unlock_capability is not None

    assert (
        unlock_capability.failure_count
        == 1
    )

    assert (
        "Failed goal: unlock the door"
        in self_model.known_failures
    )
