from agent.core.agent import Agent
from agent.core.action import Action


def create_actions():

    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda: {
                "success": True,
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
            description="Take key",
            handler=lambda: {
                "success": True,
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
            description="Unlock door",
            handler=lambda: {
                "success": True,
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


def test_agent_learns_procedure_from_experience():

    agent = Agent("hAI")

    agent.register_actions(
        create_actions()
    )

    experience = agent.start_experience(
        experience_id="experience-001",
        goal="open the door",
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

    agent.record_experience(
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

    agent.record_experience(
        action="unlock_door",
        observation_before={
            "door": "locked",
            "key": "agent",
        },
        observation_after={
            "door": "unlocked",
            "key": "agent",
        },
        result={
            "success": True,
        },
    )

    assert len(
        experience.episodes
    ) == 3

    agent.complete_experience()

    memories = (
        agent.procedural_memory.all()
    )

    assert len(memories) == 1

    procedure = memories[0]

    assert procedure.skill == (
        "open_the_door"
    )

    assert procedure.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    assert procedure.success_count == 1


def test_agent_updates_self_model():

    agent = Agent("hAI")

    agent.start_experience(
        experience_id="experience-002",
        goal="open the door",
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

    agent.complete_experience()

    self_model = (
        agent.self_model_store.get()
    )

    capability = (
        self_model.get_capability(
            "open_box"
        )
    )

    assert capability is not None

    assert capability.success_count == 1

    assert capability.failure_count == 0

    assert capability.confidence == 0.6


def test_agent_learns_failed_action():

    agent = Agent("hAI")

    agent.start_experience(
        experience_id="experience-003",
        goal="open the door",
    )

    agent.record_experience(
        action="take_key",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "closed",
        },
        result={
            "success": False,
        },
    )

    agent.fail_experience()

    self_model = (
        agent.self_model_store.get()
    )

    capability = (
        self_model.get_capability(
            "take_key"
        )
    )

    assert capability is not None

    assert capability.failure_count == 1

    assert capability.success_count == 0

    assert capability.confidence == 0.4

    assert (
        "Failed goal: open the door"
        in self_model.known_failures
    )


def test_experience_contains_recorded_episodes():

    agent = Agent("hAI")

    experience = agent.start_experience(
        experience_id="experience-004",
        goal="open the door",
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

    agent.record_experience(
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

    assert experience.action_count == 2

    assert experience.actions == [
        "open_box",
        "take_key",
    ]

    agent.complete_experience()

    assert experience.completed is True
    assert experience.failed is False
