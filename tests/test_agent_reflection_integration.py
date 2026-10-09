from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal


def test_agent_automatically_creates_reflection_on_failure():

    agent = Agent("hAI")

    action = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {
            "success": False,
            "message": "The box is closed.",
        },
        preconditions={
            "box": "open",
            "key": "inside_box",
        },
        effects={
            "key": "agent",
        },
    )

    agent.register_action(
        action
    )

    agent.set_goal(
        Goal("open the door")
    )

    episode = agent.record_experience(
        action="take_key",
        observation_before={
            "box": "closed",
            "key": "inside_box",
        },
        observation_after={
            "box": "closed",
            "key": "inside_box",
        },
        result={
            "success": False,
            "message": "The box is closed.",
        },
    )

    assert episode.action == "take_key"

    reflections = (
        agent.reflection_memory.find_by_action(
            "take_key"
        )
    )

    assert len(reflections) == 1

    reflection = reflections[0]

    assert reflection.action == "take_key"

    assert reflection.goal == (
        "open the door"
    )

    assert reflection.condition == {
        "key": "box",
        "expected": "open",
        "actual": "closed",
    }

    assert (
        "Before executing 'take_key'"
        in reflection.lesson
    )


def test_agent_does_not_create_reflection_for_success():

    agent = Agent("hAI")

    action = Action(
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
    )

    agent.register_action(
        action
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

    assert (
        agent.reflection_memory.count()
        == 0
    )


def test_agent_persists_reflection_after_failure(
    tmp_path,
):

    database_path = (
        tmp_path / "agent.db"
    )

    # -----------------------------------------
    # Session 1
    # -----------------------------------------

    agent_1 = Agent(
        "hAI",
        memory_path=database_path,
    )

    action = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {
            "success": False,
        },
        preconditions={
            "box": "open",
        },
        effects={
            "key": "agent",
        },
    )

    agent_1.register_action(
        action
    )

    agent_1.record_experience(
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

    assert (
        agent_1.reflection_memory.count()
        == 1
    )

    del agent_1

    # -----------------------------------------
    # Session 2
    # -----------------------------------------

    agent_2 = Agent(
        "hAI",
        memory_path=database_path,
    )

    assert (
        agent_2.reflection_memory.count()
        == 1
    )

    reflection = (
        agent_2.reflection_memory.latest()
    )

    assert reflection is not None

    assert reflection.action == (
        "take_key"
    )

    assert reflection.condition == {
        "key": "box",
        "expected": "open",
        "actual": "closed",
    }
