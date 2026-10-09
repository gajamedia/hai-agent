from agent.core.action import Action
from agent.learning.reflection import (
    ReflectionEngine,
)
from agent.memory.episodic import EpisodicMemory
from agent.memory.reflection_store import (
    ReflectionMemoryStore,
)


def test_reflection_engine_creates_reflection():

    action = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {},
        preconditions={
            "box": "open",
            "key": "inside_box",
        },
    )

    episode = EpisodicMemory(
        content="Failed to take key.",
        goal="Open the door",
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

    store = ReflectionMemoryStore()

    engine = ReflectionEngine(
        reflection_store=store
    )

    memory = engine.reflect(
        episode=episode,
        action=action,
    )

    assert memory is not None
    assert store.count() == 1

    assert memory.action == "take_key"
    assert memory.goal == "Open the door"

    assert memory.condition == {
        "key": "box",
        "expected": "open",
        "actual": "closed",
    }

    assert memory.applied is False


def test_reflection_engine_ignores_success():

    action = Action(
        name="open_box",
        description="Open box",
        handler=lambda: {},
    )

    episode = EpisodicMemory(
        content="Opened box.",
        action="open_box",
        result={
            "success": True,
        },
    )

    store = ReflectionMemoryStore()

    engine = ReflectionEngine(
        reflection_store=store
    )

    memory = engine.reflect(
        episode=episode,
        action=action,
    )

    assert memory is None
    assert store.count() == 0
