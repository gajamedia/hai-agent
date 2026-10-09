from agent.memory import EpisodicMemory
from agent.memory.episodic_store import EpisodicMemoryStore


def test_episodic_memory_store():

    store = EpisodicMemoryStore()

    memory = EpisodicMemory(
        content="Agent opened the box.",

        goal="Open the door",

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

        cycle=1,
    )

    store.add(memory)

    assert store.count() == 1
    assert store.latest() == memory


def test_episodic_memory_store_multiple_memories():

    store = EpisodicMemoryStore()

    first = EpisodicMemory(
        content="Opened box.",
        action="open_box",
        cycle=1,
    )

    second = EpisodicMemory(
        content="Took key.",
        action="take_key",
        cycle=2,
    )

    store.add(first)
    store.add(second)

    memories = store.all()

    assert len(memories) == 2
    assert memories[0].action == "open_box"
    assert memories[1].action == "take_key"


def test_episodic_memory_store_clear():

    store = EpisodicMemoryStore()

    store.add(
        EpisodicMemory(
            content="Test episode."
        )
    )

    assert store.count() == 1

    store.clear()

    assert store.count() == 0
    assert store.latest() is None
