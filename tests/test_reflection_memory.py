from agent.memory.reflection import ReflectionMemory
from agent.memory.reflection_store import (
    ReflectionMemoryStore,
)


def test_reflection_memory_store():

    store = ReflectionMemoryStore()

    memory = ReflectionMemory(
        content="Failed action reflection.",
        action="take_key",
        problem="Key unavailable.",
        cause="key was missing.",
        lesson="Check key availability.",
        confidence=0.7,
    )

    store.add(memory)

    assert store.count() == 1
    assert store.latest() is memory


def test_reflection_store_find_by_action():

    store = ReflectionMemoryStore()

    first = ReflectionMemory(
        content="Reflection one.",
        action="take_key",
    )

    second = ReflectionMemory(
        content="Reflection two.",
        action="unlock_door",
    )

    store.add(first)
    store.add(second)

    results = store.find_by_action(
        "take_key"
    )

    assert len(results) == 1
    assert results[0] is first
