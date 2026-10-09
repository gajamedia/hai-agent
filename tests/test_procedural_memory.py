from agent.memory.procedural import (
    ProceduralMemory,
)
from agent.memory.procedural_store import (
    ProceduralMemoryStore,
)


def test_procedural_memory_success():

    memory = ProceduralMemory(
        content="Open door procedure.",
        skill="open_the_door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        confidence=0.5,
    )

    memory.record_success(
        episode_ids=[
            "episode-1",
            "episode-2",
            "episode-3",
        ]
    )

    assert memory.success_count == 1
    assert memory.failure_count == 0
    assert memory.success_rate == 1.0
    assert memory.confidence == 0.6
    assert len(memory.source_episodes) == 3


def test_procedural_memory_failure():

    memory = ProceduralMemory(
        content="Open door procedure.",
        skill="open_the_door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        confidence=0.8,
    )

    memory.record_failure(
        episode_ids=[
            "episode-4",
            "episode-5",
        ]
    )

    assert memory.success_count == 0
    assert memory.failure_count == 1
    assert memory.success_rate == 0.0
    assert memory.confidence == 0.7


def test_procedural_store_reuses_procedure():

    store = ProceduralMemoryStore()

    steps = [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    first = store.get_or_create(
        skill="open_the_door",
        steps=steps,
    )

    second = store.get_or_create(
        skill="open_the_door",
        steps=steps,
    )

    assert first is second
    assert store.count() == 1
