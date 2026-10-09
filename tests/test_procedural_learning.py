from agent.learning.procedural_learning import (
    ProceduralLearning,
)
from agent.memory.episodic import EpisodicMemory
from agent.memory.procedural_store import (
    ProceduralMemoryStore,
)


def create_episode(
    action: str,
    cycle: int,
) -> EpisodicMemory:

    return EpisodicMemory(
        content=f"Executed {action}.",
        action=action,
        result={
            "success": True,
        },
        cycle=cycle,
    )


def test_procedural_learning_from_successful_sequence():

    episodes = [
        create_episode(
            "open_box",
            1,
        ),
        create_episode(
            "take_key",
            2,
        ),
        create_episode(
            "unlock_door",
            3,
        ),
    ]

    store = ProceduralMemoryStore()

    learning = ProceduralLearning(
        procedural_store=store,
    )

    memory = learning.learn_from_episode_sequence(
        goal="Open the door",
        episodes=episodes,
    )

    assert memory is not None
    assert memory.skill == "open_the_door"

    assert memory.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    assert memory.success_count == 1
    assert memory.success_rate == 1.0

    assert store.count() == 1


def test_procedural_learning_ignores_failed_sequence():

    episodes = [
        EpisodicMemory(
            content="Opened box.",
            action="open_box",
            result={
                "success": True,
            },
        ),
        EpisodicMemory(
            content="Failed to take key.",
            action="take_key",
            result={
                "success": False,
            },
        ),
    ]

    store = ProceduralMemoryStore()

    learning = ProceduralLearning(
        procedural_store=store,
    )

    memory = learning.learn_from_episode_sequence(
        goal="Open the door",
        episodes=episodes,
    )

    assert memory is None
    assert store.count() == 0
