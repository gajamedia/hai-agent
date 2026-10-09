from agent.core.action import Action
from agent.learning.reflection import ReflectionEngine
from agent.memory.episodic import EpisodicMemory
from agent.memory.reflection_store import (
    ReflectionMemoryStore,
)


def main():

    print("================================")
    print("FAILURE + REFLECTION")
    print("================================")
    print()

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
        content="Agent failed to take key.",
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
        cycle=1,
    )

    store = ReflectionMemoryStore()

    engine = ReflectionEngine(
        reflection_store=store
    )

    reflection = engine.reflect(
        episode=episode,
        action=action,
    )

    print("FAILED ACTION")
    print(
        "  Action:",
        episode.action,
    )

    print(
        "  Result:",
        episode.result,
    )

    print()

    print("REFLECTION")
    print(
        "  Problem:",
        reflection.problem,
    )

    print(
        "  Cause:",
        reflection.cause,
    )

    print(
        "  Condition:",
        reflection.condition,
    )

    print(
        "  Lesson:",
        reflection.lesson,
    )

    print(
        "  Confidence:",
        reflection.confidence,
    )

    print(
        "  Applied:",
        reflection.applied,
    )

    print()

    print("================================")
    print("SUMMARY")
    print("================================")

    print(
        "Reflection memories:",
        store.count(),
    )


if __name__ == "__main__":
    main()
