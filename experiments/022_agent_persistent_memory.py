from pathlib import Path

from agent.core.agent import Agent
from agent.core.goal import Goal


DATABASE_PATH = (
    Path("data")
    / "experiment_022_agent.db"
)


def main():

    print("=" * 50)
    print("AGENT PERSISTENT LONG-TERM MEMORY")
    print("=" * 50)

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    # =================================================
    # SESSION 1
    # =================================================

    print()
    print("SESSION 1")
    print("-" * 50)

    agent_1 = Agent(
        "hAI",
        memory_path=DATABASE_PATH,
    )

    agent_1.set_goal(
        Goal("open the door")
    )

    agent_1.record_experience(
        action="open_box",
        observation_before={
            "box": "closed",
            "door": "locked",
        },
        observation_after={
            "box": "open",
            "door": "locked",
        },
        result={
            "success": True,
            "message": "Box opened.",
        },
    )

    print(
        "Persistent:",
        agent_1.is_persistent,
    )

    print(
        "Memory counts:",
        agent_1.memory_counts(),
    )

    del agent_1

    # =================================================
    # SESSION 2
    # =================================================

    print()
    print("SESSION 2")
    print("-" * 50)

    agent_2 = Agent(
        "hAI",
        memory_path=DATABASE_PATH,
    )

    print(
        "Persistent:",
        agent_2.is_persistent,
    )

    print(
        "Memory counts:",
        agent_2.memory_counts(),
    )

    episode = (
        agent_2.episodic_memory.latest()
    )

    print()
    print("Restored episode:")
    print(
        "  Action:",
        episode.action
        if episode
        else None,
    )

    print(
        "  Goal:",
        episode.goal
        if episode
        else None,
    )

    semantic_memories = (
        agent_2.semantic_memory.find(
            subject="box",
            predicate="changes_from",
        )
    )

    print()
    print("Restored semantic memory:")

    for memory in semantic_memories:

        print(
            "  Subject:",
            memory.subject,
        )

        print(
            "  Predicate:",
            memory.predicate,
        )

        print(
            "  Object:",
            memory.object,
        )

        print(
            "  Confidence:",
            memory.confidence,
        )

        print(
            "  Evidence:",
            memory.evidence_count,
        )

    print()
    print("RESULT")
    print("-" * 50)

    if (
        episode is not None
        and episode.action == "open_box"
        and len(
            semantic_memories
        ) == 1
        and semantic_memories[0].object
        == {
            "before": "closed",
            "after": "open",
        }
    ):
        print(
            "SUCCESS:"
        )

        print(
            "Agent restored both experience "
            "and learned knowledge after restart."
        )

    else:

        print(
            "FAILURE:"
        )

        print(
            "Agent memory was not restored correctly."
        )


if __name__ == "__main__":
    main()
