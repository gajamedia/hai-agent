from agent.learning.contradiction_learning import (
    ContradictionLearning,
)

from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_store import (
    SemanticMemoryStore,
)


def main():

    print("================================")
    print("CONTRADICTION LEARNING")
    print("================================")
    print()

    store = SemanticMemoryStore()

    # --------------------------------------------------
    # Initial knowledge
    # --------------------------------------------------

    memory = SemanticMemory(
        content=(
            "box changes from closed to open"
        ),
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        confidence=0.8,
        evidence_count=3,
    )

    store.add(memory)

    print("INITIAL KNOWLEDGE")
    print()

    print(
        "Knowledge:",
        memory.content,
    )

    print(
        "Confidence:",
        memory.confidence,
    )

    print(
        "Evidence:",
        memory.evidence_count,
    )

    print(
        "Contradictions:",
        memory.contradiction_count,
    )

    print()

    # --------------------------------------------------
    # Contradictory experience
    # --------------------------------------------------

    episode = EpisodicMemory(
        content="Box remained closed.",
        goal="Open the door",
        action="open_box",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "closed",
        },
        result={
            "success": True,
            "message": (
                "The action reported success, "
                "but the box remained closed."
            ),
        },
        cycle=4,
    )

    learning = ContradictionLearning(
        semantic_store=store
    )

    print("================================")
    print("NEW EXPERIENCE")
    print("================================")
    print()

    print(
        "Action:",
        episode.action,
    )

    print(
        "Before:",
        episode.observation_before,
    )

    print(
        "After:",
        episode.observation_after,
    )

    print()

    # --------------------------------------------------
    # Detect contradiction
    # --------------------------------------------------

    contradictions = (
        learning.check_episode(
            episode
        )
    )

    print("================================")
    print("CONTRADICTION ANALYSIS")
    print("================================")
    print()

    if not contradictions:

        print(
            "No contradiction detected."
        )

    else:

        for contradiction in (
            contradictions
        ):

            print(
                "Subject:",
                contradiction.subject,
            )

            print(
                "Predicate:",
                contradiction.predicate,
            )

            print(
                "Expected:",
                contradiction.expected_object,
            )

            print(
                "Observed:",
                contradiction.observed_object,
            )

            print(
                "Reason:",
                contradiction.reason,
            )

            print(
                "Source episode:",
                contradiction.source_episode_id,
            )

            print()

    # --------------------------------------------------
    # Updated knowledge
    # --------------------------------------------------

    print("================================")
    print("UPDATED KNOWLEDGE")
    print("================================")
    print()

    print(
        "Knowledge:",
        memory.content,
    )

    print(
        "Confidence:",
        memory.confidence,
    )

    print(
        "Evidence:",
        memory.evidence_count,
    )

    print(
        "Contradictions:",
        memory.contradiction_count,
    )

    print(
        "Source episodes:",
        len(
            memory.source_episodes
        ),
    )

    print()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("================================")
    print("SUMMARY")
    print("================================")
    print()

    print(
        "Contradictions detected:",
        len(
            contradictions
        ),
    )


if __name__ == "__main__":
    main()
