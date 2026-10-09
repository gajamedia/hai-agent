from agent.learning.semantic_learning import (
    SemanticLearning,
)

from agent.memory.episodic import (
    EpisodicMemory,
)

from agent.memory.semantic_store import (
    SemanticMemoryStore,
)


def create_episode(
    episode_id,
    before,
    after,
):

    return EpisodicMemory(
        memory_id=episode_id,
        content="Contextual experience.",
        action="open_box",
        observation_before=before,
        observation_after=after,
        result={
            "success": True,
        },
    )


def print_memory(
    memory,
    index,
):

    print(
        f"Knowledge #{index}"
    )

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
        "  Context:",
        memory.context,
    )

    print(
        "  Confidence:",
        memory.confidence,
    )

    print(
        "  Evidence:",
        memory.evidence_count,
    )

    print(
        "  Contradictions:",
        memory.contradiction_count,
    )

    print()


def main():

    print("================================")
    print("CONTEXTUAL KNOWLEDGE")
    print("================================")
    print()

    store = SemanticMemoryStore()

    learning = SemanticLearning(
        semantic_store=store
    )

    # ----------------------------------------------
    # EXPERIENCE 1
    # Normal condition
    # ----------------------------------------------

    episode_one = create_episode(
        episode_id="episode-1",
        before={
            "box": "closed",
            "condition": "normal",
        },
        after={
            "box": "open",
            "condition": "normal",
        },
    )

    learning.learn_from_episode(
        episode_one
    )

    # ----------------------------------------------
    # EXPERIENCE 2
    # Same context
    # Reinforces existing knowledge
    # ----------------------------------------------

    episode_two = create_episode(
        episode_id="episode-2",
        before={
            "box": "closed",
            "condition": "normal",
        },
        after={
            "box": "open",
            "condition": "normal",
        },
    )

    learning.learn_from_episode(
        episode_two
    )

    # ----------------------------------------------
    # EXPERIENCE 3
    # Different context
    # Creates another contextual knowledge
    # ----------------------------------------------

    episode_three = create_episode(
        episode_id="episode-3",
        before={
            "box": "closed",
            "condition": "jammed",
        },
        after={
            "box": "closed",
            "condition": "jammed",
        },
    )

    # No semantic memory will be created because
    # there was no state change.
    learning.learn_from_episode(
        episode_three
    )

    # ----------------------------------------------
    # EXPERIENCE 4
    # Different context with different outcome
    # ----------------------------------------------

    episode_four = create_episode(
        episode_id="episode-4",
        before={
            "box": "closed",
            "condition": "locked",
        },
        after={
            "box": "closed",
            "condition": "locked",
        },
    )

    learning.learn_from_episode(
        episode_four
    )

    # ----------------------------------------------
    # Display knowledge
    # ----------------------------------------------

    print("================================")
    print("LEARNED KNOWLEDGE")
    print("================================")
    print()

    memories = store.all()

    for index, memory in enumerate(
        memories,
        start=1,
    ):

        print_memory(
            memory=memory,
            index=index,
        )

    # ----------------------------------------------
    # Context lookup
    # ----------------------------------------------

    print("================================")
    print("CONTEXT LOOKUP")
    print("================================")
    print()

    normal = store.find_by_context(
        {
            "box": "closed",
            "condition": "normal",
        }
    )

    print(
        "Normal context matches:",
        len(normal),
    )

    jammed = store.find_by_context(
        {
            "box": "closed",
            "condition": "jammed",
        }
    )

    print(
        "Jammed context matches:",
        len(jammed),
    )

    print()

    print("================================")
    print("SUMMARY")
    print("================================")
    print()

    print(
        "Semantic memories:",
        store.count(),
    )


if __name__ == "__main__":
    main()
