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
    before,
    after,
):

    return EpisodicMemory(
        content="Test contextual episode.",
        action="open_box",
        observation_before=before,
        observation_after=after,
        result={
            "success": True,
        },
    )


def test_same_outcome_different_contexts_creates_separate_knowledge():

    store = SemanticMemoryStore()

    learning = SemanticLearning(
        semantic_store=store
    )

    episode_one = create_episode(
        before={
            "box": "closed",
            "condition": "normal",
        },
        after={
            "box": "open",
            "condition": "normal",
        },
    )

    episode_two = create_episode(
        before={
            "box": "closed",
            "condition": "jammed",
        },
        after={
            "box": "open",
            "condition": "jammed",
        },
    )

    learning.learn_from_episode(
        episode_one
    )

    learning.learn_from_episode(
        episode_two
    )

    assert store.count() == 2

    memories = store.find(
        subject="box",
        predicate="changes_from",
    )

    assert len(memories) == 2


def test_same_context_reinforces_same_knowledge():

    store = SemanticMemoryStore()

    learning = SemanticLearning(
        semantic_store=store
    )

    episode_one = create_episode(
        before={
            "box": "closed",
            "condition": "normal",
        },
        after={
            "box": "open",
            "condition": "normal",
        },
    )

    episode_two = create_episode(
        before={
            "box": "closed",
            "condition": "normal",
        },
        after={
            "box": "open",
            "condition": "normal",
        },
    )

    first = learning.learn_from_episode(
        episode_one
    )[0]

    second = learning.learn_from_episode(
        episode_two
    )[0]

    assert first is second
    assert store.count() == 1
    assert second.evidence_count == 2
    assert second.confidence == 0.7


def test_context_lookup():

    store = SemanticMemoryStore()

    learning = SemanticLearning(
        semantic_store=store
    )

    episode = create_episode(
        before={
            "box": "closed",
            "condition": "jammed",
        },
        after={
            "box": "closed",
            "condition": "jammed",
        },
    )

    # No state change means no semantic memory.
    learning.learn_from_episode(
        episode
    )

    assert store.count() == 0

    episode = create_episode(
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
        episode
    )

    results = store.find_by_context(
        {
            "box": "closed",
            "condition": "normal",
        }
    )

    assert len(results) == 1

    assert results[0].subject == "box"
