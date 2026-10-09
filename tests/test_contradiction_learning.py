from agent.learning.contradiction_learning import (
    ContradictionLearning,
)

from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_store import (
    SemanticMemoryStore,
)


def test_contradiction_learning():

    store = SemanticMemoryStore()

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

    episode = EpisodicMemory(
        content="Box remained closed.",
        action="open_box",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "closed",
        },
        result={
            "success": True,
        },
    )

    learning = ContradictionLearning(
        semantic_store=store
    )

    contradictions = (
        learning.check_episode(
            episode
        )
    )

    assert len(contradictions) == 1

    assert memory.contradiction_count == 1
    assert memory.confidence == 0.7

    assert (
        episode.memory_id
        in memory.source_episodes
    )


def test_consistent_episode_does_not_change_knowledge():

    store = SemanticMemoryStore()

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

    episode = EpisodicMemory(
        content="Box opened.",
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
    )

    learning = ContradictionLearning(
        semantic_store=store
    )

    contradictions = (
        learning.check_episode(
            episode
        )
    )

    assert contradictions == []

    assert memory.contradiction_count == 0
    assert memory.confidence == 0.8
