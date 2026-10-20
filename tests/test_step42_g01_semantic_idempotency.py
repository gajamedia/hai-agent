from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_store import SemanticMemoryStore


def test_reinforce_same_episode_id_is_idempotent():
    memory = SemanticMemory(
        content="door changes from locked to unlocked",
        subject="door",
        predicate="changes_from",
        object={"before": "locked", "after": "unlocked"},
        confidence=0.5,
        evidence_count=1,
        source_episodes=["episode-001"],
    )

    original_confidence = memory.confidence
    original_evidence_count = memory.evidence_count

    memory.reinforce(episode_id="episode-001")
    memory.reinforce(episode_id="episode-001")

    assert memory.confidence == original_confidence
    assert memory.evidence_count == original_evidence_count
    assert memory.source_episodes == ["episode-001"]


def test_reinforce_new_episode_id_adds_evidence_once():
    memory = SemanticMemory(
        content="door changes from locked to unlocked",
        subject="door",
        predicate="changes_from",
        object={"before": "locked", "after": "unlocked"},
        confidence=0.5,
        evidence_count=1,
        source_episodes=["episode-001"],
    )

    memory.reinforce(episode_id="episode-002")
    memory.reinforce(episode_id="episode-002")

    assert memory.evidence_count == 2
    assert memory.confidence == 0.6
    assert memory.source_episodes == [
        "episode-001",
        "episode-002",
    ]


def test_store_does_not_double_count_replayed_episode():
    store = SemanticMemoryStore()

    kwargs = {
        "subject": "door",
        "predicate": "changes_from",
        "object_value": {
            "before": "locked",
            "after": "unlocked",
        },
        "episode_id": "episode-001",
    }

    first = store.reinforce(**kwargs)
    original_confidence = first.confidence
    original_evidence_count = first.evidence_count

    second = store.reinforce(**kwargs)

    assert second is first
    assert store.count() == 1
    assert second.confidence == original_confidence
    assert second.evidence_count == original_evidence_count
    assert second.source_episodes == ["episode-001"]
import pytest


class FailingSemanticRepository:
    def all(self):
        return []

    def save(self, memory):
        raise RuntimeError("simulated persistence failure")

    def clear(self):
        raise RuntimeError("simulated clear failure")


def test_failed_persistence_does_not_publish_new_memory():
    store = SemanticMemoryStore(repository=FailingSemanticRepository())

    with pytest.raises(RuntimeError, match="simulated persistence failure"):
        store.reinforce(
            subject="door",
            predicate="changes_from",
            object_value={"before": "locked", "after": "unlocked"},
            episode_id="episode-failed",
        )

    assert store.count() == 0


def test_failed_persistence_rolls_back_existing_memory():
    class ToggleRepository:
        def __init__(self):
            self.fail = False
            self.saved = {}

        def all(self):
            return list(self.saved.values())

        def save(self, memory):
            if self.fail:
                raise RuntimeError("simulated persistence failure")
            self.saved[memory.memory_id] = memory

        def clear(self):
            self.saved.clear()

    repository = ToggleRepository()
    store = SemanticMemoryStore(repository=repository)

    memory = store.reinforce(
        subject="door",
        predicate="changes_from",
        object_value={"before": "locked", "after": "unlocked"},
        episode_id="episode-001",
    )

    before = (
        memory.confidence,
        memory.evidence_count,
        list(memory.source_episodes),
    )

    repository.fail = True

    with pytest.raises(RuntimeError, match="simulated persistence failure"):
        store.reinforce(
            subject="door",
            predicate="changes_from",
            object_value={"before": "locked", "after": "unlocked"},
            episode_id="episode-002",
        )

    assert (
        memory.confidence,
        memory.evidence_count,
        memory.source_episodes,
    ) == before
