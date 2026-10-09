from agent.memory.semantic import SemanticMemory
from agent.memory.semantic_store import SemanticMemoryStore


def test_semantic_memory_reinforcement():

    memory = SemanticMemory(
        content="door changes from locked to unlocked",
        subject="door",
        predicate="changes_from",
        object={
            "before": "locked",
            "after": "unlocked",
        },
        confidence=0.5,
        evidence_count=1,
    )

    memory.reinforce(
        episode_id="episode-2"
    )

    assert memory.evidence_count == 2
    assert memory.confidence == 0.6
    assert "episode-2" in memory.source_episodes


def test_semantic_memory_contradiction():

    memory = SemanticMemory(
        content="door changes from locked to unlocked",
        subject="door",
        predicate="changes_from",
        object={
            "before": "locked",
            "after": "unlocked",
        },
        confidence=0.8,
        evidence_count=3,
    )

    memory.contradict()

    assert memory.contradiction_count == 1
    assert memory.confidence == 0.7


def test_semantic_store_reuses_existing_knowledge():

    store = SemanticMemoryStore()

    object_value = {
        "before": "locked",
        "after": "unlocked",
    }

    first = store.reinforce(
        subject="door",
        predicate="changes_from",
        object_value=object_value,
        episode_id="episode-1",
    )

    second = store.reinforce(
        subject="door",
        predicate="changes_from",
        object_value=object_value,
        episode_id="episode-2",
    )

    assert first is second

    assert store.count() == 1

    assert second.evidence_count == 2

    assert second.confidence == 0.7

    assert second.source_episodes == [
        "episode-1",
        "episode-2",
    ]

def test_semantic_memory_registers_contradiction():

    memory = SemanticMemory(
        content="box changes from closed to open",
        subject="box",
        predicate="changes_from",
        object={
            "before": "closed",
            "after": "open",
        },
        confidence=0.8,
        evidence_count=3,
    )

    memory.register_contradiction(
        episode_id="episode-4"
    )

    assert memory.contradiction_count == 1
    assert memory.confidence == 0.7
    assert "episode-4" in memory.source_episodes

def test_semantic_memory_context():

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
        context={
            "condition": "normal",
        },
        confidence=0.8,
        evidence_count=3,
    )

    assert memory.context == {
        "condition": "normal",
    }

    assert memory.matches_context(
        {
            "condition": "normal",
            "door": "locked",
        }
    )

    assert not memory.matches_context(
        {
            "condition": "jammed",
        }
    )
