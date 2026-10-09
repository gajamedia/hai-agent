from agent.memory.causal import CausalMemory
from agent.memory.causal_store import CausalMemoryStore


def test_causal_memory_reinforcement():

    memory = CausalMemory(
        content="open_box enables take_key",
        cause="open_box",
        relation="enables",
        effect="take_key",
        condition={
            "key": "box",
            "value": "open",
        },
        confidence=0.5,
        evidence_count=1,
    )

    memory.reinforce(
        episode_ids=["episode-2", "episode-3"]
    )

    assert memory.evidence_count == 3
    assert memory.confidence == 0.6
    assert memory.source_episodes == [
        "episode-2",
        "episode-3",
    ]


def test_causal_store_reuses_existing_knowledge():

    store = CausalMemoryStore()

    condition = {
        "key": "box",
        "value": "open",
    }

    first = store.reinforce(
        cause="open_box",
        relation="enables",
        effect="take_key",
        condition=condition,
        episode_ids=[
            "episode-1",
            "episode-2",
        ],
    )

    second = store.reinforce(
        cause="open_box",
        relation="enables",
        effect="take_key",
        condition=condition,
        episode_ids=[
            "episode-3",
            "episode-4",
        ],
    )

    assert first is second
    assert store.count() == 1
    assert second.evidence_count == 4
    assert second.confidence == 0.7
