import pytest

from agent.learning.online_causal_learning import OnlineCausalLearning
from agent.memory.causal_store import CausalMemoryStore
from agent.memory.episodic import EpisodicMemory


def test_causal_store_replay_does_not_double_count_episode_pair():
    store = CausalMemoryStore()

    kwargs = {
        "cause": "open_box",
        "relation": "enables",
        "effect": "take_key",
        "condition": {"key": "box", "value": "open"},
        "episode_ids": ["episode-001", "episode-002"],
    }

    first = store.reinforce(**kwargs)
    original = (
        first.evidence_count,
        first.confidence,
        list(first.source_episodes),
    )

    second = store.reinforce(**kwargs)

    assert second is first
    assert (
        second.evidence_count,
        second.confidence,
        second.source_episodes,
    ) == original
    assert store.count() == 1


def test_causal_store_does_not_publish_memory_when_persistence_fails():
    class FailingRepository:
        def all(self):
            return []

        def save(self, memory):
            raise RuntimeError("simulated causal persistence failure")

    store = CausalMemoryStore(repository=FailingRepository())

    with pytest.raises(
        RuntimeError,
        match="simulated causal persistence failure",
    ):
        store.reinforce(
            cause="open_box",
            relation="enables",
            effect="take_key",
            condition={"key": "box", "value": "open"},
            episode_ids=["episode-001", "episode-002"],
        )

    assert store.count() == 0


def test_causal_store_restores_existing_memory_after_persistence_failure():
    class ToggleRepository:
        def __init__(self):
            self.fail = False
            self.saved = {}

        def all(self):
            return list(self.saved.values())

        def save(self, memory):
            if self.fail:
                raise RuntimeError("simulated causal persistence failure")
            self.saved[memory.memory_id] = memory

    repository = ToggleRepository()
    store = CausalMemoryStore(repository=repository)

    memory = store.reinforce(
        cause="open_box",
        relation="enables",
        effect="take_key",
        condition={"key": "box", "value": "open"},
        episode_ids=["episode-001", "episode-002"],
    )

    original = (
        memory.evidence_count,
        memory.confidence,
        list(memory.source_episodes),
    )

    repository.fail = True

    with pytest.raises(
        RuntimeError,
        match="simulated causal persistence failure",
    ):
        store.reinforce(
            cause="open_box",
            relation="enables",
            effect="take_key",
            condition={"key": "box", "value": "open"},
            episode_ids=["episode-003"],
        )

    assert (
        memory.evidence_count,
        memory.confidence,
        memory.source_episodes,
    ) == original


def test_online_causal_learning_ignores_replayed_episode():
    class FakeCausalLearning:
        def __init__(self):
            self.calls = []

        def learn_from_episode_pair(
            self,
            previous_episode,
            next_episode,
            actions,
        ):
            self.calls.append(
                (previous_episode.memory_id, next_episode.memory_id)
            )
            return []

    learner = FakeCausalLearning()
    online = OnlineCausalLearning(learner)

    episode = EpisodicMemory(
        memory_id="episode-001",
        content="Opened box.",
        action="open_box",
        observation_before={"box": "closed"},
        observation_after={"box": "open"},
        result={"success": True},
    )

    online.observe_episode(episode, actions={})
    online.observe_episode(episode, actions={})

    assert learner.calls == []
