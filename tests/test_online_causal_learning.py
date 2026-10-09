from agent.core.action import Action
from agent.learning.causal_learning import CausalLearning
from agent.learning.online_causal_learning import (
    OnlineCausalLearning,
)
from agent.memory.causal_store import CausalMemoryStore
from agent.memory.episodic import EpisodicMemory


def test_online_causal_learning():

    open_box = Action(
        name="open_box",
        description="Open box",
        handler=lambda: {},
        effects={
            "box": "open",
        },
    )

    take_key = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {},
        preconditions={
            "box": "open",
        },
    )

    actions = {
        "open_box": open_box,
        "take_key": take_key,
    }

    first_episode = EpisodicMemory(
        content="Opened box.",
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

    second_episode = EpisodicMemory(
        content="Took key.",
        action="take_key",
        observation_before={
            "box": "open",
        },
        observation_after={
            "box": "open",
        },
        result={
            "success": True,
        },
    )

    store = CausalMemoryStore()

    causal_learning = CausalLearning(
        causal_store=store,
    )

    online_learning = OnlineCausalLearning(
        causal_learning=causal_learning,
    )

    first_result = (
        online_learning.observe_episode(
            episode=first_episode,
            actions=actions,
        )
    )

    assert first_result == []

    second_result = (
        online_learning.observe_episode(
            episode=second_episode,
            actions=actions,
        )
    )

    assert len(second_result) == 1
    assert store.count() == 1

    memory = store.latest()

    assert memory is not None
    assert memory.cause == "open_box"
    assert memory.relation == "enables"
    assert memory.effect == "take_key"
    assert memory.evidence_count == 2
    assert memory.confidence == 0.6


def test_online_causal_learning_does_not_duplicate_evidence():

    open_box = Action(
        name="open_box",
        description="Open box",
        handler=lambda: {},
        effects={
            "box": "open",
        },
    )

    take_key = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {},
        preconditions={
            "box": "open",
        },
    )

    actions = {
        "open_box": open_box,
        "take_key": take_key,
    }

    first_episode = EpisodicMemory(
        content="Opened box.",
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

    second_episode = EpisodicMemory(
        content="Took key.",
        action="take_key",
        observation_before={
            "box": "open",
        },
        observation_after={
            "box": "open",
        },
        result={
            "success": True,
        },
    )

    store = CausalMemoryStore()

    causal_learning = CausalLearning(
        causal_store=store,
    )

    online_learning = OnlineCausalLearning(
        causal_learning=causal_learning,
    )

    online_learning.observe_episode(
        episode=first_episode,
        actions=actions,
    )

    online_learning.observe_episode(
        episode=second_episode,
        actions=actions,
    )

    online_learning.observe_episode(
        episode=second_episode,
        actions=actions,
    )

    memory = store.latest()

    assert memory is not None
    assert memory.evidence_count == 2
    assert memory.confidence == 0.6
