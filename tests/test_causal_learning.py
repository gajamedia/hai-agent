from agent.core.action import Action
from agent.learning.causal_extractor import CausalExtractor
from agent.memory.episodic import EpisodicMemory


def test_causal_extractor_detects_enables_relationship():

    open_box = Action(
        name="open_box",
        description="Open the box",
        handler=lambda: {},
        effects={
            "box": "open",
        },
    )

    take_key = Action(
        name="take_key",
        description="Take the key",
        handler=lambda: {},
        preconditions={
            "box": "open",
        },
        effects={
            "key": "agent",
        },
    )

    previous_episode = EpisodicMemory(
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

    next_episode = EpisodicMemory(
        content="Took key.",
        action="take_key",
        observation_before={
            "box": "open",
        },
        observation_after={
            "box": "open",
            "key": "agent",
        },
        result={
            "success": True,
        },
    )

    actions = {
        "open_box": open_box,
        "take_key": take_key,
    }

    extractor = CausalExtractor()

    relationships = extractor.extract(
        previous_episode=previous_episode,
        next_episode=next_episode,
        actions=actions,
    )

    assert len(relationships) == 1

    relationship = relationships[0]

    assert relationship["cause"] == "open_box"
    assert relationship["relation"] == "enables"
    assert relationship["effect"] == "take_key"

    assert relationship["condition"] == {
        "key": "box",
        "value": "open",
    }


def test_causal_extractor_ignores_unrelated_state_change():

    action_a = Action(
        name="action_a",
        description="Action A",
        handler=lambda: {},
        effects={
            "box": "open",
        },
    )

    action_b = Action(
        name="action_b",
        description="Action B",
        handler=lambda: {},
        preconditions={
            "door": "unlocked",
        },
    )

    previous_episode = EpisodicMemory(
        content="Action A.",
        action="action_a",
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

    next_episode = EpisodicMemory(
        content="Action B.",
        action="action_b",
        observation_before={
            "box": "open",
            "door": "locked",
        },
        observation_after={
            "box": "open",
            "door": "unlocked",
        },
        result={
            "success": True,
        },
    )

    actions = {
        "action_a": action_a,
        "action_b": action_b,
    }

    extractor = CausalExtractor()

    relationships = extractor.extract(
        previous_episode=previous_episode,
        next_episode=next_episode,
        actions=actions,
    )

    assert relationships == []
