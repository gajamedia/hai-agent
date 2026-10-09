from agent.learning.contradiction_detector import (
    ContradictionDetector,
)

from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory


def create_memory():

    return SemanticMemory(
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


def test_detects_contradiction():

    memory = create_memory()

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

    detector = ContradictionDetector()

    contradiction = detector.detect_semantic(
        memory=memory,
        episode=episode,
    )

    assert contradiction is not None

    assert contradiction.subject == "box"

    assert (
        contradiction.expected_object
        == {
            "before": "closed",
            "after": "open",
        }
    )

    assert (
        contradiction.observed_object
        == {
            "before": "closed",
            "after": "closed",
        }
    )


def test_failed_action_is_not_contradiction():

    memory = create_memory()

    episode = EpisodicMemory(
        content="Failed to open box.",
        action="open_box",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "closed",
        },
        result={
            "success": False,
        },
    )

    detector = ContradictionDetector()

    contradiction = detector.detect_semantic(
        memory=memory,
        episode=episode,
    )

    assert contradiction is None


def test_consistent_observation_is_not_contradiction():

    memory = create_memory()

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

    detector = ContradictionDetector()

    contradiction = detector.detect_semantic(
        memory=memory,
        episode=episode,
    )

    assert contradiction is None

def test_different_context_is_not_contradiction():

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

    episode = EpisodicMemory(
        content="Box remained closed while jammed.",
        action="open_box",
        observation_before={
            "box": "closed",
            "condition": "jammed",
        },
        observation_after={
            "box": "closed",
            "condition": "jammed",
        },
        result={
            "success": True,
        },
    )

    detector = ContradictionDetector()

    contradiction = detector.detect_semantic(
        memory=memory,
        episode=episode,
    )

    assert contradiction is None
