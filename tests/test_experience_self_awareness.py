from types import SimpleNamespace

from agent.learning.agent_self_awareness import (
    AgentSelfAwareness,
)
from agent.learning.experience_self_awareness import (
    ExperienceSelfAwarenessUpdater,
)


def make_episode(
    goal: str,
    success: bool,
):
    return SimpleNamespace(
        goal=goal,
        action=None,
        result={
            "success": success,
        },
    )


def test_normalize_capability():
    assert (
        ExperienceSelfAwarenessUpdater
        .normalize_capability(
            "Open the Door"
        )
        == "open_the_door"
    )


def test_normalize_capability_handles_hyphen():
    assert (
        ExperienceSelfAwarenessUpdater
        .normalize_capability(
            "open-the-door"
        )
        == "open_the_door"
    )


def test_normalize_capability_returns_none_for_empty():
    assert (
        ExperienceSelfAwarenessUpdater
        .normalize_capability("")
        is None
    )


def test_build_capability_evidence_counts_success():
    updater = ExperienceSelfAwarenessUpdater(
        AgentSelfAwareness(
            agent_name="test-agent"
        )
    )

    episodes = [
        make_episode(
            "Open the door",
            True,
        ),
        make_episode(
            "Open the door",
            True,
        ),
    ]

    evidence = (
        updater.build_capability_evidence(
            episodes
        )
    )

    assert evidence == {
        "open_the_door": {
            "attempts": 2,
            "successes": 2,
            "failures": 0,
        }
    }


def test_build_capability_evidence_counts_failures():
    updater = ExperienceSelfAwarenessUpdater(
        AgentSelfAwareness(
            agent_name="test-agent"
        )
    )

    episodes = [
        make_episode(
            "Open the door",
            False,
        ),
        make_episode(
            "Open the door",
            True,
        ),
    ]

    evidence = (
        updater.build_capability_evidence(
            episodes
        )
    )

    assert evidence == {
        "open_the_door": {
            "attempts": 2,
            "successes": 1,
            "failures": 1,
        }
    }


def test_build_capability_evidence_supports_multiple_capabilities():
    updater = ExperienceSelfAwarenessUpdater(
        AgentSelfAwareness(
            agent_name="test-agent"
        )
    )

    episodes = [
        make_episode(
            "Open the door",
            True,
        ),
        make_episode(
            "Take the key",
            False,
        ),
        make_episode(
            "Take the key",
            True,
        ),
    ]

    evidence = (
        updater.build_capability_evidence(
            episodes
        )
    )

    assert evidence == {
        "open_the_door": {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        },
        "take_the_key": {
            "attempts": 2,
            "successes": 1,
            "failures": 1,
        },
    }


def test_episodes_without_outcome_are_ignored():
    updater = ExperienceSelfAwarenessUpdater(
        AgentSelfAwareness(
            agent_name="test-agent"
        )
    )

    episodes = [
        SimpleNamespace(
            goal="Open the door",
            action=None,
            result={},
        )
    ]

    evidence = (
        updater.build_capability_evidence(
            episodes
        )
    )

    assert evidence == {}


def test_action_is_used_when_goal_is_missing():
    updater = ExperienceSelfAwarenessUpdater(
        AgentSelfAwareness(
            agent_name="test-agent"
        )
    )

    episodes = [
        SimpleNamespace(
            goal=None,
            action="take-key",
            result={
                "success": True,
            },
        )
    ]

    evidence = (
        updater.build_capability_evidence(
            episodes
        )
    )

    assert evidence == {
        "take_key": {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        }
    }


def test_update_changes_self_awareness():
    self_awareness = AgentSelfAwareness(
        agent_name="test-agent"
    )

    updater = ExperienceSelfAwarenessUpdater(
        self_awareness
    )

    episodes = [
        make_episode(
            "Open the door",
            True,
        ),
        make_episode(
            "Open the door",
            True,
        ),
        make_episode(
            "Open the door",
            False,
        ),
    ]

    result = updater.update(
        episodes,
        persist=False,
    )

    assert result.updated
    assert result.has_evidence
    assert result.snapshot_available
    assert result.episode_count == 3
    assert result.capability_count == 1

    assert result.capabilities[
        "open_the_door"
    ] == {
        "attempts": 3,
        "successes": 2,
        "failures": 1,
    }

    assert (
        result.snapshot is not None
    )


def test_update_without_evidence_does_not_modify_self_awareness():
    self_awareness = AgentSelfAwareness(
        agent_name="test-agent"
    )

    updater = ExperienceSelfAwarenessUpdater(
        self_awareness
    )

    result = updater.update(
        [],
        persist=False,
    )

    assert not result.updated
    assert not result.has_evidence
    assert not result.snapshot_available
    assert result.episode_count == 0
    assert result.capability_count == 0


def test_repeated_update_rebuilds_instead_of_double_counting():
    self_awareness = AgentSelfAwareness(
        agent_name="test-agent"
    )

    updater = ExperienceSelfAwarenessUpdater(
        self_awareness
    )

    episodes = [
        make_episode(
            "Open the door",
            True,
        ),
    ]

    first = updater.update(
        episodes,
        persist=False,
    )

    second = updater.update(
        episodes,
        persist=False,
    )

    assert (
        first.capabilities
        == second.capabilities
    )

    assert (
        second.capabilities[
            "open_the_door"
        ]["attempts"]
        == 1
    )
