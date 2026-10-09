from pathlib import Path

import pytest

from agent.learning.agent_self_awareness import (
    AgentSelfAwareness,
    AgentSelfAwarenessSnapshot,
    build_agent_self_awareness,
)

from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
)


# ============================================================
# Helpers
# ============================================================


def strong_capabilities():
    return {
        "planning": {
            "attempts": 100,
            "successes": 100,
            "failures": 0,
            "confidence": 1.0,
            "uncertainty": 0.0,
        },
        "reasoning": {
            "attempts": 100,
            "successes": 95,
            "failures": 5,
            "confidence": 0.95,
            "uncertainty": 0.05,
        },
    }


def weak_capabilities():
    return {
        "planning": {
            "attempts": 10,
            "successes": 1,
            "failures": 9,
            "confidence": 0.10,
            "uncertainty": 0.90,
        },
        "reasoning": {
            "attempts": 10,
            "successes": 2,
            "failures": 8,
            "confidence": 0.20,
            "uncertainty": 0.80,
        },
    }


def medium_capabilities():
    return {
        "planning": {
            "attempts": 10,
            "successes": 7,
            "failures": 3,
            "confidence": 0.70,
            "uncertainty": 0.30,
        }
    }


# ============================================================
# Initialization
# ============================================================


def test_agent_self_awareness_initializes_without_persistence():
    awareness = AgentSelfAwareness(
        agent_name="agent-a"
    )

    assert awareness.agent_name == "agent-a"

    assert awareness.store is None

    assert (
        awareness.self_uncertainty
        is None
    )

    assert (
        awareness.decision_influence
        is None
    )


def test_agent_self_awareness_initializes_with_persistence(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    assert awareness.store is not None

    expected_path = (
        tmp_path
        / "self_model_continuity.sqlite3"
    )

    assert (
        awareness.persistence_path
        == expected_path
    )

    assert expected_path.exists()


def test_builder_returns_agent_self_awareness(
    tmp_path,
):
    awareness = (
        build_agent_self_awareness(
            agent_name="agent-a",
            memory_path=tmp_path,
        )
    )

    assert isinstance(
        awareness,
        AgentSelfAwareness,
    )


# ============================================================
# Evaluation
# ============================================================


def test_evaluate_returns_snapshot(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    snapshot = awareness.evaluate(
        strong_capabilities()
    )

    assert isinstance(
        snapshot,
        AgentSelfAwarenessSnapshot,
    )

    assert snapshot.agent_name == "agent-a"

    assert (
        snapshot.capability_summary
        is not None
    )

    assert (
        snapshot.self_uncertainty
        is not None
    )

    assert (
        snapshot.decision_influence
        is not None
    )


def test_evaluate_creates_self_knowledge(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    awareness.evaluate(
        strong_capabilities()
    )

    assert awareness.has_self_knowledge


def test_evaluate_strong_capabilities_prefers_exploitation(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    snapshot = awareness.evaluate(
        strong_capabilities()
    )

    assert (
        snapshot.decision_influence
        is not None
    )

    assert (
        snapshot.decision_influence.preferred_mode
        == SELF_MODEL_MODE_EXPLOIT
    )

    assert awareness.should_exploit
    assert not awareness.should_explore


def test_evaluate_weak_capabilities_prefers_exploration(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    snapshot = awareness.evaluate(
        weak_capabilities()
    )

    assert (
        snapshot.decision_influence
        is not None
    )

    assert (
        snapshot.decision_influence.preferred_mode
        == SELF_MODEL_MODE_EXPLORE
    )

    assert awareness.should_explore
    assert not awareness.should_exploit


def test_medium_capabilities_produce_self_awareness(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    snapshot = awareness.evaluate(
        medium_capabilities()
    )

    assert (
        snapshot.self_uncertainty
        is not None
    )

    assert (
        snapshot.decision_influence
        is not None
    )

    assert (
        0.0
        <= snapshot.uncertainty
        <= 1.0
    )

    assert (
        0.0
        <= snapshot.confidence
        <= 1.0
    )


# ============================================================
# Persistence
# ============================================================


def test_evaluate_persists_state(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    snapshot = awareness.evaluate(
        strong_capabilities()
    )

    assert snapshot.persisted_version == 1


def test_repeated_evaluation_increments_version(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    first = awareness.evaluate(
        strong_capabilities()
    )

    second = awareness.evaluate(
        weak_capabilities()
    )

    assert first.persisted_version == 1
    assert second.persisted_version == 2


def test_evaluate_without_persist_does_not_create_store_record(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    snapshot = awareness.evaluate(
        strong_capabilities(),
        persist=False,
    )

    assert (
        snapshot.self_uncertainty
        is not None
    )

    assert (
        snapshot.decision_influence
        is not None
    )

    assert (
        snapshot.persisted_version
        is None
    )

    assert awareness.store is not None

    assert not awareness.store.exists(
        "agent-a"
    )


def test_restore_survives_new_awareness_instance(
    tmp_path,
):
    first = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    saved = first.evaluate(
        weak_capabilities()
    )

    second = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    restored = second.restore()

    assert restored is not None

    assert restored.restored is True

    assert (
        restored.persisted_version
        == saved.persisted_version
    )

    assert (
        restored.self_uncertainty
        == saved.self_uncertainty
    )

    assert (
        restored.decision_influence
        == saved.decision_influence
    )


def test_restore_preserves_exploration_preference(
    tmp_path,
):
    first = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    first.evaluate(
        weak_capabilities()
    )

    second = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    restored = second.restore()

    assert restored is not None

    assert (
        second.preferred_mode
        == SELF_MODEL_MODE_EXPLORE
    )

    assert second.should_explore
    assert not second.should_exploit


def test_restore_preserves_exploitation_preference(
    tmp_path,
):
    first = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    first.evaluate(
        strong_capabilities()
    )

    second = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    restored = second.restore()

    assert restored is not None

    assert (
        second.preferred_mode
        == SELF_MODEL_MODE_EXPLOIT
    )

    assert second.should_exploit
    assert not second.should_explore


def test_restore_missing_state_returns_none(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="missing-agent",
        memory_path=tmp_path,
    )

    assert (
        awareness.restore()
        is None
    )


def test_save_current_state(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    awareness.evaluate(
        strong_capabilities(),
        persist=False,
    )

    saved = awareness.save()

    assert saved.version == 1


# ============================================================
# Reset
# ============================================================


def test_reset_clears_in_memory_state(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    awareness.evaluate(
        strong_capabilities()
    )

    awareness.reset()

    assert (
        awareness.self_uncertainty
        is None
    )

    assert (
        awareness.decision_influence
        is None
    )

    assert (
        awareness.capability_summary
        is None
    )

    assert (
        awareness.persisted_version
        is None
    )

    assert awareness.restored is False


def test_reset_can_clear_persistence(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    awareness.evaluate(
        strong_capabilities()
    )

    assert awareness.store is not None

    assert awareness.store.exists(
        "agent-a"
    )

    awareness.reset(
        clear_persistence=True
    )

    assert not awareness.store.exists(
        "agent-a"
    )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot_before_evaluation(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    snapshot = awareness.snapshot()

    assert snapshot.agent_name == "agent-a"

    assert (
        snapshot.capability_summary
        is None
    )

    assert (
        snapshot.self_uncertainty
        is None
    )

    assert (
        snapshot.decision_influence
        is None
    )

    assert (
        snapshot.persisted_version
        is None
    )

    assert snapshot.restored is False

    assert not snapshot.has_self_knowledge

    assert snapshot.uncertainty == 1.0

    assert snapshot.confidence == 0.0

    assert not snapshot.should_explore

    assert not snapshot.should_exploit

    assert (
        snapshot.preferred_mode
        is None
    )

    assert (
        snapshot.strongest_capability
        is None
    )

    assert (
        snapshot.weakest_capability
        is None
    )


def test_snapshot_contains_expected_fields(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    awareness.evaluate(
        strong_capabilities()
    )

    data = (
        awareness
        .snapshot()
        .snapshot()
    )

    expected = {
        "agent_name",
        "capability_summary",
        "self_uncertainty",
        "decision_influence",
        "persisted_version",
        "restored",
        "has_self_knowledge",
        "uncertainty",
        "confidence",
        "preferred_mode",
        "should_explore",
        "should_exploit",
        "strongest_capability",
        "weakest_capability",
    }

    assert expected.issubset(
        data.keys()
    )


# ============================================================
# Validation
# ============================================================


def test_none_agent_name_is_rejected():
    with pytest.raises(ValueError):
        AgentSelfAwareness(
            agent_name=None
        )


def test_empty_agent_name_is_rejected():
    with pytest.raises(ValueError):
        AgentSelfAwareness(
            agent_name=""
        )


def test_whitespace_agent_name_is_rejected():
    with pytest.raises(ValueError):
        AgentSelfAwareness(
            agent_name="   "
        )


def test_none_capabilities_are_rejected(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    with pytest.raises(ValueError):
        awareness.evaluate(
            None
        )


def test_invalid_capabilities_type_is_rejected(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    with pytest.raises(ValueError):
        awareness.evaluate(
            ["planning"]
        )


# ============================================================
# In-Memory Mode
# ============================================================


def test_in_memory_mode_can_evaluate_without_database():
    awareness = AgentSelfAwareness(
        agent_name="agent-a"
    )

    snapshot = awareness.evaluate(
        weak_capabilities()
    )

    assert (
        snapshot.persisted_version
        is None
    )

    assert awareness.store is None

    assert awareness.has_self_knowledge


def test_in_memory_mode_reset_does_not_fail():
    awareness = AgentSelfAwareness(
        agent_name="agent-a"
    )

    awareness.evaluate(
        strong_capabilities()
    )

    awareness.reset()

    assert (
        awareness.self_uncertainty
        is None
    )

    assert (
        awareness.decision_influence
        is None
    )


# ============================================================
# State Properties
# ============================================================


def test_properties_follow_current_state(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    awareness.evaluate(
        weak_capabilities()
    )

    assert (
        awareness.uncertainty
        == awareness.self_uncertainty.uncertainty
    )

    assert (
        awareness.confidence
        == awareness.self_uncertainty.confidence
    )

    assert (
        awareness.preferred_mode
        == awareness.decision_influence.preferred_mode
    )

    assert (
        awareness.strongest_capability
        == awareness.self_uncertainty.strongest_capability
    )

    assert (
        awareness.weakest_capability
        == awareness.self_uncertainty.weakest_capability
    )


def test_restore_sets_restored_flag(
    tmp_path,
):
    first = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    first.evaluate(
        strong_capabilities()
    )

    second = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    assert second.restored is False

    second.restore()

    assert second.restored is True


def test_new_evaluation_clears_restored_flag(
    tmp_path,
):
    first = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    first.evaluate(
        strong_capabilities()
    )

    second = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    second.restore()

    assert second.restored is True

    second.evaluate(
        weak_capabilities(),
        persist=False,
    )

    assert second.restored is False
