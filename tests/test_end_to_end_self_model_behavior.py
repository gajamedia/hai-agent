from pathlib import Path

import pytest

from agent.learning.agent_self_awareness import (
    AgentSelfAwareness,
)

from agent.learning.end_to_end_self_model_behavior import (
    EndToEndSelfModelBehaviorEvaluator,
    EndToEndSelfModelBehaviorResult,
    build_end_to_end_self_model_behavior,
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
            "attempts": 20,
            "successes": 14,
            "failures": 6,
            "confidence": 0.70,
            "uncertainty": 0.30,
        },
        "reasoning": {
            "attempts": 20,
            "successes": 13,
            "failures": 7,
            "confidence": 0.65,
            "uncertainty": 0.35,
        },
    }


# ============================================================
# Initialization
# ============================================================


def test_evaluator_requires_self_awareness():
    with pytest.raises(ValueError):
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=None
        )


def test_evaluator_rejects_wrong_self_awareness_type():
    with pytest.raises(ValueError):
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=object()
        )


def test_builder_returns_evaluator(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        build_end_to_end_self_model_behavior(
            self_awareness=awareness
        )
    )

    assert isinstance(
        evaluator,
        EndToEndSelfModelBehaviorEvaluator,
    )


# ============================================================
# End-to-End Evaluation
# ============================================================


def test_evaluate_returns_result(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
    )

    assert isinstance(
        result,
        EndToEndSelfModelBehaviorResult,
    )


def test_strong_initial_state_favors_exploitation(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        strong_capabilities(),
        strong_capabilities(),
    )

    assert (
        result.initial_mode
        == SELF_MODEL_MODE_EXPLOIT
    )


def test_weak_final_state_favors_exploration(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        strong_capabilities(),
        weak_capabilities(),
    )

    assert (
        result.final_mode
        == SELF_MODEL_MODE_EXPLORE
    )

    assert result.moved_to_exploration


def test_strong_final_state_favors_exploitation(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
    )

    assert (
        result.final_mode
        == SELF_MODEL_MODE_EXPLOIT
    )

    assert result.moved_to_exploitation


# ============================================================
# Self-Knowledge Direction
# ============================================================


def test_improving_capability_evidence_improves_self_knowledge(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
    )

    assert (
        result.confidence_delta
        > 0.0
    )

    assert (
        result.uncertainty_delta
        < 0.0
    )

    assert result.improved_self_knowledge


def test_degrading_capability_evidence_degrades_self_knowledge(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        strong_capabilities(),
        weak_capabilities(),
    )

    assert (
        result.confidence_delta
        < 0.0
    )

    assert (
        result.uncertainty_delta
        > 0.0
    )

    assert result.degraded_self_knowledge


def test_same_capabilities_produce_no_directional_change(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    capabilities = strong_capabilities()

    result = evaluator.evaluate(
        capabilities,
        capabilities,
    )

    assert result.uncertainty_delta == 0.0
    assert result.confidence_delta == 0.0


# ============================================================
# Mode Transition
# ============================================================


def test_mode_changes_when_self_model_changes(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
    )

    assert result.mode_changed


def test_mode_does_not_change_when_state_direction_is_same(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        strong_capabilities(),
        strong_capabilities(),
    )

    assert not result.mode_changed


# ============================================================
# Persistence
# ============================================================


def test_persistence_is_used_when_enabled(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
        persist=True,
    )

    assert (
        result.final_snapshot.persisted_version
        == 2
    )


def test_no_persistence_version_without_persistence(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
        persist=False,
    )

    assert (
        result.initial_snapshot.persisted_version
        is None
    )

    assert (
        result.final_snapshot.persisted_version
        is None
    )


def test_restore_preserves_final_state(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
        persist=True,
        restore=True,
    )

    assert (
        result.restored_snapshot
        is not None
    )

    assert result.persistence_continuity

    assert result.restore_matches_final


def test_evaluate_and_restore_matches_evaluate(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate_and_restore(
        weak_capabilities(),
        strong_capabilities(),
    )

    assert (
        result.restored_snapshot
        is not None
    )

    assert result.restore_matches_final


def test_new_awareness_instance_can_restore_final_state(
    tmp_path,
):
    first = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=first
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
        persist=True,
    )

    second = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    restored = second.restore()

    assert restored is not None

    assert (
        restored.self_uncertainty
        == result.final_snapshot.self_uncertainty
    )

    assert (
        restored.decision_influence
        == result.final_snapshot.decision_influence
    )

    assert (
        restored.persisted_version
        == result.final_snapshot.persisted_version
    )


# ============================================================
# End-to-End Success
# ============================================================


def test_end_to_end_success_without_restore(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
    )

    assert result.end_to_end_success


def test_end_to_end_success_with_restore(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
        persist=True,
        restore=True,
    )

    assert result.end_to_end_success


def test_end_to_end_fails_when_restore_not_available(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
        persist=False,
        restore=True,
    )

    assert (
        result.restored_snapshot
        is None
    )

    assert not result.persistence_continuity

    assert result.end_to_end_success


# ============================================================
# Result Properties
# ============================================================


def test_result_snapshot_contains_expected_fields(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    result = evaluator.evaluate(
        weak_capabilities(),
        strong_capabilities(),
        persist=True,
        restore=True,
    )

    snapshot = result.snapshot()

    expected_fields = {
        "initial_snapshot",
        "final_snapshot",
        "restored_snapshot",
        "uncertainty_delta",
        "confidence_delta",
        "mode_changed",
        "self_knowledge_changed",
        "persistence_continuity",
        "initial_uncertainty",
        "final_uncertainty",
        "initial_confidence",
        "final_confidence",
        "initial_mode",
        "final_mode",
        "improved_self_knowledge",
        "degraded_self_knowledge",
        "moved_to_exploration",
        "moved_to_exploitation",
        "is_final_mode_neutral",
        "restore_matches_final",
        "end_to_end_success",
    }

    assert expected_fields.issubset(
        snapshot.keys()
    )


# ============================================================
# Validation
# ============================================================


def test_none_initial_capabilities_rejected(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            None,
            strong_capabilities(),
        )


def test_none_final_capabilities_rejected(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            strong_capabilities(),
            None,
        )


def test_invalid_initial_capabilities_type_rejected(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            ["planning"],
            strong_capabilities(),
        )


def test_invalid_final_capabilities_type_rejected(
    tmp_path,
):
    awareness = AgentSelfAwareness(
        agent_name="agent-a",
        memory_path=tmp_path,
    )

    evaluator = (
        EndToEndSelfModelBehaviorEvaluator(
            self_awareness=awareness
        )
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            strong_capabilities(),
            ["planning"],
        )
