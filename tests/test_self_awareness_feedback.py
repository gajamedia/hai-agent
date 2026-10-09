import pytest

from agent.learning.self_awareness_feedback import (
    SelfAwarenessFeedbackRecord,
    SelfAwarenessFeedbackSnapshot,
    SelfAwarenessFeedbackTrace,
)


def test_empty_trace():
    trace = SelfAwarenessFeedbackTrace()

    assert trace.count() == 0
    assert len(trace) == 0
    assert trace.first() is None
    assert trace.latest() is None


def test_record_creates_feedback_record():
    trace = SelfAwarenessFeedbackTrace()

    record = trace.record(
        uncertainty=0.8,
        confidence=0.2,
        preferred_mode="explore",
        selected_mode="explore",
        evidence_count=3,
        mode_changed=True,
    )

    assert isinstance(
        record,
        SelfAwarenessFeedbackRecord,
    )

    assert record.sequence == 1
    assert record.uncertainty == 0.8
    assert record.confidence == 0.2
    assert record.preferred_mode == "explore"
    assert record.selected_mode == "explore"
    assert record.evidence_count == 3
    assert record.mode_changed is True


def test_sequence_increments():
    trace = SelfAwarenessFeedbackTrace()

    first = trace.record(
        uncertainty=0.8,
        confidence=0.2,
    )

    second = trace.record(
        uncertainty=0.7,
        confidence=0.3,
    )

    assert first.sequence == 1
    assert second.sequence == 2


def test_snapshot_contains_records():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.8,
        confidence=0.2,
    )

    trace.record(
        uncertainty=0.6,
        confidence=0.4,
    )

    snapshot = trace.snapshot()

    assert isinstance(
        snapshot,
        SelfAwarenessFeedbackSnapshot,
    )

    assert snapshot.count == 2
    assert snapshot.has_feedback
    assert snapshot.first is not None
    assert snapshot.latest is not None


def test_uncertainty_delta():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.8,
        confidence=0.2,
    )

    trace.record(
        uncertainty=0.5,
        confidence=0.5,
    )

    assert (
        trace.uncertainty_delta()
        == -0.3
    )


def test_confidence_delta():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.8,
        confidence=0.2,
    )

    trace.record(
        uncertainty=0.5,
        confidence=0.5,
    )

    assert (
        trace.confidence_delta()
        == 0.3
    )


def test_no_delta_for_single_record():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.8,
        confidence=0.2,
    )

    assert (
        trace.uncertainty_delta()
        == 0.0
    )

    assert (
        trace.confidence_delta()
        == 0.0
    )


def test_mode_values_are_normalized():
    from agent.cognition.agent_decision import (
        DecisionMode,
    )

    trace = SelfAwarenessFeedbackTrace()

    record = trace.record(
        uncertainty=0.8,
        confidence=0.2,
        preferred_mode=DecisionMode.EXPLORE,
        selected_mode=DecisionMode.EXPLOIT,
    )

    assert (
        record.preferred_mode
        == DecisionMode.EXPLORE.value
    )

    assert (
        record.selected_mode
        == DecisionMode.EXPLOIT.value
    )


def test_none_modes_are_allowed():
    trace = SelfAwarenessFeedbackTrace()

    record = trace.record(
        uncertainty=0.5,
        confidence=0.5,
    )

    assert record.preferred_mode is None
    assert record.selected_mode is None


def test_changed_decision_count():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.8,
        confidence=0.2,
        mode_changed=True,
    )

    trace.record(
        uncertainty=0.7,
        confidence=0.3,
        mode_changed=False,
    )

    trace.record(
        uncertainty=0.6,
        confidence=0.4,
        mode_changed=True,
    )

    assert (
        trace.changed_decision_count()
        == 2
    )


def test_selected_modes_snapshot():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.8,
        confidence=0.2,
        selected_mode="explore",
    )

    trace.record(
        uncertainty=0.4,
        confidence=0.6,
        selected_mode="exploit",
    )

    snapshot = trace.snapshot()

    assert (
        snapshot.selected_modes
        == (
            "explore",
            "exploit",
        )
    )


def test_preferred_modes_snapshot():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.8,
        confidence=0.2,
        preferred_mode="explore",
    )

    trace.record(
        uncertainty=0.4,
        confidence=0.6,
        preferred_mode="exploit",
    )

    snapshot = trace.snapshot()

    assert (
        snapshot.preferred_modes
        == (
            "explore",
            "exploit",
        )
    )


def test_trace_returns_immutable_records_collection():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.8,
        confidence=0.2,
    )

    records = trace.records()

    assert isinstance(
        records,
        tuple,
    )

    assert len(records) == 1


def test_probability_range_is_validated():
    trace = SelfAwarenessFeedbackTrace()

    with pytest.raises(ValueError):
        trace.record(
            uncertainty=1.1,
            confidence=0.0,
        )

    with pytest.raises(ValueError):
        trace.record(
            uncertainty=0.0,
            confidence=-0.1,
        )


def test_probability_type_is_validated():
    trace = SelfAwarenessFeedbackTrace()

    with pytest.raises(TypeError):
        trace.record(
            uncertainty="high",
            confidence=0.0,
        )


def test_evidence_count_is_validated():
    trace = SelfAwarenessFeedbackTrace()

    with pytest.raises(ValueError):
        trace.record(
            uncertainty=0.5,
            confidence=0.5,
            evidence_count=-1,
        )

    with pytest.raises(TypeError):
        trace.record(
            uncertainty=0.5,
            confidence=0.5,
            evidence_count=1.5,
        )


def test_mode_changed_is_validated():
    trace = SelfAwarenessFeedbackTrace()

    with pytest.raises(TypeError):
        trace.record(
            uncertainty=0.5,
            confidence=0.5,
            mode_changed="yes",
        )


def test_clear_resets_sequence():
    trace = SelfAwarenessFeedbackTrace()

    trace.record(
        uncertainty=0.5,
        confidence=0.5,
    )

    trace.clear()

    assert trace.count() == 0

    record = trace.record(
        uncertainty=0.4,
        confidence=0.6,
    )

    assert record.sequence == 1


def test_self_model_stable_property():
    trace = SelfAwarenessFeedbackTrace()

    record = trace.record(
        uncertainty=0.3,
        confidence=0.7,
    )

    assert record.self_model_stable


def test_self_model_stable_is_false_when_not_complementary():
    trace = SelfAwarenessFeedbackTrace()

    record = trace.record(
        uncertainty=0.3,
        confidence=0.6,
    )

    assert not record.self_model_stable
