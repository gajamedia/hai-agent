from agent.learning.learned_state_restoration import (
    UnifiedLearnedStateRestorationResult,
)


def test_no_state_result():
    result = UnifiedLearnedStateRestorationResult(
        agent_name="agent",
        self_awareness_restored=False,
        calibrated_influence_restored=False,
        self_awareness_version=None,
        calibrated_influence_version=None,
        consistent=True,
        reason="No persisted learned state found.",
    )

    assert not result.has_learned_state
    assert not result.fully_restored
    assert not result.partially_restored


def test_full_restore_result():
    result = UnifiedLearnedStateRestorationResult(
        agent_name="agent",
        self_awareness_restored=True,
        calibrated_influence_restored=True,
        self_awareness_version=3,
        calibrated_influence_version=4,
        consistent=True,
        reason="restored",
    )

    assert result.has_learned_state
    assert result.fully_restored
    assert not result.partially_restored


def test_partial_restore_result():
    result = UnifiedLearnedStateRestorationResult(
        agent_name="agent",
        self_awareness_restored=True,
        calibrated_influence_restored=False,
        self_awareness_version=3,
        calibrated_influence_version=None,
        consistent=True,
        reason="self-awareness only",
    )

    assert result.has_learned_state
    assert not result.fully_restored
    assert result.partially_restored


def test_inconsistent_restore_is_partial():
    result = UnifiedLearnedStateRestorationResult(
        agent_name="agent",
        self_awareness_restored=True,
        calibrated_influence_restored=False,
        self_awareness_version=3,
        calibrated_influence_version=4,
        consistent=False,
        reason="mismatch",
    )

    assert result.has_learned_state
    assert not result.fully_restored
    assert result.partially_restored


def test_snapshot():
    result = UnifiedLearnedStateRestorationResult(
        agent_name="agent",
        self_awareness_restored=True,
        calibrated_influence_restored=True,
        self_awareness_version=3,
        calibrated_influence_version=4,
        consistent=True,
        reason="restored",
    )

    snapshot = result.snapshot

    assert snapshot["agent_name"] == "agent"
    assert snapshot["self_awareness_restored"] is True
    assert (
        snapshot["calibrated_influence_restored"]
        is True
    )
    assert snapshot["self_awareness_version"] == 3
    assert (
        snapshot["calibrated_influence_version"]
        == 4
    )
    assert snapshot["consistent"] is True
    assert snapshot["fully_restored"] is True