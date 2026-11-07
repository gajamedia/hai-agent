import pytest

from agent.autonomous_testing.contracts import FaultInjectionPolicy, FaultType, RecoveryPolicy
from agent.autonomous_testing.fault_injection import (
    FaultInjectionAdapter,
    UnknownExecutionOutcome,
)
from agent.autonomous_testing.recovery import (
    RecoveryAction,
    RecoveryController,
)


class FakeEnvironment:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "ok", "success": True}


def test_action_failure_is_injected_before_underlying_dispatch():
    env = FakeEnvironment()
    adapter = FaultInjectionAdapter(
        env,
        FaultInjectionPolicy(
            enabled=True, fault_type=FaultType.ACTION_FAILURE,
            target_action="open_box", trigger_step=1, max_occurrences=1,
        ),
    )
    result = adapter.run(actions=["open_box"])
    assert result["status"] == "injected_fault"
    assert result["action_executed"] is False
    assert result["outcome_known"] is True
    assert env.calls == []


def test_temporary_unavailability_is_known_non_execution():
    env = FakeEnvironment()
    adapter = FaultInjectionAdapter(
        env,
        FaultInjectionPolicy(
            enabled=True, fault_type=FaultType.TEMPORARY_UNAVAILABILITY,
            target_action="open_box",
        ),
    )
    result = adapter.run(actions=["open_box"])
    assert result["fault"]["outcome_known"] is True
    assert result["fault"]["action_executed"] is False
    assert env.calls == []


def test_observation_interruption_dispatches_once_then_marks_unknown():
    env = FakeEnvironment()
    adapter = FaultInjectionAdapter(
        env,
        FaultInjectionPolicy(
            enabled=True, fault_type=FaultType.OBSERVATION_INTERRUPTION,
            target_action="open_box",
        ),
    )
    with pytest.raises(UnknownExecutionOutcome) as caught:
        adapter.run(actions=["open_box"])
    assert len(env.calls) == 1
    assert caught.value.outcome_known is False
    assert caught.value.record.action_executed is None


def test_fault_is_applied_only_to_target_action():
    env = FakeEnvironment()
    adapter = FaultInjectionAdapter(
        env,
        FaultInjectionPolicy(
            enabled=True, fault_type=FaultType.ACTION_FAILURE,
            target_action="open_box",
        ),
    )
    result = adapter.run(actions=["take_key"])
    assert result["status"] == "ok"
    assert len(env.calls) == 1
    assert adapter.occurrence_count == 0


def test_fault_occurrence_limit_is_enforced():
    env = FakeEnvironment()
    adapter = FaultInjectionAdapter(
        env,
        FaultInjectionPolicy(
            enabled=True, fault_type=FaultType.ACTION_FAILURE,
            target_action="open_box", max_occurrences=1,
        ),
    )
    first = adapter.run(actions=["open_box"])
    second = adapter.run(actions=["open_box"])
    assert first["status"] == "injected_fault"
    assert second["status"] == "ok"
    assert len(env.calls) == 1


def test_disabled_policy_never_injects():
    env = FakeEnvironment()
    adapter = FaultInjectionAdapter(env, FaultInjectionPolicy(enabled=False))
    assert adapter.run(actions=["open_box"])["status"] == "ok"
    assert len(env.calls) == 1


def test_rejects_multi_action_calls_to_avoid_bypassing_canonical_dispatch():
    adapter = FaultInjectionAdapter(
        FakeEnvironment(),
        FaultInjectionPolicy(enabled=True, fault_type=FaultType.ACTION_FAILURE),
    )
    with pytest.raises(ValueError, match="exactly one action"):
        adapter.run(actions=["open_box", "take_key"])


def test_rejects_unsupported_trigger_step():
    with pytest.raises(ValueError, match="trigger_step=1"):
        FaultInjectionAdapter(
            FakeEnvironment(),
            FaultInjectionPolicy(
                enabled=True, fault_type=FaultType.ACTION_FAILURE,
                trigger_step=2,
            ),
        )


def test_unknown_outcome_always_requires_reconciliation():
    decision = RecoveryController().decide(
        policy=RecoveryPolicy(),
        fault_type=FaultType.OBSERVATION_INTERRUPTION,
        attempt=1,
        outcome_known=False,
        action_executed=None,
    )
    assert decision.action == RecoveryAction.RECONCILE


def test_known_non_execution_retryable_fault_can_retry():
    decision = RecoveryController().decide(
        policy=RecoveryPolicy(max_attempts=2),
        fault_type=FaultType.ACTION_FAILURE,
        attempt=1,
        outcome_known=True,
        action_executed=False,
    )
    assert decision.action == RecoveryAction.RETRY


def test_retry_budget_is_respected():
    controller = RecoveryController()
    policy = RecoveryPolicy(max_attempts=2)
    assert controller.decide(
        policy=policy, fault_type=FaultType.ACTION_FAILURE, attempt=1,
        outcome_known=True, action_executed=False,
    ).action == RecoveryAction.RETRY
    assert controller.decide(
        policy=policy, fault_type=FaultType.ACTION_FAILURE, attempt=2,
        outcome_known=True, action_executed=False,
    ).action == RecoveryAction.RETRY
    assert controller.decide(
        policy=policy, fault_type=FaultType.ACTION_FAILURE, attempt=3,
        outcome_known=True, action_executed=False,
    ).action == RecoveryAction.STOP


def test_non_retryable_fault_stops():
    decision = RecoveryController().decide(
        policy=RecoveryPolicy(),
        fault_type=FaultType.OBSERVATION_INTERRUPTION,
        attempt=1,
        outcome_known=True,
        action_executed=False,
    )
    assert decision.action == RecoveryAction.STOP


def test_executed_action_is_never_replayed():
    decision = RecoveryController().decide(
        policy=RecoveryPolicy(),
        fault_type=FaultType.ACTION_FAILURE,
        attempt=1,
        outcome_known=True,
        action_executed=True,
    )
    assert decision.action == RecoveryAction.STOP


def test_disabled_recovery_stops():
    decision = RecoveryController().decide(
        policy=RecoveryPolicy(enabled=False),
        fault_type=FaultType.ACTION_FAILURE,
        attempt=1,
        outcome_known=True,
        action_executed=False,
    )
    assert decision.action == RecoveryAction.STOP


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5])
def test_invalid_attempt_is_rejected(attempt):
    with pytest.raises(ValueError):
        RecoveryController().decide(
            policy=RecoveryPolicy(),
            fault_type=FaultType.ACTION_FAILURE,
            attempt=attempt,
            outcome_known=True,
            action_executed=False,
        )


def test_fault_records_are_immutable_snapshot():
    env = FakeEnvironment()
    adapter = FaultInjectionAdapter(
        env,
        FaultInjectionPolicy(enabled=True, fault_type=FaultType.ACTION_FAILURE),
    )
    adapter.run(actions=["open_box"])
    records = adapter.records
    assert len(records) == 1
    assert records[0].occurrence == 1


@pytest.mark.parametrize("fault_type", [
    FaultType.ACTION_FAILURE,
    FaultType.TEMPORARY_UNAVAILABILITY,
])
def test_retryable_known_non_execution_faults_are_retryable(fault_type):
    decision = RecoveryController().decide(
        policy=RecoveryPolicy(max_attempts=2),
        fault_type=fault_type,
        attempt=1,
        outcome_known=True,
        action_executed=False,
    )
    assert decision.action == RecoveryAction.RETRY
