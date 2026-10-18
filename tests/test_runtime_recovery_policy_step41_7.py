from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from agent.llm.recovery_policy import FailureCategory, FailureContext, RecoveryAction
from agent.llm.runtime_recovery_policy import (
    RuntimeRecoveryPolicyIntegrationError,
    bind_recovery_policy,
)


@dataclass(frozen=True)
class FakeScenario:
    plan: object = field(default_factory=object)
    is_recoverable_failure: object = lambda report, state: True
    recovery_plan_factory: object = lambda scenario, report, state: {"recovered": True}
    metadata: object = None


def ctx(category=FailureCategory.STATE_CONFLICT, **kwargs):
    defaults = dict(
        category=category,
        attempt=1,
        max_attempts=3,
        idempotent=False,
        state_changed=False,
        state_verified=False,
        recovery_available=True,
        approval_required=False,
        approval_granted=False,
    )
    defaults.update(kwargs)
    return FailureContext(**defaults)


def test_recover_decision_allows_explicit_recovery_plan():
    sink = {}
    scenario = bind_recovery_policy(FakeScenario(), lambda s, r, st: ctx(), decision_sink=sink)
    assert scenario.is_recoverable_failure({"success": False}, {}) is True
    assert sink["decision"].action == RecoveryAction.RECOVER
    assert scenario.recovery_plan_factory(scenario, {}, {}) == {"recovered": True}


def test_retry_decision_is_not_misrepresented_as_recovery():
    sink = {}
    scenario = bind_recovery_policy(
        FakeScenario(),
        lambda s, r, st: ctx(FailureCategory.TRANSIENT, idempotent=True),
        decision_sink=sink,
    )
    assert scenario.is_recoverable_failure({"success": False}, {}) is False
    assert sink["decision"].action == RecoveryAction.RETRY
    with pytest.raises(RuntimeRecoveryPolicyIntegrationError):
        scenario.recovery_plan_factory(scenario, {}, {})


def test_unknown_failure_fails_closed():
    scenario = bind_recovery_policy(
        FakeScenario(), lambda s, r, st: ctx(FailureCategory.UNKNOWN)
    )
    assert scenario.is_recoverable_failure({}, {}) is False


def test_missing_approval_stops_recovery():
    scenario = bind_recovery_policy(
        FakeScenario(),
        lambda s, r, st: ctx(approval_required=True, approval_granted=False),
    )
    assert scenario.is_recoverable_failure({}, {}) is False


def test_existing_recoverability_guard_is_preserved():
    scenario = bind_recovery_policy(
        FakeScenario(is_recoverable_failure=lambda report, state: False),
        lambda s, r, st: ctx(),
    )
    assert scenario.is_recoverable_failure({}, {}) is False


def test_invalid_context_type_raises():
    scenario = bind_recovery_policy(FakeScenario(), lambda s, r, st: SimpleNamespace())
    with pytest.raises(RuntimeRecoveryPolicyIntegrationError):
        scenario.is_recoverable_failure({}, {})


def test_requires_explicit_recovery_plan_factory():
    with pytest.raises(RuntimeRecoveryPolicyIntegrationError):
        bind_recovery_policy(FakeScenario(recovery_plan_factory=None), lambda s, r, st: ctx())


def test_metadata_marks_adapter_retry_limitation():
    scenario = bind_recovery_policy(FakeScenario(metadata={"existing": True}), lambda s, r, st: ctx())
    assert scenario.metadata["existing"] is True
    assert scenario.metadata["recovery_policy_integration"] == "step41_7"
    assert scenario.metadata["retry_execution_supported_by_adapter"] is False


def test_permanent_failure_stops():
    scenario = bind_recovery_policy(
        FakeScenario(), lambda s, r, st: ctx(FailureCategory.PERMANENT)
    )
    assert scenario.is_recoverable_failure({}, {}) is False
