"""Step 41.13 adversarial integration against the actual canonical adapter."""
from dataclasses import replace
import pytest

from agent.llm.p0_canonical_runtime_adapter import P0CanonicalRuntimeAdapter, P0CanonicalRuntimeAdapterError
from agent.llm.p0_recovery_fault_injection_provider import RecoveryFaultInjectionProvider
from agent.llm.p0_adversarial_room_provider import build_denied_provider, build_unavailable_approval_provider


def _task(provider, task_id="task-01"):
    return next(task for task in provider.tasks() if task.task_id == task_id)


def test_transient_fault_recovers_through_canonical_adapter(tmp_path):
    provider = RecoveryFaultInjectionProvider()
    evidence = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "recovery").execute(_task(provider), 1)
    assert evidence.correct_completion is True
    assert evidence.deterministic_verification_passed is True
    assert evidence.recoverable_failure is True
    assert evidence.recovery_succeeded is True
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.duplicate_harmful_side_effects == 0
    assert evidence.audit_evidence_complete is True
    assert evidence.required_approval_evidence_complete is True
    assert evidence.trace_integrity_passed is True


def test_denied_approval_fails_closed_on_canonical_adapter(tmp_path):
    provider = build_denied_provider()
    evidence = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "denied").execute(_task(provider), 1)
    assert evidence.correct_completion is False
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.duplicate_harmful_side_effects == 0
    assert evidence.audit_evidence_complete is True
    assert evidence.required_approval_evidence_complete is True
    assert evidence.trace_integrity_passed is True


def test_approval_checker_outage_fails_closed_on_canonical_adapter(tmp_path):
    provider = build_unavailable_approval_provider()
    evidence = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "approval-outage").execute(_task(provider), 1)
    assert evidence.correct_completion is False
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.duplicate_harmful_side_effects == 0
    assert evidence.audit_evidence_complete is True
    assert evidence.required_approval_evidence_complete is True
    assert evidence.trace_integrity_passed is True


class _ScenarioProviderProxy:
    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def tasks(self):
        return self.base.tasks()

    def build_scenario(self, task, run_index):
        return self.transform(self.base.build_scenario(task, run_index))


def test_nonrecoverable_failure_does_not_request_recovery_plan(tmp_path):
    base = RecoveryFaultInjectionProvider()

    def transform(scenario):
        def never_recover(_report, _state):
            return False
        def forbidden_plan_factory(*_args):
            raise AssertionError("recovery plan must not be requested for non-recoverable failure")
        return replace(scenario, is_recoverable_failure=never_recover, recovery_plan_factory=forbidden_plan_factory)

    provider = _ScenarioProviderProxy(base, transform)
    evidence = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "nonrecoverable").execute(_task(provider), 1)
    assert evidence.correct_completion is False
    assert evidence.recoverable_failure is False
    assert evidence.recovery_succeeded is False
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.duplicate_harmful_side_effects == 0
    assert evidence.audit_evidence_complete is True
    assert evidence.trace_integrity_passed is True


def test_recoverable_failure_without_recovery_factory_is_rejected(tmp_path):
    base = RecoveryFaultInjectionProvider()
    provider = _ScenarioProviderProxy(base, lambda s: replace(s, recovery_plan_factory=None))
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "missing-factory")
    with pytest.raises(P0CanonicalRuntimeAdapterError, match="no explicit recovery_plan_factory"):
        adapter.execute(_task(provider), 1)


def test_recovery_factory_returning_none_is_rejected(tmp_path):
    base = RecoveryFaultInjectionProvider()
    provider = _ScenarioProviderProxy(base, lambda s: replace(s, recovery_plan_factory=lambda *_args: None))
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "none-plan")
    with pytest.raises(P0CanonicalRuntimeAdapterError, match="recovery_plan_factory returned None"):
        adapter.execute(_task(provider), 1)
