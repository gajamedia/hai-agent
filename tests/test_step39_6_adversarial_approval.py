from __future__ import annotations

import pytest

from agent.llm.p0_canonical_runtime_adapter import P0CanonicalRuntimeAdapter
from agent.llm.p0_adversarial_room_provider import (
    build_approved_provider,
    build_denied_provider,
    build_unavailable_approval_provider,
)


def _task_01(provider):
    return next(task for task in provider.tasks() if task.task_id == "task-01")


def test_high_risk_action_is_blocked_when_approval_is_denied(tmp_path):
    provider = build_denied_provider()
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "denied")

    evidence = adapter.execute(_task_01(provider), 1)

    assert evidence.correct_completion is False
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.audit_evidence_complete is True
    assert evidence.required_approval_evidence_complete is True
    assert evidence.trace_integrity_passed is True
    assert evidence.duplicate_harmful_side_effects == 0
    assert "runtime_success=False" in evidence.notes


def test_high_risk_action_executes_when_approval_is_explicitly_granted(tmp_path):
    provider = build_approved_provider()
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "approved")

    evidence = adapter.execute(_task_01(provider), 1)

    assert evidence.correct_completion is True
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.audit_evidence_complete is True
    assert evidence.required_approval_evidence_complete is True
    assert evidence.trace_integrity_passed is True
    assert evidence.duplicate_harmful_side_effects == 0


def test_approval_checker_exception_fails_closed(tmp_path):
    provider = build_unavailable_approval_provider()
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "approval-outage")

    evidence = adapter.execute(_task_01(provider), 1)

    assert evidence.correct_completion is False
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.audit_evidence_complete is True
    assert evidence.required_approval_evidence_complete is True
    assert evidence.trace_integrity_passed is True
    assert evidence.duplicate_harmful_side_effects == 0


def test_every_adversarial_provider_uses_a_high_risk_approval_guard():
    for factory in (
        build_denied_provider,
        build_approved_provider,
        build_unavailable_approval_provider,
    ):
        provider = factory()
        task = _task_01(provider)
        scenario = provider.build_scenario(task, 1)
        assert "unlock_door" in scenario.approval_required_actions
        assert scenario.action_risks["unlock_door"] >= scenario.risk_threshold
        assert callable(scenario.approval_checker)
