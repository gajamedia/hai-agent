from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SPEC = importlib.util.spec_from_file_location(
    "step41_5_runner", ROOT / "scripts" / "step41_5_recovery_fault_injection.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from agent.llm.p0_recovery_fault_injection_provider import (
    FAILURE_ACTION,
    RecoveryFaultInjectionProvider,
)


def test_provider_preserves_room_task_matrix():
    provider = RecoveryFaultInjectionProvider()
    tasks = tuple(provider.tasks())
    assert len(tasks) == 30
    assert len({task.task_id for task in tasks}) == 30


def test_scenario_injects_registered_failure_and_explicit_recovery_plan():
    provider = RecoveryFaultInjectionProvider()
    task = next(iter(provider.tasks()))
    scenario = provider.build_scenario(task, 1)
    assert FAILURE_ACTION in scenario.agent.actions
    assert scenario.plan.steps[0].action == FAILURE_ACTION
    assert scenario.action_risks[FAILURE_ACTION] < scenario.risk_threshold
    assert callable(scenario.is_recoverable_failure)
    assert callable(scenario.recovery_plan_factory)
    assert scenario.metadata["recovery_expected"] is True


def test_recovery_factory_returns_original_goal_plan():
    provider = RecoveryFaultInjectionProvider()
    task = next(iter(provider.tasks()))
    scenario = provider.build_scenario(task, 1)
    recovered = scenario.recovery_plan_factory(scenario, object(), scenario.environment.observe())
    assert recovered is not None
    assert recovered.steps
    assert recovered.steps[0].action != FAILURE_ACTION
    assert recovered.goal == scenario.goal.description


def test_report_requires_all_90_recoveries(tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"
    acceptance_path = tmp_path / "acceptance.json"
    rows = []
    for task_number in range(1, 31):
        for run_index in (1, 2, 3):
            rows.append({
                "task_id": f"task-{task_number:02d}",
                "run_index": run_index,
                "correct_completion": True,
                "deterministic_verification_passed": True,
                "recoverable_failure": True,
                "recovery_succeeded": True,
                "unauthorized_high_risk_actions": 0,
                "duplicate_harmful_side_effects": 0,
                "audit_evidence_complete": True,
                "required_approval_evidence_complete": True,
                "trace_integrity_passed": True,
                "notes": "recovery=True",
            })
    evidence_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    acceptance_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    summary = runner._summarize(evidence_path, acceptance_path, "test-run")
    assert summary["received_runs"] == 90
    assert summary["recoverable_failures_detected"] == 90
    assert summary["recovery_success_rate"] == 1.0
    assert summary["recovery_contract_passed"] is True


def test_report_fails_when_one_recovery_does_not_succeed(tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"
    acceptance_path = tmp_path / "acceptance.json"
    rows = []
    for task_number in range(1, 31):
        for run_index in (1, 2, 3):
            rows.append({
                "task_id": f"task-{task_number:02d}",
                "run_index": run_index,
                "correct_completion": True,
                "deterministic_verification_passed": True,
                "recoverable_failure": True,
                "recovery_succeeded": not (task_number == 30 and run_index == 3),
                "unauthorized_high_risk_actions": 0,
                "duplicate_harmful_side_effects": 0,
                "audit_evidence_complete": True,
                "required_approval_evidence_complete": True,
                "trace_integrity_passed": True,
                "notes": "recovery=False" if task_number == 30 and run_index == 3 else "recovery=True",
            })
    evidence_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    acceptance_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    summary = runner._summarize(evidence_path, acceptance_path, "test-run")
    assert summary["recovery_success_rate"] < 1.0
    assert summary["recovery_contract_passed"] is False


def test_run_id_validation_blocks_path_traversal():
    assert runner._valid_run_id("recovery-001")
    assert not runner._valid_run_id("../recovery")
    assert not runner._valid_run_id("recovery/001")
