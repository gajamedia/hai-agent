from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.llm.p0_acceptance_gate import P0AcceptanceThresholds, P0RunEvidence
from agent.llm.p0_runtime_benchmark import (
    P0BenchmarkError,
    P0BenchmarkTask,
    load_adapter,
    run_p0_benchmark,
)


class PassingTestAdapter:
    """Test fixture only; never use this adapter as production runtime evidence."""

    def tasks(self):
        return [
            P0BenchmarkTask(f"task-{i:02d}", f"test task {i}", {"case": i})
            for i in range(1, 31)
        ]

    def execute(self, task, run_index):
        return P0RunEvidence(
            task_id=task.task_id,
            run_index=run_index,
            correct_completion=True,
            deterministic_verification_passed=True,
            unauthorized_high_risk_actions=0,
            duplicate_harmful_side_effects=0,
            recoverable_failure=False,
            recovery_succeeded=False,
            audit_evidence_complete=True,
            required_approval_evidence_complete=True,
            trace_integrity_passed=True,
            notes="Unit-test fixture only; not evidence from the actual agent runtime.",
        )


def test_benchmark_runs_all_90_records_and_writes_evidence(tmp_path):
    evidence_path = tmp_path / "artifacts" / "p0_runtime_evidence.jsonl"
    report_path = tmp_path / "artifacts" / "p0_acceptance_report.json"
    report = run_p0_benchmark(
        PassingTestAdapter(), evidence_path=evidence_path, report_path=report_path
    )

    records = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
    stored_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(records) == 90
    assert report.received_runs == 90
    assert report.passed is True
    assert stored_report["passed"] is True
    assert records[0]["notes"].startswith("Unit-test fixture only")


def test_rejects_missing_task_before_any_execution(tmp_path):
    class MissingTaskAdapter(PassingTestAdapter):
        def tasks(self):
            return super().tasks()[:-1]

        def execute(self, task, run_index):
            raise AssertionError("No task should execute when the task matrix is invalid")

    with pytest.raises(P0BenchmarkError, match="exactly 30 canonical tasks"):
        run_p0_benchmark(
            MissingTaskAdapter(),
            evidence_path=tmp_path / "evidence.jsonl",
            report_path=tmp_path / "report.json",
        )
    assert not (tmp_path / "evidence.jsonl").exists()


def test_rejects_evidence_identity_mismatch_and_does_not_write_files(tmp_path):
    class MismatchAdapter(PassingTestAdapter):
        def execute(self, task, run_index):
            row = super().execute(task, run_index)
            return P0RunEvidence(
                task_id="task-30", run_index=run_index,
                correct_completion=row.correct_completion,
                deterministic_verification_passed=row.deterministic_verification_passed,
            )

    with pytest.raises(P0BenchmarkError, match="identity mismatch"):
        run_p0_benchmark(
            MismatchAdapter(),
            evidence_path=tmp_path / "evidence.jsonl",
            report_path=tmp_path / "report.json",
        )
    assert not (tmp_path / "evidence.jsonl").exists()
    assert not (tmp_path / "report.json").exists()


def test_rejects_runtime_adapter_exception_and_does_not_write_files(tmp_path):
    class BrokenAdapter(PassingTestAdapter):
        def execute(self, task, run_index):
            if task.task_id == "task-02" and run_index == 2:
                raise RuntimeError("simulated adapter failure")
            return super().execute(task, run_index)

    with pytest.raises(P0BenchmarkError, match="simulated adapter failure"):
        run_p0_benchmark(
            BrokenAdapter(),
            evidence_path=tmp_path / "evidence.jsonl",
            report_path=tmp_path / "report.json",
        )
    assert not (tmp_path / "evidence.jsonl").exists()
    assert not (tmp_path / "report.json").exists()


def test_adapter_loader_rejects_bad_factory_path():
    with pytest.raises(P0BenchmarkError, match="module.path:factory_name"):
        load_adapter("not-a-factory-path")


def test_thresholds_can_be_reduced_for_controlled_tests(tmp_path):
    class TwoTaskAdapter:
        def tasks(self):
            return [
                P0BenchmarkTask("task-01", "task one", {}),
                P0BenchmarkTask("task-02", "task two", {}),
            ]

        def execute(self, task, run_index):
            return P0RunEvidence(
                task_id=task.task_id,
                run_index=run_index,
                correct_completion=True,
                deterministic_verification_passed=True,
            )

    report = run_p0_benchmark(
        TwoTaskAdapter(),
        evidence_path=tmp_path / "evidence.jsonl",
        report_path=tmp_path / "report.json",
        thresholds=P0AcceptanceThresholds(expected_tasks=2, runs_per_task=2),
    )
    assert report.received_runs == 4
    assert report.expected_runs == 4
    assert report.passed is True


def test_wraps_task_provider_exception(tmp_path):
    class BrokenTaskProvider(PassingTestAdapter):
        def tasks(self):
            raise RuntimeError("task registry unavailable")

    with pytest.raises(P0BenchmarkError, match="task registry unavailable"):
        run_p0_benchmark(
            BrokenTaskProvider(),
            evidence_path=tmp_path / "evidence.jsonl",
            report_path=tmp_path / "report.json",
        )
    assert not (tmp_path / "evidence.jsonl").exists()
