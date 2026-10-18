from __future__ import annotations

import json

import pytest

from agent.llm.p0_evidence_jsonl import (
    P0EvidenceLoadError,
    evaluate_p0_evidence_file,
    load_p0_evidence_jsonl,
    write_p0_report_json,
)
from agent.llm.p0_acceptance_gate import P0AcceptanceThresholds


def record(task: int, run: int, **overrides):
    value = {
        "task_id": f"task-{task:02d}",
        "run_index": run,
        "correct_completion": True,
        "deterministic_verification_passed": True,
        "unauthorized_high_risk_actions": 0,
        "duplicate_harmful_side_effects": 0,
        "recoverable_failure": False,
        "recovery_succeeded": False,
        "audit_evidence_complete": True,
        "required_approval_evidence_complete": True,
        "trace_integrity_passed": True,
    }
    value.update(overrides)
    return value


def small_thresholds():
    return P0AcceptanceThresholds(expected_tasks=2, runs_per_task=2)


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )


def test_loads_jsonl_and_ignores_blank_lines(tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text(
        json.dumps(record(1, 1)) + "\n\n" + json.dumps(record(1, 2)) + "\n",
        encoding="utf-8",
    )
    loaded = load_p0_evidence_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].task_id == "task-01"


def test_valid_small_evidence_file_passes_and_writes_report(tmp_path):
    path = tmp_path / "evidence.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(path, [record(task, run) for task in (1, 2) for run in (1, 2)])
    report = evaluate_p0_evidence_file(path, thresholds=small_thresholds())
    assert report.passed is True
    write_p0_report_json(report, report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["received_runs"] == 4


def test_malformed_json_fails_closed(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"task_id":\n', encoding="utf-8")
    with pytest.raises(P0EvidenceLoadError, match="Line 1: invalid JSON"):
        load_p0_evidence_jsonl(path)


def test_missing_required_field_is_rejected(tmp_path):
    path = tmp_path / "missing.jsonl"
    row = record(1, 1)
    del row["deterministic_verification_passed"]
    write_jsonl(path, [row])
    with pytest.raises(P0EvidenceLoadError, match="missing required fields"):
        load_p0_evidence_jsonl(path)


def test_unknown_fields_are_rejected_to_catch_schema_drift(tmp_path):
    path = tmp_path / "unknown.jsonl"
    write_jsonl(path, [record(1, 1, approved_by_llm=True)])
    with pytest.raises(P0EvidenceLoadError, match="unknown fields"):
        load_p0_evidence_jsonl(path)


def test_boolean_like_values_are_not_coerced(tmp_path):
    path = tmp_path / "bad-bool.jsonl"
    write_jsonl(path, [record(1, 1, correct_completion=1)])
    with pytest.raises(P0EvidenceLoadError, match="explicit boolean"):
        load_p0_evidence_jsonl(path)


def test_missing_evidence_does_not_pass(tmp_path):
    path = tmp_path / "partial.jsonl"
    write_jsonl(path, [record(1, 1)])
    report = evaluate_p0_evidence_file(path, thresholds=small_thresholds())
    assert report.passed is False
    assert report.errors


def test_audit_failure_in_file_fails_acceptance(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_jsonl(
        path,
        [record(task, run, audit_evidence_complete=False if (task, run) == (1, 1) else True)
         for task in (1, 2) for run in (1, 2)],
    )
    report = evaluate_p0_evidence_file(path, thresholds=small_thresholds())
    assert report.passed is False
    assert not next(c for c in report.checks if c.name == "audit_evidence").passed


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(P0EvidenceLoadError, match="Could not read evidence file"):
        load_p0_evidence_jsonl(tmp_path / "does-not-exist.jsonl")
