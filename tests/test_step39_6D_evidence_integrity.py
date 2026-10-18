import json
from pathlib import Path

import pytest

from agent.llm.p0_evidence_integrity import (
    EvidenceIntegrityError,
    create_manifest,
    verify_manifest,
)


def record(task_id="task-01", run_index=1):
    return {
        "task_id": task_id,
        "run_index": run_index,
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


def write_jsonl(path: Path, *records):
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")


def test_create_and_verify_manifest(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    manifest = tmp_path / "evidence.manifest.json"
    write_jsonl(evidence, record(), record("task-01", 2))
    created = create_manifest(evidence, manifest)
    verified = verify_manifest(evidence, manifest)
    assert created["record_count"] == 2
    assert verified["valid"] is True
    assert all(verified["checks"].values())


def test_modified_evidence_is_detected(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    manifest = tmp_path / "manifest.json"
    write_jsonl(evidence, record())
    create_manifest(evidence, manifest)
    changed = record()
    changed["correct_completion"] = False
    write_jsonl(evidence, changed)
    result = verify_manifest(evidence, manifest)
    assert result["valid"] is False
    assert result["checks"]["sha256_matches"] is False


def test_duplicate_task_run_pair_is_rejected(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    write_jsonl(evidence, record(), record())
    with pytest.raises(EvidenceIntegrityError, match="Duplicate task/run pair"):
        create_manifest(evidence, tmp_path / "manifest.json")


def test_missing_required_field_is_rejected(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    bad = record()
    del bad["trace_integrity_passed"]
    write_jsonl(evidence, bad)
    with pytest.raises(EvidenceIntegrityError, match="missing fields"):
        create_manifest(evidence, tmp_path / "manifest.json")


def test_malformed_jsonl_is_rejected(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text('{"task_id":\n', encoding="utf-8")
    with pytest.raises(EvidenceIntegrityError, match="Invalid JSON"):
        create_manifest(evidence, tmp_path / "manifest.json")


def test_missing_manifest_is_rejected(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    write_jsonl(evidence, record())
    with pytest.raises(EvidenceIntegrityError, match="Manifest does not exist"):
        verify_manifest(evidence, tmp_path / "missing.json")
