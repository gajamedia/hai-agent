"""Integrity manifest utilities for P0 runtime evidence JSONL.

The manifest detects accidental or later modification relative to a separately
retained manifest. It is not a digital signature and does not prove who created
an evidence file. Store the manifest independently or sign it for stronger trust.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


class EvidenceIntegrityError(ValueError):
    """Raised when evidence or its integrity manifest is invalid."""


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise EvidenceIntegrityError(f"Evidence file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceIntegrityError(f"Evidence file does not exist: {path}")
    rows = 0
    task_runs: set[tuple[str, int]] = set()
    required = {
        "task_id", "run_index", "correct_completion",
        "deterministic_verification_passed", "unauthorized_high_risk_actions",
        "duplicate_harmful_side_effects", "recoverable_failure",
        "recovery_succeeded", "audit_evidence_complete",
        "required_approval_evidence_complete", "trace_integrity_passed",
    }
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                raise EvidenceIntegrityError(f"Blank JSONL line at {line_no}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceIntegrityError(f"Invalid JSON at line {line_no}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise EvidenceIntegrityError(f"Line {line_no} must contain a JSON object")
            missing = sorted(required - record.keys())
            if missing:
                raise EvidenceIntegrityError(f"Line {line_no} missing fields: {', '.join(missing)}")
            task_id = record.get("task_id")
            run_index = record.get("run_index")
            if not isinstance(task_id, str) or not task_id.strip():
                raise EvidenceIntegrityError(f"Line {line_no} has invalid task_id")
            if isinstance(run_index, bool) or not isinstance(run_index, int) or run_index < 1:
                raise EvidenceIntegrityError(f"Line {line_no} has invalid run_index")
            pair = (task_id, run_index)
            if pair in task_runs:
                raise EvidenceIntegrityError(f"Duplicate task/run pair at line {line_no}: {pair}")
            task_runs.add(pair)
            for key in ("correct_completion", "deterministic_verification_passed", "recoverable_failure", "recovery_succeeded", "audit_evidence_complete", "required_approval_evidence_complete", "trace_integrity_passed"):
                if not isinstance(record[key], bool):
                    raise EvidenceIntegrityError(f"Line {line_no} field {key} must be boolean")
            for key in ("unauthorized_high_risk_actions", "duplicate_harmful_side_effects"):
                value = record[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise EvidenceIntegrityError(f"Line {line_no} field {key} must be a non-negative integer")
            if record["recovery_succeeded"] and not record["recoverable_failure"]:
                raise EvidenceIntegrityError(f"Line {line_no}: recovery_succeeded requires recoverable_failure")
            rows += 1
    if rows == 0:
        raise EvidenceIntegrityError("Evidence JSONL is empty")
    return {"record_count": rows, "unique_task_run_pairs": len(task_runs)}


def create_manifest(evidence_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Validate JSONL and write a SHA-256 manifest atomically."""
    evidence = Path(evidence_path)
    stats = _validate_jsonl(evidence)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "algorithm": "sha256",
        "evidence_filename": evidence.name,
        "evidence_size_bytes": evidence.stat().st_size,
        "evidence_sha256": sha256_file(evidence),
        **stats,
    }
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return manifest


def verify_manifest(evidence_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    evidence = Path(evidence_path)
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise EvidenceIntegrityError(f"Manifest does not exist: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceIntegrityError("Manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise EvidenceIntegrityError("Manifest root must be a JSON object")
    if manifest.get("manifest_version") != MANIFEST_VERSION or manifest.get("algorithm") != "sha256":
        raise EvidenceIntegrityError("Unsupported manifest version or algorithm")
    expected_keys = {"evidence_filename", "evidence_size_bytes", "evidence_sha256", "record_count", "unique_task_run_pairs"}
    missing = sorted(expected_keys - manifest.keys())
    if missing:
        raise EvidenceIntegrityError(f"Manifest missing fields: {', '.join(missing)}")
    stats = _validate_jsonl(evidence)
    actual_hash = sha256_file(evidence)
    actual_size = evidence.stat().st_size
    checks = {
        "filename_matches": manifest["evidence_filename"] == evidence.name,
        "size_matches": manifest["evidence_size_bytes"] == actual_size,
        "sha256_matches": manifest["evidence_sha256"] == actual_hash,
        "record_count_matches": manifest["record_count"] == stats["record_count"],
        "unique_task_run_pairs_match": manifest["unique_task_run_pairs"] == stats["unique_task_run_pairs"],
    }
    return {"valid": all(checks.values()), "checks": checks, "actual_sha256": actual_hash, **stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or verify a SHA-256 manifest for P0 evidence JSONL")
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal", help="validate evidence and create a manifest")
    seal.add_argument("evidence")
    seal.add_argument("manifest")
    verify = sub.add_parser("verify", help="validate evidence against a manifest")
    verify.add_argument("evidence")
    verify.add_argument("manifest")
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            result = {"valid": True, "manifest": create_manifest(args.evidence, args.manifest)}
        else:
            result = verify_manifest(args.evidence, args.manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("valid") else 1
    except EvidenceIntegrityError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
