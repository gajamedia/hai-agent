"""Load JSONL run evidence and produce a fail-closed P0 acceptance report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.llm.p0_acceptance_gate import (
    P0AcceptanceReport,
    P0AcceptanceThresholds,
    P0RunEvidence,
    evaluate_p0_acceptance,
)


_REQUIRED = {
    "task_id",
    "run_index",
    "correct_completion",
    "deterministic_verification_passed",
    "unauthorized_high_risk_actions",
    "duplicate_harmful_side_effects",
    "recoverable_failure",
    "recovery_succeeded",
    "audit_evidence_complete",
    "required_approval_evidence_complete",
    "trace_integrity_passed",
}
_ALLOWED = _REQUIRED | {"notes"}


class P0EvidenceLoadError(ValueError):
    """Raised when a JSONL evidence file is malformed or has an invalid schema."""


def _parse_record(value: Any, line_number: int) -> P0RunEvidence:
    if not isinstance(value, dict):
        raise P0EvidenceLoadError(f"Line {line_number}: each JSONL record must be an object.")
    missing = sorted(_REQUIRED - value.keys())
    unknown = sorted(value.keys() - _ALLOWED)
    if missing:
        raise P0EvidenceLoadError(
            f"Line {line_number}: missing required fields: {', '.join(missing)}."
        )
    if unknown:
        raise P0EvidenceLoadError(
            f"Line {line_number}: unknown fields: {', '.join(unknown)}."
        )
    try:
        return P0RunEvidence(**value)
    except (TypeError, ValueError) as exc:
        raise P0EvidenceLoadError(f"Line {line_number}: invalid evidence: {exc}") from exc


def load_p0_evidence_jsonl(path: str | Path) -> tuple[P0RunEvidence, ...]:
    """Read JSONL evidence; blank lines are ignored, malformed rows fail closed."""
    evidence_path = Path(path)
    rows: list[P0RunEvidence] = []
    try:
        with evidence_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise P0EvidenceLoadError(
                        f"Line {line_number}: invalid JSON: {exc.msg}."
                    ) from exc
                rows.append(_parse_record(record, line_number))
    except OSError as exc:
        raise P0EvidenceLoadError(f"Could not read evidence file '{evidence_path}': {exc}") from exc
    return tuple(rows)


def evaluate_p0_evidence_file(
    path: str | Path,
    *,
    thresholds: P0AcceptanceThresholds = P0AcceptanceThresholds(),
) -> P0AcceptanceReport:
    """Load evidence and evaluate it. Invalid file/schema raises, never silently passes."""
    evidence = load_p0_evidence_jsonl(path)
    return evaluate_p0_acceptance(evidence, thresholds=thresholds)


def write_p0_report_json(report: P0AcceptanceReport, path: str | Path) -> None:
    """Write deterministic, machine-readable report JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "P0EvidenceLoadError",
    "evaluate_p0_evidence_file",
    "load_p0_evidence_jsonl",
    "write_p0_report_json",
]
