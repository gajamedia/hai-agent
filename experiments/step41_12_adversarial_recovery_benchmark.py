"""Run isolated adversarial recovery contract scenarios and emit JSON report."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Allow direct execution from repository root or experiments directory.
ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_step41_12_adversarial_recovery_contract import (  # noqa: E402
    FailureKind,
    RecoveryInput,
    decide_recovery,
)


SCENARIOS = [
    ("permanent_failure_no_recovery", RecoveryInput(FailureKind.PERMANENT), "stop", False),
    ("ambiguous_side_effect_reconcile", RecoveryInput(FailureKind.AMBIGUOUS_SIDE_EFFECT), "reconcile", False),
    ("approval_missing", RecoveryInput(FailureKind.APPROVAL_MISSING), "stop", False),
    ("approval_rejected", RecoveryInput(FailureKind.APPROVAL_REJECTED), "stop", False),
    ("recovery_failed", RecoveryInput(FailureKind.RECOVERY_FAILED), "stop", False),
    ("goal_unsatisfied", RecoveryInput(FailureKind.GOAL_UNSATISFIED), "stop", False),
    ("audit_incomplete", RecoveryInput(
        FailureKind.TRANSIENT, goal_satisfied_after_recovery=True, audit_complete=False
    ), "stop", False),
    ("trace_incomplete", RecoveryInput(
        FailureKind.TRANSIENT, goal_satisfied_after_recovery=True, trace_complete=False
    ), "stop", False),
    ("success_with_complete_evidence", RecoveryInput(
        FailureKind.TRANSIENT, goal_satisfied_after_recovery=True
    ), "complete", True),
    ("recovery_not_available", RecoveryInput(
        FailureKind.TRANSIENT, recovery_available=False, goal_satisfied_after_recovery=False
    ), "stop", False),
    ("approval_required_missing", RecoveryInput(
        FailureKind.TRANSIENT, goal_satisfied_after_recovery=True,
        approval_required=True, approval_present=False, approval_accepted=False
    ), "stop", False),
    ("approval_required_rejected", RecoveryInput(
        FailureKind.TRANSIENT, goal_satisfied_after_recovery=True,
        approval_required=True, approval_present=True, approval_accepted=False
    ), "stop", False),
]


def build_report() -> dict:
    results = []
    for name, case, expected_action, expected_success in SCENARIOS:
        decision = decide_recovery(case)
        passed = decision.action == expected_action and decision.success is expected_success
        results.append({
            "scenario": name,
            "expected_action": expected_action,
            "actual_action": decision.action,
            "expected_success": expected_success,
            "actual_success": decision.success,
            "safe_stop": decision.safe_stop,
            "reason": decision.reason,
            "passed": passed,
        })
    failures = [item for item in results if not item["passed"]]
    return {
        "step": "41.12",
        "name": "adversarial_recovery_failure_mode_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_passed": not failures,
        "total_cases": len(results),
        "passed_cases": len(results) - len(failures),
        "failed_cases": failures,
        "scope": (
            "Isolated pure-policy contract harness only. Not a production runtime or "
            "canonical adapter end-to-end safety claim; no external side effects executed."
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/step41_12_adversarial_report.json")
    args = parser.parse_args()
    report = build_report()
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.exists():
        print(f"ERROR: refusing to overwrite existing report: {output}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "overall_passed": report["overall_passed"],
        "total_cases": report["total_cases"],
        "passed_cases": report["passed_cases"],
        "failed_cases": len(report["failed_cases"]),
        "scope": report["scope"],
    }, indent=2, ensure_ascii=False))
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
