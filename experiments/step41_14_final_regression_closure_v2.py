from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def normalize_failed_cases(value: Any) -> int | None:
    """Accept either a numeric count or a list of failed-case records."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return None


def validate_step_report(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"valid": None, "reason": "report_missing_or_unreadable"}

    required = ("overall_passed", "total_cases", "passed_cases", "failed_cases")
    missing = [key for key in required if key not in data]
    if missing:
        return {"valid": False, "reason": "missing_fields", "missing_fields": missing}

    failed_count = normalize_failed_cases(data["failed_cases"])
    if failed_count is None:
        return {"valid": False, "reason": "unsupported_failed_cases_format"}

    total = data["total_cases"]
    passed = data["passed_cases"]
    if not isinstance(total, int) or isinstance(total, bool):
        return {"valid": False, "reason": "total_cases_not_integer"}
    if not isinstance(passed, int) or isinstance(passed, bool):
        return {"valid": False, "reason": "passed_cases_not_integer"}

    valid = (
        data["overall_passed"] is True
        and failed_count == 0
        and passed == total
    )
    return {
        "valid": valid,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed_count,
    }


def validate_matrix_report(data: Any) -> dict[str, Any]:
    result = validate_step_report(data)
    if result.get("valid") is not True:
        return result
    if (
        result["total_cases"] != 90
        or result["passed_cases"] != 90
        or result["failed_cases"] != 0
    ):
        return {
            "valid": False,
            "reason": "expected_30_tasks_x_3_runs_all_90_passed",
            **result,
        }
    return result


def read_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 41.14 final regression and closure report (validator fix)"
    )
    parser.add_argument(
        "--output",
        default="artifacts/step41_14_closure_report_v2.json",
    )
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--matrix-report",
        help="Existing JSON evidence for the 30 tasks x 3 runs matrix",
    )
    parser.add_argument(
        "--step41-12-report",
        default="artifacts/step41_12_adversarial_report.json",
    )
    parser.add_argument(
        "--step41-13-report",
        default="artifacts/step41_13_adversarial_runtime_report.json",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        print(
            f"ERROR: output already exists; refusing to overwrite: {output}",
            file=sys.stderr,
        )
        return 2

    pytest_result: dict[str, Any] = {"status": "not_run"}
    if not args.skip_pytest:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            text=True,
            capture_output=True,
        )
        pytest_result = {
            "status": "passed" if proc.returncode == 0 else "failed",
            "return_code": proc.returncode,
            "stdout_tail": proc.stdout[-12000:],
            "stderr_tail": proc.stderr[-4000:],
        }

    report12 = validate_step_report(read_json(Path(args.step41_12_report)))

    report13_data = read_json(Path(args.step41_13_report))
    if not isinstance(report13_data, dict):
        report13 = {"valid": None, "reason": "report_missing_or_unreadable"}
    elif "total_cases" in report13_data:
        report13 = validate_step_report(report13_data)
    else:
        passed = (
            report13_data.get("overall_passed") is True
            and report13_data.get("pytest_return_code") == 0
        )
        report13 = {
            "valid": passed,
            "overall_passed": report13_data.get("overall_passed"),
            "pytest_return_code": report13_data.get("pytest_return_code"),
            "reason": "validated_using_available_step41_13_fields",
        }

    matrix = (
        validate_matrix_report(read_json(Path(args.matrix_report)))
        if args.matrix_report
        else {"valid": None, "reason": "matrix_report_not_supplied"}
    )

    criteria = {
        "full_pytest_suite": (
            pytest_result["status"] == "passed"
            if pytest_result["status"] != "not_run"
            else None
        ),
        "step41_12_report": report12["valid"],
        "step41_13_report": report13["valid"],
        "30_tasks_x_3_runs_matrix": matrix["valid"],
    }
    known_failures = any(value is False for value in criteria.values())
    all_verified = all(value is True for value in criteria.values())
    overall = "passed" if all_verified else ("failed" if known_failures else "incomplete")

    result = {
        "step": "41.14",
        "title": "Final Regression & Closure Review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "criteria": criteria,
        "pytest": pytest_result,
        "step41_12_report_validation": report12,
        "step41_13_report_validation": report13,
        "matrix_report_validation": matrix,
        "scope_note": (
            "Repository tests and supplied reports only; "
            "not a production-wide safety guarantee."
        ),
        "roadmap_boundary": "Step 41 ends at Step 41.14; no Step 41.15.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(
            f"ERROR: output appeared during execution; refusing to overwrite: {output}",
            file=sys.stderr,
        )
        return 2
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "overall_status": overall,
                "criteria": criteria,
                "pytest_status": pytest_result["status"],
            },
            indent=2,
        )
    )
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
