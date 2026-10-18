from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def validate_step_report(data):
    if data is None:
        return {"valid": None, "reason": "report_missing_or_unreadable"}
    required = ("overall_passed", "total_cases", "passed_cases", "failed_cases")
    missing = [key for key in required if key not in data]
    if missing:
        return {"valid": False, "reason": "missing_fields", "missing_fields": missing}
    passed = data["overall_passed"] is True and data["failed_cases"] == 0 and data["passed_cases"] == data["total_cases"]
    return {"valid": passed, "total_cases": data["total_cases"], "passed_cases": data["passed_cases"], "failed_cases": data["failed_cases"]}


def validate_matrix_report(data):
    result = validate_step_report(data)
    if result["valid"] is not True:
        return result
    if data.get("total_cases") != 90 or data.get("passed_cases") != 90 or data.get("failed_cases") != 0:
        return {"valid": False, "reason": "expected_30_tasks_x_3_runs_all_90_passed", **result}
    return result


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Step 41.14 final regression and closure report")
    parser.add_argument("--output", default="artifacts/step41_14_closure_report.json")
    parser.add_argument("--skip-pytest", action="store_true", help="Do not execute pytest; record as not_run")
    parser.add_argument("--matrix-report", help="Existing JSON evidence for the 30 tasks x 3 runs matrix")
    parser.add_argument("--step41-12-report", default="artifacts/step41_12_adversarial_report.json")
    parser.add_argument("--step41-13-report", default="artifacts/step41_13_adversarial_runtime_report.json")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        print(f"ERROR: output already exists; refusing to overwrite: {output}", file=sys.stderr)
        return 2

    pytest_result = {"status": "not_run"}
    if not args.skip_pytest:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], text=True, capture_output=True)
        pytest_result = {"status": "passed" if proc.returncode == 0 else "failed", "return_code": proc.returncode,
                         "stdout_tail": proc.stdout[-12000:], "stderr_tail": proc.stderr[-4000:]}

    report12 = validate_step_report(read_json(Path(args.step41_12_report)))
    report13_data = read_json(Path(args.step41_13_report))
    # Step 41.13's benchmark includes pytest_return_code and overall_passed; case counts may not be included.
    if report13_data is None:
        report13 = {"valid": None, "reason": "report_missing_or_unreadable"}
    elif "total_cases" in report13_data:
        report13 = validate_step_report(report13_data)
    else:
        passed = report13_data.get("overall_passed") is True and report13_data.get("pytest_return_code") == 0
        report13 = {"valid": passed, "overall_passed": report13_data.get("overall_passed"), "pytest_return_code": report13_data.get("pytest_return_code"),
                    "reason": "validated_using_available_step41_13_fields"}

    matrix = validate_matrix_report(read_json(Path(args.matrix_report))) if args.matrix_report else {"valid": None, "reason": "matrix_report_not_supplied"}
    criteria = {
        "full_pytest_suite": pytest_result["status"] == "passed" if pytest_result["status"] != "not_run" else None,
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
        "scope_note": "Repository tests and supplied reports only; not a production-wide safety guarantee.",
        "roadmap_boundary": "Step 41 ends at Step 41.14; no Step 41.15."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Recheck immediately before writing to preserve no-overwrite behavior.
    if output.exists():
        print(f"ERROR: output appeared during execution; refusing to overwrite: {output}", file=sys.stderr)
        return 2
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "overall_status": overall, "criteria": criteria, "pytest_status": pytest_result["status"]}, indent=2))
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
