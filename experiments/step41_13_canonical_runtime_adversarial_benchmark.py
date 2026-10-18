"""Run Step 41.13 tests against the repository's actual canonical adapter."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "tests" / "test_step41_13_canonical_runtime_adversarial_integration.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/step41_13_adversarial_runtime_report.json")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.exists():
        print(f"ERROR: refusing to overwrite existing report: {output}", file=sys.stderr)
        return 2
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", str(TEST_PATH)],
                               cwd=ROOT, capture_output=True, text=True)
    report = {
        "step": "41.13",
        "name": "canonical_runtime_adversarial_integration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pytest_return_code": completed.returncode,
        "overall_passed": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "scope": "Actual P0CanonicalRuntimeAdapter with repository-controlled fixtures; no external services. Not a production-wide safety claim."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"output": str(output), "overall_passed": report["overall_passed"],
                      "pytest_return_code": report["pytest_return_code"], "scope": report["scope"]},
                     indent=2, ensure_ascii=False))
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
