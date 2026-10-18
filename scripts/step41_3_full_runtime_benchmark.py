"""Step 41.3: execute the full 30-task x 3-run P0 benchmark via canonical runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_ADAPTER = "agent.llm.p0_canonical_runtime_adapter:create_adapter"
DEFAULT_PROVIDER = "agent.llm.p0_room_scenario_provider:build_provider"


def build_cli_args(output_dir: Path, run_id: str) -> list[str]:
    """Build explicit CLI arguments without modifying existing evidence files."""
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
        raise ValueError("run_id may contain only letters, digits, '-' and '_'.")
    return [
        "--adapter", DEFAULT_ADAPTER,
        "--tasks", "30",
        "--runs-per-task", "3",
        "--evidence", str(output_dir / f"evidence-{run_id}.jsonl"),
        "--report", str(output_dir / f"report-{run_id}.json"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Step 41.3: 30 tasks x 3 repetitions using the canonical runtime."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/step41_3",
        help="Directory for unique evidence/report files (default: artifacts/step41_3).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional unique run label. If omitted, a UTC timestamp is used.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cli_args = build_cli_args(output_dir, run_id)
    evidence_path = Path(cli_args[cli_args.index("--evidence") + 1])
    report_path = Path(cli_args[cli_args.index("--report") + 1])

    # Fail closed rather than overwrite prior evidence.
    collisions = [path for path in (evidence_path, report_path) if path.exists()]
    if collisions:
        print("ERROR: output already exists; choose another --run-id:")
        for path in collisions:
            print(f"  {path}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    # Pin this step to the reviewed Room fixture; do not inherit another provider silently.
    os.environ["HAI_P0_SCENARIO_PROVIDER"] = DEFAULT_PROVIDER

    print("STEP 41.3 — CANONICAL RUNTIME BENCHMARK")
    print("Scope: controlled Room P0 fixture; 30 tasks x 3 runs = 90 executions")
    print(f"Provider: {os.environ['HAI_P0_SCENARIO_PROVIDER']}")
    print(f"Evidence: {evidence_path}")
    print(f"Report:   {report_path}")
    print("This is a fixture benchmark, not production-domain generalization evidence.")

    from agent.llm.p0_runtime_benchmark_cli import main as benchmark_main

    return benchmark_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
