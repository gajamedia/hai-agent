"""CLI for executing the Step 39 benchmark through an explicit project adapter."""
from __future__ import annotations

import argparse
import json
import sys

from agent.llm.p0_acceptance_gate import P0AcceptanceThresholds
from agent.llm.p0_runtime_benchmark import P0BenchmarkError, load_adapter, run_p0_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 30-task x 3-run P0 benchmark through a project adapter that "
            "executes the actual runtime and reports measured evidence."
        )
    )
    parser.add_argument("--adapter", required=True, help="Python factory as module.path:factory_name")
    parser.add_argument("--evidence", default="artifacts/p0_runtime_evidence.jsonl")
    parser.add_argument("--report", default="artifacts/p0_acceptance_report.json")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--runs-per-task", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        thresholds = P0AcceptanceThresholds(
            expected_tasks=args.tasks,
            runs_per_task=args.runs_per_task,
        )
        adapter = load_adapter(args.adapter)
        report = run_p0_benchmark(
            adapter,
            evidence_path=args.evidence,
            report_path=args.report,
            thresholds=thresholds,
        )
    except (P0BenchmarkError, ValueError) as exc:
        print(f"BENCHMARK ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
