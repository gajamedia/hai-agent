"""CLI for Step 39.4 scenario-provider readiness preflight."""
from __future__ import annotations

import argparse
import json
import sys

from agent.llm.p0_scenario_readiness import (
    P0ScenarioReadinessError,
    run_provider_preflight,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a P0 scenario provider without executing actions."
    )
    parser.add_argument("--provider", required=True, help="module.path:factory_name")
    parser.add_argument("--report", default="artifacts/p0_scenario_readiness.json")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--runs-per-task", type=int, default=3)
    args = parser.parse_args(argv)

    if args.tasks < 1 or args.runs_per_task < 1:
        parser.error("--tasks and --runs-per-task must be positive integers")

    try:
        report = run_provider_preflight(
            args.provider,
            expected_tasks=args.tasks,
            runs_per_task=args.runs_per_task,
        )
        write_report(report, args.report)
    except P0ScenarioReadinessError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2, ensure_ascii=False))
        return 2

    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    print(f"Report written: {args.report}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
