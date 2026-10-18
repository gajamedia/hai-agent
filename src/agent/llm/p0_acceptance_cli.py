"""Command line entry point for the P0 acceptance evidence evaluator."""
from __future__ import annotations

import argparse
import json
import sys

from agent.llm.p0_evidence_jsonl import (
    P0EvidenceLoadError,
    evaluate_p0_evidence_file,
    write_p0_report_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate JSONL runtime evidence against the Step 39 P0 gate."
    )
    parser.add_argument("evidence_jsonl", help="JSONL file containing one record per execution run")
    parser.add_argument("--report", help="Optional path to write the full JSON report")
    args = parser.parse_args(argv)

    try:
        report = evaluate_p0_evidence_file(args.evidence_jsonl)
    except P0EvidenceLoadError as exc:
        print(f"EVIDENCE ERROR: {exc}", file=sys.stderr)
        return 2

    if args.report:
        write_p0_report_json(report, args.report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
