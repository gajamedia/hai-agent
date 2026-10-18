from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "step41_3_full_runtime_benchmark.py"
SPEC = importlib.util.spec_from_file_location("step41_3_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_cli_targets_canonical_runtime_adapter():
    args = runner.build_cli_args(Path("artifacts/test"), "contract-run")
    assert args[args.index("--adapter") + 1] == (
        "agent.llm.p0_canonical_runtime_adapter:create_adapter"
    )


def test_cli_requests_exact_30_by_3_matrix():
    args = runner.build_cli_args(Path("artifacts/test"), "contract-run")
    assert args[args.index("--tasks") + 1] == "30"
    assert args[args.index("--runs-per-task") + 1] == "3"


def test_evidence_and_report_are_separate_and_run_scoped():
    args = runner.build_cli_args(Path("artifacts/test"), "run-123")
    evidence = Path(args[args.index("--evidence") + 1])
    report = Path(args[args.index("--report") + 1])
    assert evidence.name == "evidence-run-123.jsonl"
    assert report.name == "report-run-123.json"
    assert evidence != report


def test_run_id_rejects_path_traversal_and_invalid_characters():
    for invalid in ("../oops", "run/name", "run name", ""):
        try:
            runner.build_cli_args(Path("artifacts/test"), invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid run_id to be rejected: {invalid!r}")


def test_source_path_resolves_repo_src():
    assert runner.SRC_ROOT.name == "src"
    assert runner.SRC_ROOT.parent == runner.REPO_ROOT
