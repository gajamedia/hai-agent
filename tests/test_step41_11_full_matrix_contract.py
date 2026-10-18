from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'step41_11_full_matrix_recovery_regression.py'
SOURCE = SCRIPT.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def test_benchmark_is_valid_python():
    assert isinstance(TREE, ast.Module)


def test_uses_real_canonical_adapter():
    assert 'P0CanonicalRuntimeAdapter' in SOURCE
    assert 'RecoveryFaultInjectionProvider' in SOURCE
    assert 'class FakeRuntime' not in SOURCE


def test_requires_full_30_task_matrix():
    assert 'len(tasks) != 30' in SOURCE
    assert 'Expected exactly 30 canonical tasks' in SOURCE


def test_requires_three_runs_per_task():
    assert 'runs_per_task != 3' in SOURCE
    assert 'adapter.execute(task, run_index)' in SOURCE


def test_checks_recovery_and_safety_evidence():
    for field in ('recoverable_failure', 'recovery_succeeded', 'unauthorized_high_risk_actions',
                  'duplicate_harmful_side_effects', 'audit_evidence_complete',
                  'required_approval_evidence_complete', 'trace_integrity_passed'):
        assert field in SOURCE


def test_report_refuses_overwrite():
    assert 'Refusing to overwrite report' in SOURCE
    assert 'exist_ok=False' in SOURCE
