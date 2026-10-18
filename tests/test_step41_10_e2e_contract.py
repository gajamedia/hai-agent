from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'step41_10_canonical_recovery_e2e.py'
SOURCE = SCRIPT.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def test_benchmark_is_valid_python():
    assert isinstance(TREE, ast.Module)


def test_uses_real_canonical_adapter_not_fake_runtime():
    assert 'P0CanonicalRuntimeAdapter' in SOURCE
    assert 'RecoveryFaultInjectionProvider' in SOURCE
    assert 'class Runtime:' not in SOURCE


def test_checks_recovery_and_completion_evidence():
    for field in ('correct_completion', 'deterministic_verification_passed', 'recoverable_failure', 'recovery_succeeded'):
        assert field in SOURCE


def test_checks_security_audit_and_trace_evidence():
    for field in ('unauthorized_high_risk_actions', 'duplicate_harmful_side_effects', 'audit_evidence_complete', 'required_approval_evidence_complete', 'trace_integrity_passed'):
        assert field in SOURCE


def test_report_outputs_are_timestamped_and_refuse_overwrite():
    assert '%Y%m%dT%H%M%SZ' in SOURCE
    assert 'Refusing to overwrite report' in SOURCE
    assert 'exist_ok=False' in SOURCE
