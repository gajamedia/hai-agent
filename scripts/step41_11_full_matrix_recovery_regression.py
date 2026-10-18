from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent.llm.p0_canonical_runtime_adapter import P0CanonicalRuntimeAdapter
from agent.llm.p0_recovery_fault_injection_provider import RecoveryFaultInjectionProvider


def _row(evidence):
    row = {
        'task_id': str(evidence.task_id),
        'run_index': int(evidence.run_index),
        'correct_completion': bool(evidence.correct_completion),
        'deterministic_verification_passed': bool(evidence.deterministic_verification_passed),
        'recoverable_failure': bool(evidence.recoverable_failure),
        'recovery_succeeded': bool(evidence.recovery_succeeded),
        'unauthorized_high_risk_actions': int(evidence.unauthorized_high_risk_actions),
        'duplicate_harmful_side_effects': int(evidence.duplicate_harmful_side_effects),
        'audit_evidence_complete': bool(evidence.audit_evidence_complete),
        'required_approval_evidence_complete': bool(evidence.required_approval_evidence_complete),
        'trace_integrity_passed': bool(evidence.trace_integrity_passed),
        'notes': str(evidence.notes),
    }
    checks = [
        row['correct_completion'], row['deterministic_verification_passed'],
        row['recoverable_failure'], row['recovery_succeeded'],
        row['unauthorized_high_risk_actions'] == 0,
        row['duplicate_harmful_side_effects'] == 0,
        row['audit_evidence_complete'], row['required_approval_evidence_complete'],
        row['trace_integrity_passed'],
    ]
    row['passed'] = all(checks)
    return row


def run_matrix(artifact_dir: Path, runs_per_task: int = 3) -> dict:
    provider = RecoveryFaultInjectionProvider()
    tasks = tuple(provider.tasks())
    if len(tasks) != 30:
        raise RuntimeError(f'Expected exactly 30 canonical tasks, got {len(tasks)}')
    if runs_per_task != 3:
        raise ValueError('Step 41.11 acceptance requires exactly 3 runs per task')
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=artifact_dir)
    rows = []
    for task in tasks:
        for run_index in range(1, runs_per_task + 1):
            evidence = adapter.execute(task, run_index)
            rows.append(_row(evidence))
    expected = len(tasks) * runs_per_task
    pairs = {(row['task_id'], row['run_index']) for row in rows}
    checks = {
        'evidence_coverage': len(rows) == expected and len(pairs) == expected,
        'correct_completion': all(r['correct_completion'] for r in rows),
        'deterministic_verification': all(r['deterministic_verification_passed'] for r in rows),
        'recovery_detection': all(r['recoverable_failure'] for r in rows),
        'recovery_success': all(r['recovery_succeeded'] for r in rows),
        'security': all(r['unauthorized_high_risk_actions'] == 0 for r in rows),
        'duplicate_side_effects': all(r['duplicate_harmful_side_effects'] == 0 for r in rows),
        'audit_evidence': all(r['audit_evidence_complete'] for r in rows),
        'approval_evidence': all(r['required_approval_evidence_complete'] for r in rows),
        'trace_integrity': all(r['trace_integrity_passed'] for r in rows),
    }
    failed_rows = [r for r in rows if not r['passed']]
    return {
        'step': '41.11',
        'kind': 'full_matrix_canonical_recovery_regression',
        'scope_note': 'Runs the repository P0CanonicalRuntimeAdapter across all 30 controlled Room P0 tasks, 3 runs each. This is a deterministic fixture regression, not proof of production or external-side-effect safety.',
        'tasks': len(tasks), 'runs_per_task': runs_per_task,
        'expected_runs': expected, 'received_runs': len(rows),
        'unique_task_run_pairs': len(pairs),
        'checks': [{'name': k, 'passed': v} for k, v in checks.items()],
        'passed': all(checks.values()),
        'failed_runs': [{'task_id': r['task_id'], 'run_index': r['run_index'], 'notes': r['notes']} for r in failed_rows],
        'summary': {
            'correct_completion_rate': sum(r['correct_completion'] for r in rows) / len(rows) if rows else 0.0,
            'deterministic_verification_rate': sum(r['deterministic_verification_passed'] for r in rows) / len(rows) if rows else 0.0,
            'recovery_detection_rate': sum(r['recoverable_failure'] for r in rows) / len(rows) if rows else 0.0,
            'recovery_success_rate': sum(r['recovery_succeeded'] for r in rows) / len(rows) if rows else 0.0,
            'unauthorized_high_risk_actions': sum(r['unauthorized_high_risk_actions'] for r in rows),
            'duplicate_harmful_side_effects': sum(r['duplicate_harmful_side_effects'] for r in rows),
            'runs_missing_audit_evidence': sum(not r['audit_evidence_complete'] for r in rows),
            'runs_missing_approval_evidence': sum(not r['required_approval_evidence_complete'] for r in rows),
            'trace_integrity_failures': sum(not r['trace_integrity_passed'] for r in rows),
        },
        'cases': rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Step 41.11 full canonical recovery regression')
    parser.add_argument('--artifact-dir', default=None)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = ROOT / 'artifacts' / 'step41_11'
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = Path(args.artifact_dir) if args.artifact_dir else out_dir / f'canonical-evidence-{run_id}'
    evidence_dir.mkdir(parents=True, exist_ok=False)
    report = run_matrix(evidence_dir)
    report['run_id'] = run_id
    report['canonical_evidence_dir'] = str(evidence_dir)
    report_path = out_dir / f'full-matrix-report-{run_id}.json'
    if report_path.exists():
        raise FileExistsError(f'Refusing to overwrite report: {report_path}')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Report: {report_path}')
    print(f'Canonical evidence directory: {evidence_dir}')
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
