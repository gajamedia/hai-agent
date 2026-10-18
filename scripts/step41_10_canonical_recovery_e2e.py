from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
sys.path.insert(0, str(REPO / 'src'))

from agent.llm.p0_canonical_runtime_adapter import P0CanonicalRuntimeAdapter
from agent.llm.p0_recovery_fault_injection_provider import RecoveryFaultInjectionProvider


def run_smoke(artifact_dir: Path) -> dict:
    provider = RecoveryFaultInjectionProvider()
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=artifact_dir)
    tasks = tuple(provider.tasks())
    if len(tasks) < 3:
        raise RuntimeError(f'Expected at least 3 canonical tasks, got {len(tasks)}')
    # Sample first, middle, and final task while each execution uses a fresh scenario.
    selected = (tasks[0], tasks[len(tasks) // 2], tasks[-1])
    rows = []
    for task in selected:
        evidence = adapter.execute(task, 1)
        row = {
            'task_id': evidence.task_id,
            'run_index': evidence.run_index,
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
        rows.append(row)
    return {
        'step': '41.10',
        'kind': 'canonical_adapter_recovery_end_to_end_smoke',
        'scope_note': 'Runs the repository P0CanonicalRuntimeAdapter with the existing Step 41.5 controlled recovery provider. This validates the real adapter path for the Room fixture only; it does not integrate retry into the adapter or prove production safety.',
        'selected_tasks': [task.task_id for task in selected],
        'total_cases': len(rows),
        'passed_cases': sum(1 for row in rows if row['passed']),
        'failed_cases': [row['task_id'] for row in rows if not row['passed']],
        'passed': all(row['passed'] for row in rows),
        'cases': rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Step 41.10 canonical adapter recovery smoke validation')
    parser.add_argument('--artifact-dir', default=None, help='Optional output directory for canonical adapter evidence')
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ROOT / 'artifacts' / 'step41_10' / f'canonical-evidence-{run_id}'
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report = run_smoke(artifact_dir)
    report['run_id'] = run_id
    report['canonical_evidence_dir'] = str(artifact_dir)
    out_dir = ROOT / 'artifacts' / 'step41_10'
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f'e2e-report-{run_id}.json'
    # Unique timestamped output: refuse overwrite even if clock/run IDs collide.
    if report_path.exists():
        raise FileExistsError(f'Refusing to overwrite report: {report_path}')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Report: {report_path}')
    print(f'Canonical evidence directory: {artifact_dir}')
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
