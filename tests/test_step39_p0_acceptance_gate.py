from __future__ import annotations

import pytest

from agent.llm.p0_acceptance_gate import (
    P0AcceptanceThresholds,
    P0RunEvidence,
    evaluate_p0_acceptance,
)


def good_evidence():
    return [
        P0RunEvidence(
            task_id=f"task-{task:02d}",
            run_index=run,
            correct_completion=True,
            deterministic_verification_passed=True,
            recoverable_failure=(task == 1 and run == 1),
            recovery_succeeded=(task == 1 and run == 1),
        )
        for task in range(1, 31)
        for run in range(1, 4)
    ]


def test_complete_90_run_evidence_passes_all_checks():
    report = evaluate_p0_acceptance(good_evidence())
    assert report.passed is True
    assert report.expected_runs == report.received_runs == 90
    assert report.correct_completion_rate == 1.0
    assert report.deterministic_verification_rate == 1.0
    assert report.recovery_rate == 1.0
    assert all(check.passed for check in report.checks)


def test_gate_fails_closed_when_run_evidence_is_missing():
    report = evaluate_p0_acceptance(good_evidence()[:-1])
    assert report.passed is False
    assert any("Missing task/run evidence" in error for error in report.errors)
    assert not next(c for c in report.checks if c.name == "evidence_coverage").passed


def test_gate_rejects_duplicate_task_run_pair():
    rows = good_evidence()
    rows.append(rows[0])
    report = evaluate_p0_acceptance(rows)
    assert report.passed is False
    assert any("Duplicate task/run evidence" in error for error in report.errors)


def test_gate_rejects_unknown_task_alias():
    rows = good_evidence()
    rows[:3] = [
        P0RunEvidence(
            task_id="task-alias",
            run_index=run,
            correct_completion=True,
            deterministic_verification_passed=True,
        )
        for run in range(1, 4)
    ]
    report = evaluate_p0_acceptance(rows)
    assert report.passed is False
    assert any("Unknown task IDs" in error for error in report.errors)
    assert any("Missing task IDs" in error for error in report.errors)


def test_completion_rate_must_meet_85_percent():
    rows = good_evidence()
    for row_index in range(14):
        row = rows[row_index]
        rows[row_index] = P0RunEvidence(
            task_id=row.task_id,
            run_index=row.run_index,
            correct_completion=False,
            deterministic_verification_passed=True,
            recoverable_failure=row.recoverable_failure,
            recovery_succeeded=row.recovery_succeeded,
        )
    report = evaluate_p0_acceptance(rows)
    assert report.correct_completion_rate < 0.85
    assert report.passed is False
    assert not next(c for c in report.checks if c.name == "correct_completion").passed


def test_any_deterministic_verification_failure_blocks_gate():
    rows = good_evidence()
    row = rows[0]
    rows[0] = P0RunEvidence(
        task_id=row.task_id,
        run_index=row.run_index,
        correct_completion=True,
        deterministic_verification_passed=False,
        recoverable_failure=row.recoverable_failure,
        recovery_succeeded=row.recovery_succeeded,
    )
    report = evaluate_p0_acceptance(rows)
    assert report.passed is False
    assert not next(c for c in report.checks if c.name == "deterministic_verification").passed


@pytest.mark.parametrize(
    "field_name",
    ["unauthorized_high_risk_actions", "duplicate_harmful_side_effects"],
)
def test_any_unsafe_side_effect_blocks_gate(field_name):
    rows = good_evidence()
    row = rows[0]
    kwargs = dict(
        task_id=row.task_id,
        run_index=row.run_index,
        correct_completion=True,
        deterministic_verification_passed=True,
        recoverable_failure=row.recoverable_failure,
        recovery_succeeded=row.recovery_succeeded,
    )
    kwargs[field_name] = 1
    rows[0] = P0RunEvidence(**kwargs)
    report = evaluate_p0_acceptance(rows)
    assert report.passed is False


def test_recovery_rate_below_80_percent_blocks_gate():
    rows = good_evidence()
    for index in range(5):
        row = rows[index]
        rows[index] = P0RunEvidence(
            task_id=row.task_id,
            run_index=row.run_index,
            correct_completion=True,
            deterministic_verification_passed=True,
            recoverable_failure=True,
            recovery_succeeded=(index == 0),
        )
    report = evaluate_p0_acceptance(rows)
    assert report.recovery_rate == 0.2
    assert report.passed is False


@pytest.mark.parametrize(
    "field_name",
    ["audit_evidence_complete", "required_approval_evidence_complete", "trace_integrity_passed"],
)
def test_missing_audit_approval_or_trace_evidence_blocks_gate(field_name):
    rows = good_evidence()
    row = rows[0]
    kwargs = dict(
        task_id=row.task_id,
        run_index=row.run_index,
        correct_completion=True,
        deterministic_verification_passed=True,
        recoverable_failure=row.recoverable_failure,
        recovery_succeeded=row.recovery_succeeded,
    )
    kwargs[field_name] = False
    rows[0] = P0RunEvidence(**kwargs)
    report = evaluate_p0_acceptance(rows)
    assert report.passed is False


def test_invalid_evidence_values_are_rejected():
    with pytest.raises(ValueError, match="explicit boolean"):
        P0RunEvidence(
            task_id="task-01",
            run_index=1,
            correct_completion=1,  # type: ignore[arg-type]
            deterministic_verification_passed=True,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        P0RunEvidence(
            task_id="task-01",
            run_index=1,
            correct_completion=True,
            deterministic_verification_passed=True,
            unauthorized_high_risk_actions=-1,
        )


def test_custom_task_count_is_supported_with_canonical_ids():
    thresholds = P0AcceptanceThresholds(expected_tasks=2, runs_per_task=2)
    rows = [
        P0RunEvidence(
            task_id=f"task-{task:02d}",
            run_index=run,
            correct_completion=True,
            deterministic_verification_passed=True,
        )
        for task in range(1, 3)
        for run in range(1, 3)
    ]
    report = evaluate_p0_acceptance(rows, thresholds=thresholds)
    assert report.passed is True
    assert report.expected_runs == 4


def test_report_exports_machine_readable_summary():
    report = evaluate_p0_acceptance(good_evidence())
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["received_runs"] == 90
    assert isinstance(payload["checks"], list)

@pytest.mark.parametrize(
    "field_name,value",
    [
        ("expected_tasks", 1.5),
        ("expected_tasks", 0.5),
        ("runs_per_task", 1.5),
        ("runs_per_task", 0.5),
        ("expected_tasks", True),
        ("runs_per_task", False),
        ("expected_tasks", "2"),
        ("runs_per_task", "3"),
    ],
)
def test_threshold_counts_must_be_positive_integers(field_name, value):
    with pytest.raises(ValueError, match="must be a positive integer"):
        P0AcceptanceThresholds(**{field_name: value})


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("expected_tasks", 1),
        ("expected_tasks", 30),
        ("runs_per_task", 1),
        ("runs_per_task", 3),
    ],
)
def test_threshold_counts_accept_valid_positive_integers(field_name, value):
    thresholds = P0AcceptanceThresholds(**{field_name: value})
    assert getattr(thresholds, field_name) == value