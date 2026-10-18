
import pytest

from experiments.step30_context_adaptation_benchmark import (
    MODES,
    run_benchmark,
)


def test_benchmark_runs_all_modes_in_both_environment_scenarios():
    report = run_benchmark(repeats=2, training_repeats=2)

    assert report["benchmark"] == "step30_context_adaptation_benchmark_v1"
    assert report["evaluation_runs"] == 2 * 2 * len(MODES)
    assert report["evaluation_design"]["same_initial_action_for_all_modes"]
    assert report["evaluation_design"]["same_recovery_budget_for_all_modes"]
    assert report["evaluation_design"]["evaluation_updates_training"] is False

    for scenario in (
        "spare_key_available",
        "spare_key_unavailable",
    ):
        for mode in MODES:
            summary = report["summary_by_scenario"][scenario][mode]
            assert summary["runs"] == 2
            assert 0.0 <= summary["completion_rate"] <= 1.0
            assert 0.0 <= summary["verification_rate"] <= 1.0


def test_experience_only_selects_learned_shorter_route():
    report = run_benchmark(repeats=2, training_repeats=2)

    for training_pair in report["training"]:
        assert training_pair["frozen_before_evaluation"] is True
        assert (
            training_pair["experience_only_selected_strategy"]
            == "find_spare_key"
        )
        for candidate in training_pair["candidate_strategies"]:
            assert candidate["training_episodes"] == 2
            assert candidate["successes"] == 2
            assert candidate["failures"] == 0


def test_context_adaptive_uses_feasible_route_when_spare_key_unavailable():
    report = run_benchmark(repeats=2, training_repeats=2)
    summary = report["summary_by_scenario"]["spare_key_unavailable"]

    assert summary["context_adaptive"]["completion_rate"] == 1.0
    assert summary["context_adaptive"]["verification_rate"] == 1.0

    rows = [
        row for row in report["trials"]
        if row["scenario"] == "spare_key_unavailable"
        and row["mode"] == "context_adaptive"
    ]
    assert all(
        row["selected_strategy"] == "open_box_then_take_key"
        for row in rows
    )


def test_experience_only_can_fail_when_learned_route_is_infeasible():
    report = run_benchmark(repeats=2, training_repeats=2)
    summary = report["summary_by_scenario"]["spare_key_unavailable"]

    assert summary["experience_only"]["completion_rate"] == 0.0
    assert summary["context_adaptive"]["completion_rate"] == 1.0


def test_all_modes_complete_when_spare_key_is_available():
    report = run_benchmark(repeats=2, training_repeats=2)
    summary = report["summary_by_scenario"]["spare_key_available"]

    for mode in MODES:
        assert summary[mode]["completion_rate"] == 1.0
        assert summary[mode]["verification_rate"] == 1.0


def test_benchmark_does_not_claim_unmeasured_security_metrics():
    report = run_benchmark(repeats=1, training_repeats=1)

    for mode in MODES:
        assert (
            report["overall_summary"][mode]["unauthorized_high_risk_actions"]
            is None
        )
        assert (
            report["overall_summary"][mode]["harmful_duplicate_side_effects"]
            is None
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repeats": 0}, "repeats"),
        ({"repeats": True}, "repeats"),
        ({"training_repeats": 0}, "training_repeats"),
        ({"training_repeats": True}, "training_repeats"),
    ],
)
def test_benchmark_rejects_invalid_repeat_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_benchmark(**kwargs)
