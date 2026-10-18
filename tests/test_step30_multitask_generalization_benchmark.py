from experiments.step30_multitask_generalization_benchmark import run_benchmark


def test_multitask_benchmark_contains_in_distribution_and_holdout_profiles():
    report = run_benchmark(repeats=2, training_repeats=2)

    assert report["task_count"] == 2
    assert report["evaluation_runs"] == 8
    assert report["interpretation"]["holdout"] is True
    assert report["training"]["context"] == "spare_key_available only"
    assert {p["split"] for p in report["task_profiles"]} == {"in_distribution", "holdout"}


def test_baseline_recovers_holdout_but_global_adaptive_selector_exposes_context_gap():
    report = run_benchmark(repeats=2, training_repeats=2)
    static = report["summary"]["static"]
    adaptive = report["summary"]["adaptive"]

    assert report["training"]["selected_strategy"] == "find_spare_key"
    assert static["completion_rate"] == 1.0
    assert static["holdout"]["completion_rate"] == 1.0
    assert adaptive["by_task_profile"]["spare_key_available"]["completion_rate"] == 1.0
    assert adaptive["holdout"]["completion_rate"] == 0.0
    assert report["summary"]["holdout_completion_rate_difference"] == -1.0


def test_holdout_failure_is_reported_without_claiming_p0_acceptance():
    report = run_benchmark(repeats=1, training_repeats=1)

    assert report["p0_acceptance"]["status"] == "NOT_EVALUATED_BY_THIS_PILOT"
    assert report["summary"]["adaptive"]["unauthorized_high_risk_actions"] is None
    assert report["summary"]["adaptive"]["harmful_duplicate_side_effects"] is None
    holdout_trials = [
        row for row in report["trials"]
        if row["holdout"] and row["mode"] == "adaptive"
    ]
    assert len(holdout_trials) == 1
    assert holdout_trials[0]["verified"] is False
