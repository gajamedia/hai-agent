from experiments.step30_meta_learning_benchmark import run_benchmark


def test_matched_benchmark_gives_both_modes_same_recovery_budget():
    report = run_benchmark(repeats=2, training_repeats=2)

    static = report["summary"]["static"]
    adaptive = report["summary"]["adaptive"]

    assert report["evaluation_runs"] == 4
    assert static["runs"] == adaptive["runs"] == 2
    assert static["recovery_attempt_rate"] == 1.0
    assert adaptive["recovery_attempt_rate"] == 1.0
    assert static["completion_rate"] == 1.0
    assert adaptive["completion_rate"] == 1.0
    assert static["verification_rate"] == 1.0
    assert adaptive["verification_rate"] == 1.0


def test_adaptive_uses_actual_training_outcomes_and_selects_lower_action_route():
    report = run_benchmark(repeats=2, training_repeats=2)

    assert report["training_episodes_total"] == 8
    for pair in report["training"]:
        assert pair["candidate_strategies"][0]["successes"] == 2
        assert pair["candidate_strategies"][1]["successes"] == 2
        assert pair["selected_strategy"] == "find_spare_key"

    static = report["summary"]["static"]
    adaptive = report["summary"]["adaptive"]
    assert static["mean_actions_attempted"] == 4.0
    assert adaptive["mean_actions_attempted"] == 3.0
    assert report["summary"]["relative_action_reduction"] == 0.25
    assert report["summary"]["completion_rate_difference"] == 0.0


def test_pilot_does_not_claim_unmeasured_p0_security_metrics():
    report = run_benchmark(repeats=1, training_repeats=1)

    assert report["p0_acceptance"]["status"] == "NOT_EVALUATED_BY_THIS_PILOT"
    assert report["summary"]["static"]["unauthorized_high_risk_actions"] is None
    assert report["summary"]["adaptive"]["harmful_duplicate_side_effects"] is None
