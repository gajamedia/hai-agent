"""Tests for the Step 30.5.3 controlled adaptive strategy-value benchmark."""
from experiments import step30_5_3_adaptive_strategy_value as benchmark
from agent.cognition.adaptive_planner import AdaptivePlanner


def test_adaptive_strategy_reduces_actions_and_verifies_all_scenarios():
    report = benchmark.run_benchmark(repeats=1)

    assert report["evaluation_runs"] == 8
    assert report["summary"]["fixed_baseline"]["completion_rate"] == 1.0
    assert report["summary"]["adaptive"]["completion_rate"] == 1.0
    assert report["summary"]["fixed_baseline"]["verification_rate"] == 1.0
    assert report["summary"]["adaptive"]["verification_rate"] == 1.0
    assert report["summary"]["adaptive"]["mean_actions"] < report["summary"]["fixed_baseline"]["mean_actions"]
    assert report["p0_acceptance"]["status"] == "NOT_EVALUATED_BY_THIS_BENCHMARK"


def test_adaptive_planner_is_called_once_per_scenario_per_repeat(monkeypatch):
    calls = []
    original = AdaptivePlanner.plan

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("goal"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AdaptivePlanner, "plan", spy)
    benchmark.run_benchmark(repeats=2)

    assert len(calls) == 8
    assert len(set(calls)) == 4


def test_adaptive_plans_use_short_alternative_routes():
    report = benchmark.run_benchmark(repeats=1)
    adaptive_rows = [row for row in report["trials"] if row["mode"] == "adaptive"]

    assert len(adaptive_rows) == 4
    assert all(row["completed"] and row["verified"] for row in adaptive_rows)
    assert all(row["action_count"] == 1 for row in adaptive_rows)
