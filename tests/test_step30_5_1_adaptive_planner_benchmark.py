"""Tests for the Step 30.5.1 benchmark's real AdaptivePlanner call path."""
from experiments import step30_5_1_adaptive_planner_benchmark as benchmark
from agent.cognition.adaptive_planner import AdaptivePlanner


def test_benchmark_invokes_adaptive_planner_and_generalizes_to_holdout(monkeypatch):
    calls = []
    original_plan = AdaptivePlanner.plan

    def spy_plan(self, *args, **kwargs):
        calls.append({
            "world_state": dict(kwargs["world_state"]),
            "goal_state": dict(kwargs["goal_state"]),
        })
        return original_plan(self, *args, **kwargs)

    monkeypatch.setattr(AdaptivePlanner, "plan", spy_plan)
    report = benchmark.run_benchmark(repeats=1, training_repeats=1)

    adaptive = report["summary"]["adaptive"]
    assert report["benchmark"] == "step30_5_1_adaptive_planner_integration_v1"
    assert len(calls) == 2
    assert adaptive["completion_rate"] == 1.0
    assert adaptive["verification_rate"] == 1.0
    assert adaptive["holdout"]["completion_rate"] == 1.0
    assert adaptive["holdout"]["verification_rate"] == 1.0

    by_profile = adaptive["by_task_profile"]
    assert by_profile["spare_key_available"]["completion_rate"] == 1.0
    assert by_profile["spare_key_unavailable"]["completion_rate"] == 1.0

    trials = [
        trial for trial in report["trials"]
        if trial["mode"] == "adaptive"
    ]
    selected = {trial["task_profile"]: trial["selected_route"] for trial in trials}
    assert selected["spare_key_available"] == ["find_spare_key", "unlock_door"]
    assert selected["spare_key_unavailable"] == [
        "open_box", "take_key", "unlock_door"
    ]


def test_adaptive_benchmark_records_live_unavailable_key_context(monkeypatch):
    observed_states = []
    original_plan = AdaptivePlanner.plan

    def spy_plan(self, *args, **kwargs):
        observed_states.append(dict(kwargs["world_state"]))
        return original_plan(self, *args, **kwargs)

    monkeypatch.setattr(AdaptivePlanner, "plan", spy_plan)
    benchmark.run_benchmark(repeats=1, training_repeats=1)

    assert any(state.get("spare_key") == "unavailable" for state in observed_states)
    assert any(state.get("spare_key") == "available" for state in observed_states)
