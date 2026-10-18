"""Regression tests for Step 30.5.2 multi-task generalization benchmark."""
from experiments import step30_5_2_multitask_generalization as benchmark
from agent.cognition.adaptive_planner import AdaptivePlanner


def test_benchmark_covers_four_task_types_and_held_out_profiles():
    report = benchmark.run_benchmark(repeats=1)

    assert report["benchmark"] == "step30_5_2_multitask_generalization_v1"
    assert len(report["task_types"]) == 4
    assert report["evaluation_runs"] == 16
    assert report["summary"]["static"]["completion_rate"] == 1.0
    assert report["summary"]["adaptive"]["completion_rate"] == 1.0
    assert report["summary"]["static"]["verification_rate"] == 1.0
    assert report["summary"]["adaptive"]["verification_rate"] == 1.0
    assert report["summary"]["adaptive"]["holdout"]["completion_rate"] == 1.0
    assert report["summary"]["adaptive"]["holdout"]["verification_rate"] == 1.0
    assert report["p0_acceptance"]["status"] == "NOT_EVALUATED_BY_THIS_BENCHMARK"


def test_adaptive_planner_is_invoked_for_every_adaptive_case(monkeypatch):
    calls = []
    original_plan = AdaptivePlanner.plan

    def spy_plan(self, *args, **kwargs):
        calls.append({
            "goal": kwargs.get("goal"),
            "world_state": dict(kwargs["world_state"]),
            "goal_state": dict(kwargs["goal_state"]),
        })
        return original_plan(self, *args, **kwargs)

    monkeypatch.setattr(AdaptivePlanner, "plan", spy_plan)
    report = benchmark.run_benchmark(repeats=1)

    assert len(calls) == 8
    assert len({call["goal"] for call in calls}) == 4
    assert any(call["world_state"].get("spare_key") == "unavailable" for call in calls)
    assert any(call["world_state"].get("primary_switch") == "unavailable" for call in calls)
    assert any(call["world_state"].get("van") == "unavailable" for call in calls)
    assert any(call["world_state"].get("kettle") == "broken" for call in calls)
    assert report["summary"]["adaptive"]["completion_rate"] == 1.0


def test_holdout_routes_avoid_unavailable_resources():
    report = benchmark.run_benchmark(repeats=1)
    adaptive_holdout = [
        row for row in report["trials"]
        if row["mode"] == "adaptive" and row["holdout"]
    ]
    assert len(adaptive_holdout) == 4
    assert all(row["verified"] for row in adaptive_holdout)

    plans = {row["task"]: row["plan"] for row in adaptive_holdout}
    assert "find_spare_key" not in plans["unlock_door"]
    assert "locate_primary_switch" not in plans["turn_on_light"]
    assert "drive_to_customer" not in plans["deliver_package"]
    assert "boil_in_kettle" not in plans["make_tea"]
