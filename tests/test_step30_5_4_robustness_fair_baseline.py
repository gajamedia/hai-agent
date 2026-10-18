"""Tests for the Step 30.5.4 fair-baseline robustness benchmark."""
from experiments import step30_5_4_robustness_fair_baseline as benchmark
from agent.cognition.adaptive_planner import AdaptivePlanner


def test_benchmark_uses_fair_bfs_baseline_and_verifies_every_run():
    report = benchmark.run_benchmark(repeats=1)

    # 4 scenarios x 2 shortcut states x 2 history profiles x 2 methods.
    assert report["evaluation_runs"] == 32
    assert report["summary"]["bfs_baseline"]["completion_rate"] == 1.0
    assert report["summary"]["bfs_baseline"]["verification_rate"] == 1.0
    assert report["summary"]["adaptive"]["completion_rate"] == 1.0
    assert report["summary"]["adaptive"]["verification_rate"] == 1.0
    assert report["p0_acceptance"]["status"] == "NOT_EVALUATED_BY_THIS_BENCHMARK"


def test_adaptive_planner_is_called_for_every_condition_and_repeat(monkeypatch):
    calls = []
    original = AdaptivePlanner.plan

    def spy(self, *args, **kwargs):
        calls.append((kwargs.get("goal"), kwargs.get("world_state", {}).get("shortcut_available")))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AdaptivePlanner, "plan", spy)
    report = benchmark.run_benchmark(repeats=2)

    assert len(calls) == 4 * 2 * 2 * 2
    assert report["evaluation_runs"] == 64


def test_bfs_finds_shortest_route_when_shortcut_is_available():
    case = benchmark.build_scenarios()[0]
    planner = benchmark.BFSPlanner(case.actions)

    plan = planner.plan(case.initial_state, case.goal_state)

    assert len(plan) == 1
    assert plan[0].name == case.short_step


def test_bfs_uses_feasible_long_route_when_shortcut_is_unavailable():
    case = benchmark.build_scenarios()[0]
    state = dict(case.initial_state)
    state["shortcut_available"] = False
    planner = benchmark.BFSPlanner(case.actions)

    plan = planner.plan(state, case.goal_state)

    assert len(plan) == 3
    assert tuple(action.name for action in plan) == case.long_steps


def test_noisy_history_exposes_adaptive_action_gap_without_breaking_completion():
    report = benchmark.run_benchmark(repeats=1)
    noisy_available = report["summary"]["stratified"]["noisy|shortcut_available=true"]

    assert noisy_available["bfs_baseline"]["mean_actions"] == 1.0
    assert noisy_available["adaptive"]["mean_actions"] >= noisy_available["bfs_baseline"]["mean_actions"]
    assert noisy_available["adaptive"]["completion_rate"] == 1.0
    assert noisy_available["adaptive"]["verification_rate"] == 1.0
