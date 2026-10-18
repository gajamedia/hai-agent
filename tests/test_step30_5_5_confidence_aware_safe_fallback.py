"""Regression tests for Step 30.5.5 confidence-aware safe fallback."""
from experiments import step30_5_5_confidence_aware_safe_fallback as benchmark
from agent.cognition.adaptive_planner import AdaptivePlanner
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector


def test_benchmark_keeps_completion_and_verification_at_100_percent():
    report = benchmark.run_benchmark(repeats=1)
    assert report["evaluation_runs"] == 32
    for method in ("bfs_baseline", "adaptive"):
        assert report["summary"][method]["completion_rate"] == 1.0
        assert report["summary"][method]["verification_rate"] == 1.0
    assert report["p0_acceptance"]["status"] == "NOT_EVALUATED_BY_THIS_BENCHMARK"


def test_misleading_history_does_not_choose_long_route_when_shortcut_exists():
    report = benchmark.run_benchmark(repeats=1)
    for scenario in benchmark.build_scenarios():
        rows = [row for row in report["trials"]
                if row["scenario"] == scenario.name
                and row["shortcut_available"] is True
                and row["history_profile"] == "noisy"]
        baseline = next(row for row in rows if row["mode"] == "bfs_baseline")
        adaptive = next(row for row in rows if row["mode"] == "adaptive")
        assert adaptive["action_count"] == baseline["action_count"] == 1
        assert adaptive["plan"] == baseline["plan"]


def test_adaptive_never_uses_more_actions_than_valid_bfs_baseline_on_fixture():
    report = benchmark.run_benchmark(repeats=2)
    trials = report["trials"]
    keys = ("scenario", "shortcut_available", "history_profile", "repeat")
    baseline = {tuple(row[key] for key in keys): row for row in trials if row["mode"] == "bfs_baseline"}
    adaptive = [row for row in trials if row["mode"] == "adaptive"]
    assert len(adaptive) == len(baseline)
    for row in adaptive:
        reference = baseline[tuple(row[key] for key in keys)]
        assert row["action_count"] <= reference["action_count"]


def test_already_satisfied_goal_returns_empty_plan():
    case = benchmark.build_scenarios()[0]
    planner = benchmark.BFSPlanner(case.actions)
    adaptive = AdaptivePlanner(planner, StrategySelector())
    state = dict(case.initial_state)
    state.update(case.goal_state)
    result = adaptive.plan(state, case.goal_state, case.goal, [])
    assert result == []


def test_infeasible_learned_strategy_falls_back_to_base_planner():
    case = benchmark.build_scenarios()[0]
    planner = benchmark.BFSPlanner(case.actions)
    adaptive = AdaptivePlanner(planner, StrategySelector())
    bad = Strategy(name="bad", goal=case.goal, steps=["missing_action"])
    result = adaptive.plan(case.initial_state, case.goal_state, case.goal, [bad])
    expected = planner.plan(case.initial_state, case.goal_state)
    assert [action.name for action in result] == [action.name for action in expected]


def test_learned_plan_is_used_if_base_planner_cannot_find_valid_plan():
    case = benchmark.build_scenarios()[0]

    class EmptyPlanner:
        def __init__(self, actions):
            self.actions = list(actions)
        def plan(self, world_state, goal_state):
            return []

    planner = EmptyPlanner(case.actions)
    adaptive = AdaptivePlanner(planner, StrategySelector())
    learned = Strategy(name="short", goal=case.goal, steps=[case.short_step])
    result = adaptive.plan(case.initial_state, case.goal_state, case.goal, [learned])
    assert [action.name for action in result] == [case.short_step]
