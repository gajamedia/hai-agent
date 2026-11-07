"""Regression: an Agent with memory must stop repeating a route that keeps failing.

Backed by experiments/learning_curve_benchmark.py (hidden-fault room).
"""
from experiments import learning_curve_benchmark as bench
from agent.cognition.adaptive_planner import AdaptivePlanner
from agent.cognition.planner import Planner
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector


def test_learning_agent_beats_no_memory_agent_on_decoy_route():
    learning = [bench.run_arm("learning", 30, seed, 0.0) for seed in range(3)]
    fresh = [bench.run_arm("no_learning", 30, seed, 0.0) for seed in range(3)]

    def last_third_rate(rows):
        tail = rows[-10:]
        return sum(r["success"] for r in tail) / len(tail)

    assert all(last_third_rate(rows) == 1.0 for rows in learning)
    assert all(last_third_rate(rows) == 0.0 for rows in fresh)


def test_failures_of_different_routes_are_not_merged_into_one_strategy():
    agent = bench.build_agent("attribution")
    for i in range(2):
        bench.run_episode(agent, bench.FaultyRoom(False), f"bad-{i}")
    bench.run_episode(agent, bench.FaultyRoom(False), "good-0")

    routes = {tuple(s.steps): s for s in agent.get_registered_strategies()}
    bad = routes[("find_spare_key", "unlock_door")]
    good = routes[("open_box", "take_key", "unlock_door")]
    assert (bad.attempts, bad.success_count) == (2, 0)
    assert (good.attempts, good.success_count) == (1, 1)


def test_avoidance_is_opt_in_and_default_planner_keeps_safe_reference():
    actions = bench.build_actions()
    bad = Strategy(name="s", goal="g", steps=["find_spare_key", "unlock_door"])
    for i in range(3):
        bad.record_failure(experience_id=f"f{i}", action_count=2, progress_delta=0.0)
    world = {"door": "locked", "box": "closed", "key": "inside_box",
             "spare_key": "available"}

    default = AdaptivePlanner(Planner(actions), StrategySelector())
    opted_in = AdaptivePlanner(
        Planner(actions), StrategySelector(), avoid_unreliable_routes=True
    )
    kwargs = dict(world_state=world, goal_state={"door": "unlocked"},
                  goal="g", strategies=[bad])

    assert [a.name for a in default.plan(**kwargs)] == ["find_spare_key", "unlock_door"]
    assert [a.name for a in opted_in.plan(**kwargs)] == ["open_box", "take_key", "unlock_door"]


def test_no_alternative_falls_back_to_base_plan():
    only = [a for a in bench.build_actions() if a.name in ("find_spare_key", "unlock_door")]
    bad = Strategy(name="s", goal="g", steps=["find_spare_key", "unlock_door"])
    for i in range(3):
        bad.record_failure(experience_id=f"f{i}", action_count=2, progress_delta=0.0)
    planner = AdaptivePlanner(Planner(only), StrategySelector(), avoid_unreliable_routes=True)
    plan = planner.plan(
        world_state={"door": "locked", "key": "inside_box", "spare_key": "available"},
        goal_state={"door": "unlocked"}, goal="g", strategies=[bad],
    )
    assert [a.name for a in plan] == ["find_spare_key", "unlock_door"]
