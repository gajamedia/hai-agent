"""Step 30.5.4: robustness and fair baseline evaluation for AdaptivePlanner.

Both methods receive the same action registry, initial state, and goal. The
baseline uses breadth-first search (BFS) to find a shortest feasible plan.
AdaptivePlanner uses StrategySelector and falls back to that same BFS planner
when its selected strategy is infeasible or fails goal-state validation.

This is a deterministic synthetic benchmark, not P0 acceptance.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.core.action import Action
from agent.cognition.adaptive_planner import AdaptivePlanner
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector


@dataclass(frozen=True)
class Scenario:
    name: str
    goal: str
    initial_state: dict[str, Any]
    goal_state: dict[str, Any]
    actions: tuple[Action, ...]
    long_steps: tuple[str, ...]
    short_step: str
    shortcut_state_key: str = "shortcut_available"


def make_action(
    name: str,
    preconditions: dict[str, Any],
    effects: dict[str, Any],
) -> Action:
    return Action(
        name=name,
        description=f"Step 30.5.4 controlled action: {name}",
        handler=lambda **kwargs: {"success": True},
        preconditions=preconditions,
        effects=effects,
    )


def build_scenarios() -> list[Scenario]:
    cases: list[Scenario] = []
    definitions = [
        ("unlock_door", "unlock door", "door", "locked", "unlocked", "key_stage",
         ("inspect_key", "prepare_key", "unlock_door"), "use_master_key"),
        ("turn_on_light", "turn on light", "light", "off", "on", "switch_stage",
         ("inspect_switch", "connect_switch", "turn_on_light"), "use_remote"),
        ("deliver_package", "deliver package", "package", "at_origin", "delivered", "delivery_stage",
         ("pick_up_package", "drive_to_customer", "deliver_package"), "instant_dispatch"),
        ("make_tea", "make tea", "tea", "not_ready", "ready", "tea_stage",
         ("inspect_kettle", "prepare_kettle", "brew_tea"), "use_tea_maker"),
    ]

    for name, goal, goal_key, start_value, goal_value, stage_key, long_steps, short_step in definitions:
        initial: dict[str, Any] = {
            goal_key: start_value,
            stage_key: "start",
            "shortcut_available": True,
        }
        goal_state = {goal_key: goal_value}

        if name == "unlock_door":
            long_actions = (
                make_action(long_steps[0], {stage_key: "start"}, {stage_key: "inspected"}),
                make_action(long_steps[1], {stage_key: "inspected"}, {stage_key: "prepared"}),
                make_action(long_steps[2], {stage_key: "prepared", goal_key: start_value}, {goal_key: goal_value}),
            )
            shortcut_preconditions = {"shortcut_available": True, goal_key: start_value}
        elif name == "turn_on_light":
            long_actions = (
                make_action(long_steps[0], {stage_key: "start"}, {stage_key: "inspected"}),
                make_action(long_steps[1], {stage_key: "inspected"}, {stage_key: "connected"}),
                make_action(long_steps[2], {stage_key: "connected", goal_key: start_value}, {goal_key: goal_value}),
            )
            shortcut_preconditions = {"shortcut_available": True, goal_key: start_value}
        elif name == "deliver_package":
            long_actions = (
                make_action(long_steps[0], {stage_key: "start"}, {stage_key: "carried"}),
                make_action(long_steps[1], {stage_key: "carried"}, {stage_key: "at_customer"}),
                make_action(long_steps[2], {stage_key: "at_customer", goal_key: start_value}, {goal_key: goal_value}),
            )
            shortcut_preconditions = {"shortcut_available": True, goal_key: start_value}
        else:
            long_actions = (
                make_action(long_steps[0], {stage_key: "start"}, {stage_key: "inspected"}),
                make_action(long_steps[1], {stage_key: "inspected"}, {stage_key: "boiled"}),
                make_action(long_steps[2], {stage_key: "boiled", goal_key: start_value}, {goal_key: goal_value}),
            )
            shortcut_preconditions = {"shortcut_available": True, goal_key: start_value}

        shortcut_action = make_action(short_step, shortcut_preconditions, {goal_key: goal_value})
        cases.append(
            Scenario(
                name=name,
                goal=goal,
                initial_state=initial,
                goal_state=goal_state,
                actions=long_actions + (shortcut_action,),
                long_steps=long_steps,
                short_step=short_step,
            )
        )
    return cases


class BFSPlanner:
    """Small deterministic shortest-plan baseline over the supplied actions."""

    def __init__(self, actions: tuple[Action, ...] | list[Action]):
        self.actions = list(actions)

    @staticmethod
    def _state_key(state: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(key), repr(value)) for key, value in state.items()))

    @staticmethod
    def _goal_reached(state: dict[str, Any], goal_state: dict[str, Any]) -> bool:
        return all(state.get(key) == expected for key, expected in goal_state.items())

    def plan(self, world_state: dict[str, Any], goal_state: dict[str, Any]) -> list[Action]:
        start = dict(world_state)
        if self._goal_reached(start, goal_state):
            return []

        queue = deque([(start, [])])
        visited = {self._state_key(start)}
        while queue:
            state, path = queue.popleft()
            for action in self.actions:
                if not action.can_execute(state):
                    continue
                next_state = dict(state)
                next_state.update(action.effects)
                next_path = path + [action]
                if self._goal_reached(next_state, goal_state):
                    return next_path
                key = self._state_key(next_state)
                if key not in visited:
                    visited.add(key)
                    queue.append((next_state, next_path))
        return []


def strategy_with_history(
    name: str,
    goal: str,
    steps: tuple[str, ...],
    successes: int,
    failures: int,
    prefix: str,
) -> Strategy:
    strategy = Strategy(name=name, goal=goal, steps=list(steps))
    for index in range(successes):
        strategy.record_success(
            experience_id=f"{prefix}-{name}-success-{index + 1}",
            action_count=len(steps),
            progress_delta=1.0,
        )
    for index in range(failures):
        strategy.record_failure(
            experience_id=f"{prefix}-{name}-failure-{index + 1}",
            action_count=len(steps),
            progress_delta=0.0,
        )
    return strategy


def build_strategies(case: Scenario, profile: str, prefix: str) -> list[Strategy]:
    if profile == "accurate":
        # Short-route evidence is reliable; long route remains a valid alternative.
        short_history = (10, 0)
        long_history = (6, 0)
    elif profile == "noisy":
        # Deliberately misleading history: long route has higher historical success.
        short_history = (2, 8)
        long_history = (8, 2)
    else:
        raise ValueError(f"Unknown history profile: {profile}")

    return [
        strategy_with_history(
            f"{case.name}_long", case.goal, case.long_steps,
            long_history[0], long_history[1], prefix,
        ),
        strategy_with_history(
            f"{case.name}_short", case.goal, (case.short_step,),
            short_history[0], short_history[1], prefix,
        ),
    ]


def simulate(initial_state: dict[str, Any], plan: list[Action]) -> tuple[bool, dict[str, Any], int]:
    state = dict(initial_state)
    for count, action in enumerate(plan, start=1):
        if not action.can_execute(state):
            return False, state, count
        state.update(action.effects)
    return True, state, len(plan)


@dataclass
class Trial:
    scenario: str
    shortcut_available: bool
    history_profile: str
    mode: str
    repeat: int
    plan: list[str]
    action_count: int
    completed: bool
    verified: bool


def run_benchmark(repeats: int = 3) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    trials: list[Trial] = []
    cases = build_scenarios()
    for case in cases:
        for shortcut_available in (True, False):
            initial_state = dict(case.initial_state)
            initial_state[case.shortcut_state_key] = shortcut_available
            for profile in ("accurate", "noisy"):
                for repeat in range(1, repeats + 1):
                    # Same state, goal, and action registry for both methods.
                    baseline_planner = BFSPlanner(case.actions)
                    baseline_plan = baseline_planner.plan(initial_state, case.goal_state)
                    ok, final_state, count = simulate(initial_state, baseline_plan)
                    verified = ok and all(final_state.get(k) == v for k, v in case.goal_state.items())
                    trials.append(Trial(
                        case.name, shortcut_available, profile, "bfs_baseline", repeat,
                        [item.name for item in baseline_plan], count, verified, verified,
                    ))

                    adaptive_fallback = BFSPlanner(case.actions)
                    adaptive = AdaptivePlanner(
                        planner=adaptive_fallback,
                        strategy_selector=StrategySelector(),
                    )
                    strategies = build_strategies(
                        case, profile, f"step30-5-4-{case.name}-{shortcut_available}-{profile}-{repeat}"
                    )
                    adaptive_plan = adaptive.plan(
                        world_state=initial_state,
                        goal_state=case.goal_state,
                        goal=case.goal,
                        strategies=strategies,
                    )
                    ok, final_state, count = simulate(initial_state, adaptive_plan)
                    verified = ok and all(final_state.get(k) == v for k, v in case.goal_state.items())
                    trials.append(Trial(
                        case.name, shortcut_available, profile, "adaptive", repeat,
                        [item.name for item in adaptive_plan], count, verified, verified,
                    ))

    summaries: dict[str, Any] = {}
    for mode in ("bfs_baseline", "adaptive"):
        rows = [row for row in trials if row.mode == mode]
        summaries[mode] = {
            "runs": len(rows),
            "completed": sum(row.completed for row in rows),
            "completion_rate": sum(row.completed for row in rows) / len(rows),
            "verified": sum(row.verified for row in rows),
            "verification_rate": sum(row.verified for row in rows) / len(rows),
            "mean_actions": round(sum(row.action_count for row in rows) / len(rows), 6),
        }

    grouped: dict[str, Any] = {}
    for profile in ("accurate", "noisy"):
        for shortcut_available in (True, False):
            key = f"{profile}|shortcut_available={str(shortcut_available).lower()}"
            grouped_rows = [row for row in trials if row.history_profile == profile and row.shortcut_available == shortcut_available]
            by_mode: dict[str, Any] = {}
            for mode in ("bfs_baseline", "adaptive"):
                rows = [row for row in grouped_rows if row.mode == mode]
                by_mode[mode] = {
                    "runs": len(rows),
                    "completion_rate": sum(row.completed for row in rows) / len(rows),
                    "verification_rate": sum(row.verified for row in rows) / len(rows),
                    "mean_actions": round(sum(row.action_count for row in rows) / len(rows), 6),
                }
            by_mode["adaptive_minus_bfs_mean_actions"] = round(
                by_mode["adaptive"]["mean_actions"] - by_mode["bfs_baseline"]["mean_actions"], 6
            )
            grouped[key] = by_mode

    delta = summaries["adaptive"]["mean_actions"] - summaries["bfs_baseline"]["mean_actions"]
    return {
        "benchmark": "step30_5_4_robustness_fair_baseline_v1",
        "purpose": "compare AdaptivePlanner with a shortest-path BFS baseline using identical actions, states, and goals; test reliable versus misleading strategy histories and shortcut availability",
        "evaluation_runs": len(trials),
        "repeats": repeats,
        "scenario_count": len(cases),
        "scenarios": [case.name for case in cases],
        "conditions": {
            "shortcut_available": [True, False],
            "strategy_history_profiles": ["accurate", "noisy"],
            "modes": ["bfs_baseline", "adaptive"],
        },
        "summary": {
            **summaries,
            "adaptive_minus_bfs_mean_actions": round(delta, 6),
            "adaptive_excess_actions_percent": round(100 * delta / summaries["bfs_baseline"]["mean_actions"], 6)
                if summaries["bfs_baseline"]["mean_actions"] else 0.0,
            "stratified": grouped,
        },
        "limitations": [
            "deterministic synthetic action/state models, not real-world execution",
            "BFS is optimal only for this benchmark's unit-cost deterministic action model",
            "strategy history profiles are constructed test fixtures, not learned from real interaction",
            "does not measure authorization, harmful side effects, recovery quality, or P0 acceptance",
        ],
        "p0_acceptance": {"status": "NOT_EVALUATED_BY_THIS_BENCHMARK"},
        "trials": [asdict(row) for row in trials],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run_benchmark(args.repeats)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nJSON report written to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
