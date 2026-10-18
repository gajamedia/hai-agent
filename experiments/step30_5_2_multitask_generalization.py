"""Step 30.5.2: multi-task generalization benchmark for AdaptivePlanner.

Run from repository root:
    pytest -q tests/test_step30_5_2_multitask_generalization.py
    python experiments/step30_5_2_multitask_generalization.py --repeats 3 --json-out artifacts/step30_5_2_multitask_generalization.json

This is a focused development benchmark, not P0 acceptance. It invokes the project's
real AdaptivePlanner, Action, Strategy, and StrategySelector. A small deterministic
reference planner is used only as the baseline/fallback so the benchmark remains
reproducible across unrelated Planner implementation changes.
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
class TaskCase:
    task: str
    profile: str
    holdout: bool
    initial_state: dict[str, Any]
    goal_state: dict[str, Any]
    goal: str
    actions: tuple[Action, ...]
    strategies: tuple[Strategy, ...]


@dataclass
class Trial:
    task: str
    profile: str
    holdout: bool
    mode: str
    repeat: int
    completed: bool
    verified: bool
    plan: list[str]
    actions_attempted: int
    fallback_expected: bool


class ReferencePlanner:
    """Small deterministic breadth-first planner for the benchmark harness."""

    def __init__(self, actions: list[Action]):
        self.actions = list(actions)

    def plan(self, world_state: dict[str, Any], goal_state: dict[str, Any]) -> list[Action]:
        if all(world_state.get(k) == v for k, v in goal_state.items()):
            return []
        queue = deque([(dict(world_state), [])])
        seen = {self._key(world_state)}
        while queue:
            state, path = queue.popleft()
            for action in self.actions:
                if not action.can_execute(state):
                    continue
                next_state = dict(state)
                next_state.update(action.effects)
                if next_state == state:
                    continue
                next_path = path + [action]
                if all(next_state.get(k) == v for k, v in goal_state.items()):
                    return next_path
                key = self._key(next_state)
                if key not in seen:
                    seen.add(key)
                    queue.append((next_state, next_path))
        return []

    @staticmethod
    def _key(state: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(k), repr(v)) for k, v in state.items()))


def _action(name: str, pre: dict[str, Any], effects: dict[str, Any]) -> Action:
    return Action(
        name=name,
        description=f"Benchmark action: {name}",
        handler=lambda **kwargs: {"success": True},
        preconditions=pre,
        effects=effects,
    )


def _strategy(name: str, goal: str, steps: list[str], episodes: int = 3) -> Strategy:
    strategy = Strategy(name=name, goal=goal, steps=list(steps))
    # Controlled prior evidence: both candidate routes are observed in the
    # shared training profile; evaluation never updates these strategy scores.
    for index in range(episodes):
        strategy.record_success(
            experience_id=f"step30-5-2-training-{name}-{index + 1}",
            action_count=len(steps),
            progress_delta=1.0,
        )
    return strategy


def build_cases() -> list[TaskCase]:
    cases: list[TaskCase] = []

    # Task 1: unlock a door via either the box key or spare key.
    goal = "unlock the door"
    actions = (
        _action("open_box", {"box": "closed"}, {"box": "open"}),
        _action("take_key", {"box": "open", "key": "inside_box"}, {"key": "agent"}),
        _action("find_spare_key", {"spare_key": "available"}, {"spare_key": "found", "key": "agent"}),
        _action("unlock_door", {"key": "agent", "door": "locked"}, {"door": "unlocked"}),
    )
    strategies = (
        _strategy("door_box_route", goal, ["open_box", "take_key", "unlock_door"]),
        _strategy("door_spare_route", goal, ["find_spare_key", "unlock_door"]),
    )
    for profile, holdout, spare in (
        ("both_keys_available", False, "available"),
        ("spare_key_unavailable", True, "unavailable"),
    ):
        cases.append(TaskCase(
            "unlock_door", profile, holdout,
            {"box": "closed", "key": "inside_box", "spare_key": spare, "door": "locked"},
            {"door": "unlocked"}, goal, actions, strategies,
        ))

    # Task 2: turn on a light using a primary or backup switch.
    goal = "turn on the light"
    actions = (
        _action("locate_primary_switch", {"primary_switch": "available"}, {"primary_switch": "found"}),
        _action("locate_backup_switch", {"backup_switch": "available"}, {"backup_switch": "found"}),
        _action("turn_on_primary", {"primary_switch": "found", "light": "off"}, {"light": "on"}),
        _action("turn_on_backup", {"backup_switch": "found", "light": "off"}, {"light": "on"}),
    )
    strategies = (
        _strategy("light_primary_route", goal, ["locate_primary_switch", "turn_on_primary"]),
        _strategy("light_backup_route", goal, ["locate_backup_switch", "turn_on_backup"]),
    )
    for profile, holdout, primary in (
        ("both_switches_available", False, "available"),
        ("primary_switch_unavailable", True, "unavailable"),
    ):
        cases.append(TaskCase(
            "turn_on_light", profile, holdout,
            {"primary_switch": primary, "backup_switch": "available", "light": "off"},
            {"light": "on"}, goal, actions, strategies,
        ))

    # Task 3: deliver a package by van or by walking.
    goal = "deliver the package"
    actions = (
        _action("pick_up_package", {"package": "at_origin"}, {"package": "carried"}),
        _action("drive_to_customer", {"package": "carried", "location": "origin", "van": "available"},
                {"location": "customer"}),
        _action("walk_to_customer", {"package": "carried", "location": "origin", "path": "clear"},
                {"location": "customer"}),
        _action("deliver_package", {"package": "carried", "location": "customer"},
                {"package": "delivered"}),
    )
    strategies = (
        _strategy("delivery_van_route", goal, ["pick_up_package", "drive_to_customer", "deliver_package"]),
        _strategy("delivery_walk_route", goal, ["pick_up_package", "walk_to_customer", "deliver_package"]),
    )
    for profile, holdout, van in (
        ("van_and_walkway_available", False, "available"),
        ("van_unavailable", True, "unavailable"),
    ):
        cases.append(TaskCase(
            "deliver_package", profile, holdout,
            {"package": "at_origin", "location": "origin", "van": van, "path": "clear"},
            {"package": "delivered"}, goal, actions, strategies,
        ))

    # Task 4: make tea with an electric kettle or a stove.
    goal = "make tea"
    actions = (
        _action("fill_kettle", {"water": "empty"}, {"water": "filled"}),
        _action("boil_in_kettle", {"water": "filled", "kettle": "working"}, {"water": "boiled"}),
        _action("boil_on_stove", {"water": "filled", "stove": "available"}, {"water": "boiled"}),
        _action("add_tea_bag", {"cup": "empty"}, {"cup": "tea_bag_added"}),
        _action("brew_tea", {"water": "boiled", "cup": "tea_bag_added"}, {"tea": "ready"}),
    )
    strategies = (
        _strategy("tea_kettle_route", goal, ["fill_kettle", "boil_in_kettle", "add_tea_bag", "brew_tea"]),
        _strategy("tea_stove_route", goal, ["fill_kettle", "boil_on_stove", "add_tea_bag", "brew_tea"]),
    )
    for profile, holdout, kettle in (
        ("kettle_and_stove_available", False, "working"),
        ("kettle_broken", True, "broken"),
    ):
        cases.append(TaskCase(
            "make_tea", profile, holdout,
            {"water": "empty", "kettle": kettle, "stove": "available", "cup": "empty", "tea": "not_ready"},
            {"tea": "ready"}, goal, actions, strategies,
        ))

    return cases


def _simulate(initial_state: dict[str, Any], plan: list[Action]) -> tuple[bool, dict[str, Any], int]:
    state = dict(initial_state)
    attempted = 0
    for action in plan:
        attempted += 1
        if not action.can_execute(state):
            return False, state, attempted
        state.update(action.effects)
    return True, state, attempted


def _trial(case: TaskCase, mode: str, repeat: int) -> Trial:
    actions = list(case.actions)
    if mode == "static":
        planner = ReferencePlanner(actions)
        plan = planner.plan(dict(case.initial_state), dict(case.goal_state))
        fallback_expected = True
    elif mode == "adaptive":
        planner = ReferencePlanner(actions)
        adaptive = AdaptivePlanner(planner=planner, strategy_selector=StrategySelector())
        plan = adaptive.plan(
            world_state=dict(case.initial_state),
            goal_state=dict(case.goal_state),
            goal=case.goal,
            strategies=list(case.strategies),
        )
        # A holdout should exclude at least one strategy route by feasibility.
        fallback_expected = False
    else:
        raise ValueError("mode must be 'static' or 'adaptive'")

    executable, final_state, attempted = _simulate(case.initial_state, plan)
    verified = executable and all(
        final_state.get(key) == value for key, value in case.goal_state.items()
    )
    return Trial(
        task=case.task,
        profile=case.profile,
        holdout=case.holdout,
        mode=mode,
        repeat=repeat,
        completed=verified,
        verified=verified,
        plan=[action.name for action in plan],
        actions_attempted=attempted,
        fallback_expected=fallback_expected,
    )


def _summary(rows: list[Trial]) -> dict[str, Any]:
    count = len(rows)
    return {
        "runs": count,
        "completed": sum(row.completed for row in rows),
        "completion_rate": sum(row.completed for row in rows) / count if count else None,
        "verified": sum(row.verified for row in rows),
        "verification_rate": sum(row.verified for row in rows) / count if count else None,
        "mean_actions_attempted": (
            sum(row.actions_attempted for row in rows) / count if count else None
        ),
        "by_task": {
            task: {
                "runs": len(task_rows),
                "completion_rate": sum(row.completed for row in task_rows) / len(task_rows),
                "verification_rate": sum(row.verified for row in task_rows) / len(task_rows),
            }
            for task in sorted({row.task for row in rows})
            for task_rows in [[row for row in rows if row.task == task]]
        },
        "holdout": {
            "runs": len(holdout_rows),
            "completion_rate": sum(row.completed for row in holdout_rows) / len(holdout_rows)
            if holdout_rows else None,
            "verification_rate": sum(row.verified for row in holdout_rows) / len(holdout_rows)
            if holdout_rows else None,
        } if (holdout_rows := [row for row in rows if row.holdout]) else {
            "runs": 0, "completion_rate": None, "verification_rate": None
        },
    }


def run_benchmark(repeats: int = 3) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    cases = build_cases()
    trials: list[Trial] = []
    for case in cases:
        for repeat in range(1, repeats + 1):
            for mode in ("static", "adaptive"):
                trials.append(_trial(case, mode, repeat))

    static = _summary([row for row in trials if row.mode == "static"])
    adaptive = _summary([row for row in trials if row.mode == "adaptive"])
    return {
        "benchmark": "step30_5_2_multitask_generalization_v1",
        "purpose": "evaluate context-feasible strategy selection across four task types and held-out initial states",
        "evaluation_runs": len(trials),
        "repeats_per_task_profile_mode": repeats,
        "task_types": sorted({case.task for case in cases}),
        "profiles": [
            {"task": case.task, "profile": case.profile, "split": "holdout" if case.holdout else "training_context"}
            for case in cases
        ],
        "conditions": {
            "static": "deterministic reference planner only",
            "adaptive": "project AdaptivePlanner + StrategySelector, with the reference planner as fallback",
            "training_evidence": "three controlled success records per candidate route, based on the route's declared training context",
            "evaluation_updates_training": False,
            "p0_acceptance": "NOT_EVALUATED_BY_THIS_BENCHMARK",
            "security_metrics": "not measured",
        },
        "summary": {
            "static": static,
            "adaptive": adaptive,
            "adaptive_minus_static_mean_actions": round(
                adaptive["mean_actions_attempted"] - static["mean_actions_attempted"], 6
            ),
            "completion_rate_difference": round(
                adaptive["completion_rate"] - static["completion_rate"], 6
            ),
            "holdout_completion_rate_difference": round(
                adaptive["holdout"]["completion_rate"] - static["holdout"]["completion_rate"], 6
            ),
        },
        "trials": [asdict(row) for row in trials],
        "p0_acceptance": {
            "30_controlled_tasks_x_3_runs": False,
            "correct_completion_threshold": 0.85,
            "verification_threshold": 1.0,
            "unauthorized_high_risk_actions_allowed": 0,
            "harmful_duplicate_side_effects_allowed": 0,
            "recoverable_failure_recovery_threshold": 0.80,
            "status": "NOT_EVALUATED_BY_THIS_BENCHMARK",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run_benchmark(repeats=args.repeats)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nJSON report written to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
