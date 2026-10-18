"""Step 30.5.3: controlled measurement of AdaptivePlanner strategy value.

This benchmark tests whether the real AdaptivePlanner selects a feasible,
lower-action strategy when the supplied alternatives have different costs.
It is intentionally a small controlled ablation, not P0 acceptance.

Run:
    pytest -q tests/test_step30_5_3_adaptive_strategy_value.py
    python experiments/step30_5_3_adaptive_strategy_value.py --repeats 3 --json-out artifacts/step30_5_3_adaptive_strategy_value.json
"""
from __future__ import annotations

import argparse
import json
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
    initial_state: dict[str, Any]
    goal_state: dict[str, Any]
    goal: str
    actions: tuple[Action, ...]
    strategy_steps: tuple[tuple[str, ...], ...]
    baseline_steps: tuple[str, ...]


def action(name: str, preconditions: dict[str, Any], effects: dict[str, Any]) -> Action:
    return Action(
        name=name,
        description=f"Controlled benchmark action: {name}",
        handler=lambda **kwargs: {"success": True},
        preconditions=preconditions,
        effects=effects,
    )


def make_strategy(name: str, goal: str, steps: list[str], successes: int = 5, failures: int = 0) -> Strategy:
    strategy = Strategy(name=name, goal=goal, steps=steps)
    for i in range(successes):
        strategy.record_success(
            experience_id=f"step30-5-3-{name}-success-{i + 1}",
            action_count=len(steps),
            progress_delta=1.0,
        )
    for i in range(failures):
        strategy.record_failure(
            experience_id=f"step30-5-3-{name}-failure-{i + 1}",
            action_count=len(steps),
            progress_delta=0.0,
        )
    return strategy


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="unlock_door",
            initial_state={"key": "available", "door": "locked", "route": "unknown"},
            goal_state={"door": "unlocked"},
            goal="unlock door",
            actions=(
                action("inspect_key", {"key": "available"}, {"key": "inspected"}),
                action("prepare_key", {"key": "inspected"}, {"key": "ready"}),
                action("unlock_door", {"key": "ready", "door": "locked"}, {"door": "unlocked"}),
                action("use_master_key", {"key": "available", "door": "locked"}, {"door": "unlocked"}),
            ),
            strategy_steps=(
                ("inspect_key", "prepare_key", "unlock_door"),
                ("use_master_key",),
            ),
            baseline_steps=("inspect_key", "prepare_key", "unlock_door"),
        ),
        Scenario(
            name="turn_on_light",
            initial_state={"power": "available", "light": "off"},
            goal_state={"light": "on"},
            goal="turn on light",
            actions=(
                action("inspect_switch", {"power": "available"}, {"switch": "inspected"}),
                action("connect_switch", {"switch": "inspected"}, {"switch": "connected"}),
                action("turn_on_light", {"switch": "connected", "light": "off"}, {"light": "on"}),
                action("use_remote", {"power": "available", "light": "off"}, {"light": "on"}),
            ),
            strategy_steps=(
                ("inspect_switch", "connect_switch", "turn_on_light"),
                ("use_remote",),
            ),
            baseline_steps=("inspect_switch", "connect_switch", "turn_on_light"),
        ),
        Scenario(
            name="deliver_package",
            initial_state={"package": "at_origin", "location": "origin", "van": "available"},
            goal_state={"package": "delivered"},
            goal="deliver package",
            actions=(
                action("pick_up_package", {"package": "at_origin"}, {"package": "carried"}),
                action("drive_to_customer", {"package": "carried", "location": "origin", "van": "available"},
                       {"location": "customer"}),
                action("deliver_package", {"package": "carried", "location": "customer"},
                       {"package": "delivered"}),
                action("instant_dispatch", {"package": "at_origin"}, {"package": "delivered"}),
            ),
            strategy_steps=(
                ("pick_up_package", "drive_to_customer", "deliver_package"),
                ("instant_dispatch",),
            ),
            baseline_steps=("pick_up_package", "drive_to_customer", "deliver_package"),
        ),
        Scenario(
            name="make_tea",
            initial_state={"water": "filled", "kettle": "working", "cup": "tea_bag_added", "tea": "not_ready"},
            goal_state={"tea": "ready"},
            goal="make tea",
            actions=(
                action("inspect_kettle", {"kettle": "working"}, {"kettle": "inspected"}),
                action("prepare_kettle", {"kettle": "inspected", "water": "filled"}, {"water": "boiled"}),
                action("brew_tea", {"water": "boiled", "cup": "tea_bag_added"}, {"tea": "ready"}),
                action("use_tea_maker", {"water": "filled", "cup": "tea_bag_added"}, {"tea": "ready"}),
            ),
            strategy_steps=(
                ("inspect_kettle", "prepare_kettle", "brew_tea"),
                ("use_tea_maker",),
            ),
            baseline_steps=("inspect_kettle", "prepare_kettle", "brew_tea"),
        ),
    ]


def simulate(initial_state: dict[str, Any], plan: list[Action]) -> tuple[bool, dict[str, Any], int]:
    state = dict(initial_state)
    attempted = 0
    for item in plan:
        attempted += 1
        if not item.can_execute(state):
            return False, state, attempted
        state.update(item.effects)
    return True, state, attempted


def resolve_names(names: tuple[str, ...], actions: tuple[Action, ...]) -> list[Action]:
    by_name = {item.name: item for item in actions}
    return [by_name[name] for name in names]


@dataclass
class Trial:
    scenario: str
    mode: str
    repeat: int
    plan: list[str]
    action_count: int
    completed: bool
    verified: bool


class EmptyFallbackPlanner:
    """Fallback stub exposing the action registry required by AdaptivePlanner."""

    def __init__(self, actions: tuple[Action, ...]):
        self.actions = list(actions)

    def plan(self, world_state: dict[str, Any], goal_state: dict[str, Any]) -> list[Action]:
        return []


def run_benchmark(repeats: int = 3) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    trials: list[Trial] = []
    for case in scenarios():
        strategies = [
            make_strategy(f"{case.name}_long_route", case.goal, list(case.strategy_steps[0])),
            make_strategy(f"{case.name}_short_route", case.goal, list(case.strategy_steps[1])),
        ]
        for repeat in range(1, repeats + 1):
            baseline_plan = resolve_names(case.baseline_steps, case.actions)
            ok, final_state, count = simulate(case.initial_state, baseline_plan)
            verified = ok and all(final_state.get(k) == v for k, v in case.goal_state.items())
            trials.append(Trial(case.name, "fixed_baseline", repeat,
                                [item.name for item in baseline_plan], count, verified, verified))

            planner = AdaptivePlanner(
                planner=EmptyFallbackPlanner(case.actions),
                strategy_selector=StrategySelector(),
            )
            plan = planner.plan(
                world_state=dict(case.initial_state),
                goal_state=dict(case.goal_state),
                goal=case.goal,
                strategies=list(strategies),
            )
            ok, final_state, count = simulate(case.initial_state, plan)
            verified = ok and all(final_state.get(k) == v for k, v in case.goal_state.items())
            trials.append(Trial(case.name, "adaptive", repeat,
                                [item.name for item in plan], count, verified, verified))

    summaries = {}
    for mode in ("fixed_baseline", "adaptive"):
        rows = [row for row in trials if row.mode == mode]
        summaries[mode] = {
            "runs": len(rows),
            "completed": sum(row.completed for row in rows),
            "completion_rate": sum(row.completed for row in rows) / len(rows),
            "verified": sum(row.verified for row in rows),
            "verification_rate": sum(row.verified for row in rows) / len(rows),
            "mean_actions": sum(row.action_count for row in rows) / len(rows),
        }
    delta = summaries["adaptive"]["mean_actions"] - summaries["fixed_baseline"]["mean_actions"]
    return {
        "benchmark": "step30_5_3_adaptive_strategy_value_v1",
        "purpose": "measure whether adaptive strategy selection reduces actions against a deliberately fixed longer route",
        "evaluation_runs": len(trials),
        "repeats": repeats,
        "scenarios": [case.name for case in scenarios()],
        "summary": {
            **summaries,
            "adaptive_minus_baseline_mean_actions": round(delta, 6),
            "relative_action_reduction": round(-delta / summaries["fixed_baseline"]["mean_actions"], 6)
                if summaries["fixed_baseline"]["mean_actions"] else 0.0,
        },
        "limitations": [
            "controlled synthetic state/action models",
            "the baseline is intentionally fixed to a longer route and is not a comparison against an optimal planner",
            "strategy histories are synthetic priors, not learned from real-world interactions",
            "does not measure security or P0 acceptance",
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
