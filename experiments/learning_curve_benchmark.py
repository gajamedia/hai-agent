"""Learning-curve benchmark: does experience actually improve outcomes?

Compares two arms over N sequential episodes of the same task, using the real
``Agent`` (memory, strategy store, AdaptivePlanner) and a hidden-fault room:

* ``no_learning`` - a brand-new Agent for every episode (no carried memory).
* ``learning``    - one Agent kept across all episodes (memory/strategies persist).

The environment has a hidden fault: the "spare key" fits only some episodes
(``spare_fit_rate``). Two routes reach the goal:

    spare_route : find_spare_key -> unlock_door              (2 steps, can fail)
    box_route   : open_box -> take_key -> unlock_door        (3 steps, reliable)

Metrics per episode: success, number of actions executed, failed actions.
Deterministic (seeded). No LLM, no network.

Run:
    python experiments/learning_curve_benchmark.py --episodes 30 --seeds 5
    python experiments/learning_curve_benchmark.py --spare-fit-rate 0.0 --json-out artifacts/learning_curve.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.learning.strategy import Strategy

GOAL_TEXT = "unlock the door"
GOAL_STATE = {"door": "unlocked"}


class FaultyRoom:
    """Room where the spare key may not fit the door (hidden from the agent)."""

    def __init__(self, spare_fits: bool) -> None:
        self.spare_fits = spare_fits
        self.state: dict[str, Any] = {
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": "available",
            "key_kind": "none",
        }

    def observe(self) -> dict[str, Any]:
        return {k: v for k, v in self.state.items() if k != "key_kind"}

    def execute(self, name: str) -> bool:
        s = self.state
        if name == "open_box" and s["box"] == "closed":
            s["box"] = "open"
            return True
        if name == "take_key" and s["box"] == "open" and s["key"] == "inside_box":
            s["key"], s["key_kind"] = "agent", "real"
            return True
        if name == "find_spare_key" and s["spare_key"] == "available":
            s["spare_key"], s["key"], s["key_kind"] = "found", "agent", "spare"
            return True
        if name == "unlock_door" and s["key"] == "agent" and s["door"] == "locked":
            if s["key_kind"] == "spare" and not self.spare_fits:
                return False  # hidden fault
            s["door"] = "unlocked"
            return True
        return False


def _action(name: str, pre: dict[str, Any], eff: dict[str, Any]) -> Action:
    return Action(
        name=name,
        description=name,
        handler=lambda **kw: {"success": True},
        preconditions=pre,
        effects=eff,
    )


def build_actions() -> list[Action]:
    # find_spare_key is registered first so the base planner prefers the
    # (hidden-fault) 2-step route; learning must be what steers away from it.
    return [
        _action("find_spare_key", {"spare_key": "available"}, {"key": "agent"}),
        _action("open_box", {"box": "closed"}, {"box": "open"}),
        _action("take_key", {"box": "open", "key": "inside_box"}, {"key": "agent"}),
        _action("unlock_door", {"key": "agent", "door": "locked"}, {"door": "unlocked"}),
    ]


def build_agent(name: str) -> Agent:
    agent = Agent(name)
    agent.register_actions(build_actions())
    return agent


def run_episode(
    agent: Agent, room: FaultyRoom, episode_id: str, evolve_on_failure: bool = False
) -> dict[str, Any]:
    """One episode: plan -> execute -> record -> complete/fail experience."""
    goal = Goal(description=GOAL_TEXT, target_state=dict(GOAL_STATE))
    agent.set_goal(goal)
    agent.start_experience(experience_id=episode_id, goal=GOAL_TEXT)

    strategies = list(agent.get_registered_strategies())
    plan = agent.choose_plan(
        world_state=room.observe(),
        goal_state=GOAL_STATE,
        strategies=strategies,
        goal=GOAL_TEXT,
    )

    executed = 0
    failed_actions = 0
    for act in plan:
        before = room.observe()
        ok = room.execute(act.name)
        executed += 1
        agent.record_experience(
            action=act.name,
            observation_before=before,
            observation_after=room.observe(),
            result={"success": ok},
        )
        if not ok:
            failed_actions += 1
            break

    success = room.state["door"] == "unlocked"
    if success:
        agent.complete_experience()
    else:
        agent.fail_experience()
        if evolve_on_failure:
            agent.evolve_strategy_from_experience()
    return {
        "success": success,
        "actions": executed,
        "failed_actions": failed_actions,
        "plan": [a.name for a in plan],
    }


def run_arm(arm: str, episodes: int, seed: int, spare_fit_rate: float) -> list[dict[str, Any]]:
    evolve = arm == "learning_evolve"
    rng = random.Random(seed)
    fits = [rng.random() < spare_fit_rate for _ in range(episodes)]
    persistent = build_agent(f"learner-{seed}") if arm in ("learning", "learning_evolve") else None
    rows = []
    for i in range(episodes):
        agent = persistent if persistent is not None else build_agent(f"fresh-{seed}-{i}")
        rows.append(run_episode(agent, FaultyRoom(fits[i]), f"{arm}-{seed}-{i}", evolve))
    return rows


@dataclass
class ArmSummary:
    arm: str
    success_rate_first_third: float
    success_rate_last_third: float
    success_rate_overall: float
    mean_actions_overall: float
    mean_failed_actions: float


def summarize(arm: str, all_seeds: list[list[dict[str, Any]]]) -> ArmSummary:
    n = len(all_seeds[0])
    third = max(1, n // 3)

    def rate(rows: list[dict[str, Any]]) -> float:
        return sum(r["success"] for r in rows) / len(rows)

    def mean_over(fn) -> float:
        return statistics.mean(fn(rows) for rows in all_seeds)

    return ArmSummary(
        arm=arm,
        success_rate_first_third=mean_over(lambda r: rate(r[:third])),
        success_rate_last_third=mean_over(lambda r: rate(r[-third:])),
        success_rate_overall=mean_over(rate),
        mean_actions_overall=mean_over(lambda r: statistics.mean(x["actions"] for x in r)),
        mean_failed_actions=mean_over(lambda r: statistics.mean(x["failed_actions"] for x in r)),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--spare-fit-rate", type=float, default=0.0,
                   help="Probability the spare key fits (0.0 = always a decoy).")
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    results: dict[str, list[list[dict[str, Any]]]] = {}
    for arm in ("no_learning", "learning", "learning_evolve"):
        results[arm] = [run_arm(arm, a.episodes, s, a.spare_fit_rate) for s in range(a.seeds)]

    summaries = [summarize(arm, rows) for arm, rows in results.items()]
    print(f"episodes={a.episodes} seeds={a.seeds} spare_fit_rate={a.spare_fit_rate}")
    print(f"{'arm':<15} {'first1/3':>9} {'last1/3':>9} {'overall':>9} {'avg_act':>8} {'avg_fail':>9}")
    for s in summaries:
        print(f"{s.arm:<15} {s.success_rate_first_third:9.2f} {s.success_rate_last_third:9.2f} "
              f"{s.success_rate_overall:9.2f} {s.mean_actions_overall:8.2f} {s.mean_failed_actions:9.2f}")

    sample = results["learning_evolve"][0]
    print("\nlearning_evolve arm, seed 0 plan per episode:")
    print("  " + " | ".join(f"{i}:{'>'.join(r['plan'])}{'' if r['success'] else '(FAIL)'}"
                            for i, r in enumerate(sample[:8])))

    if a.json_out:
        out = Path(a.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summaries": [asdict(s) for s in summaries],
                                   "raw": results}, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
