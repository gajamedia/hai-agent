"""Step 30.5 diagnostic benchmark: meta-learning generalization across task contexts.

This benchmark deliberately trains the existing StrategySelector on one observed
context (spare key available), then evaluates both the in-distribution context and
a held-out context (spare key unavailable). It is designed to reveal whether the
current goal-level selector generalizes when action preconditions change.

Run from repository root:
    pytest -q tests/test_step30_multitask_generalization_benchmark.py
    python experiments/step30_multitask_generalization_benchmark.py --repeats 3 --training-repeats 3 --json-out artifacts/step30_multitask_benchmark.json

This is a diagnostic pilot, not P0 acceptance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.agent_decision import AgentDecision, DecisionMode
from agent.cognition.active_explorer import ActiveExplorer
from agent.cognition.unified_executor import UnifiedDecisionExecutor
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector
from agent.world.room import RoomEnvironment

GOAL_DESCRIPTION = "unlock the door"
GOAL_STATE = {"door": "unlocked"}
BOX_ROUTE = ("open_box", "take_key", "unlock_door")
SPARE_ROUTE = ("find_spare_key", "unlock_door")


@dataclass
class TrialResult:
    mode: str
    task_profile: str
    holdout: bool
    trial: int
    completed: bool
    verified: bool
    initial_strategy: str
    final_strategy: str
    initial_attempt_success: bool
    recovery_attempted: bool
    recovery_success: bool
    actions_attempted: int
    final_state: dict[str, Any]
    failure_types: list[str]


def _decision() -> AgentDecision:
    return AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason="Controlled Step 30.5 multi-task generalization benchmark.",
        uncertainty=0.1,
        expected_information_gain=0.0,
        risk=0.1,
    )


def _build_runtime(name: str, spare_key_available: bool = True):
    environment = RoomEnvironment(spare_key_available=spare_key_available)
    agent = Agent(name=name)
    agent.register_actions([
        Action(
            name="open_box", description="Open the box",
            handler=lambda **kwargs: environment.execute("open_box"),
            preconditions={"box": "closed"}, effects={"box": "open"},
        ),
        Action(
            name="take_key", description="Take the key from the box",
            handler=lambda **kwargs: environment.execute("take_key"),
            preconditions={"box": "open", "key": "inside_box"},
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key", description="Find the spare key",
            handler=lambda **kwargs: environment.execute("find_spare_key"),
            preconditions={"spare_key": "available"},
            effects={"key": "agent", "spare_key": "found"},
        ),
        Action(
            name="unlock_door", description="Unlock the door",
            handler=lambda **kwargs: environment.execute("unlock_door"),
            preconditions={"key": "agent", "door": "locked"},
            effects={"door": "unlocked"},
        ),
    ])
    executor = UnifiedDecisionExecutor(explorer=ActiveExplorer())
    goal = Goal(GOAL_DESCRIPTION, target_state=dict(GOAL_STATE))
    return environment, agent, executor, goal


def _execute_route(
    route: tuple[str, ...], *, experience_id: str, name: str,
    spare_key_available: bool = True,
):
    environment, agent, executor, goal = _build_runtime(
        name=name, spare_key_available=spare_key_available
    )
    result = executor.execute_plan(
        decision=_decision(), agent=agent, environment=environment, goal=goal,
        actions=[agent.actions[action_name] for action_name in route],
        world_state=environment.observe(), experience_id=experience_id,
    )
    final_state = environment.observe()
    verified = all(final_state.get(k) == v for k, v in GOAL_STATE.items())
    return result, verified, final_state, len(result.plan_execution_results or ())


def _train_strategy(strategy: Strategy, route: tuple[str, ...], *, training_repeats: int):
    successes = failures = action_total = 0
    for episode in range(1, training_repeats + 1):
        event_id = f"step30-5-training-{strategy.name}-{episode}"
        result, verified, _state, attempted = _execute_route(
            route, experience_id=event_id, name=event_id,
            spare_key_available=True,
        )
        action_total += attempted
        if result.success and verified:
            strategy.record_success(
                experience_id=event_id, action_count=attempted, progress_delta=1.0
            )
            successes += 1
        else:
            strategy.record_failure(
                experience_id=event_id, action_count=attempted, progress_delta=0.0
            )
            failures += 1
    return {
        "strategy": strategy.name,
        "training_context": "spare_key_available",
        "training_episodes": training_repeats,
        "successes": successes,
        "failures": failures,
        "empirical_success_rate": successes / training_repeats,
        "mean_actions": action_total / training_repeats,
        "evidence_source": "actual RoomEnvironment execution and final-state verification",
    }


def _train_candidates(training_repeats: int):
    box_strategy = Strategy(
        name="open_box_then_take_key", goal=GOAL_DESCRIPTION, steps=list(BOX_ROUTE)
    )
    spare_strategy = Strategy(
        name="find_spare_key", goal=GOAL_DESCRIPTION, steps=list(SPARE_ROUTE)
    )
    reports = [
        _train_strategy(box_strategy, BOX_ROUTE, training_repeats=training_repeats),
        _train_strategy(spare_strategy, SPARE_ROUTE, training_repeats=training_repeats),
    ]
    selected = StrategySelector().select(
        [box_strategy, spare_strategy], goal=GOAL_DESCRIPTION
    )
    if selected is None:
        raise RuntimeError("StrategySelector returned no trained strategy.")
    return box_strategy, spare_strategy, selected, reports


def run_trial(
    mode: str, task_profile: str, trial: int, *,
    selected_strategy: Strategy, spare_key_available: bool,
) -> TrialResult:
    if mode not in {"static", "adaptive"}:
        raise ValueError("mode must be 'static' or 'adaptive'.")
    environment, agent, executor, goal = _build_runtime(
        name=f"step30-5-eval-{mode}-{task_profile}-{trial}",
        spare_key_available=spare_key_available,
    )
    initial_result = executor.execute_action(
        decision=_decision(), action=agent.actions["unlock_door"],
        agent=agent, environment=environment, goal=goal,
        experience_id=f"step30-5-eval-{mode}-{task_profile}-{trial}-initial",
    )
    initial_success = bool(initial_result.success)
    failures: list[str] = []
    if not initial_success and isinstance(initial_result.direct_result, dict):
        failure_type = initial_result.direct_result.get("failure_type")
        if failure_type:
            failures.append(str(failure_type))

    if mode == "static":
        final_strategy = "fixed_fallback_open_box_then_take_key"
        route = BOX_ROUTE
    else:
        final_strategy = selected_strategy.name
        route = tuple(selected_strategy.steps)

    recovery_attempted = not initial_success
    recovery_success = False
    actions_attempted = 1
    if recovery_attempted:
        recovery_goal = Goal(GOAL_DESCRIPTION, target_state=dict(GOAL_STATE))
        result = executor.execute_plan(
            decision=_decision(), agent=agent, environment=environment,
            goal=recovery_goal, actions=[agent.actions[n] for n in route],
            world_state=environment.observe(),
            experience_id=f"step30-5-eval-{mode}-{task_profile}-{trial}-recovery",
        )
        recovery_success = bool(result.success)
        actions_attempted += len(result.plan_execution_results or ())
        for step in result.plan_execution_results or ():
            step_result = getattr(step, "result", None)
            if isinstance(step_result, dict) and step_result.get("failure_type"):
                failures.append(str(step_result["failure_type"]))

    final_state = environment.observe()
    verified = all(final_state.get(k) == v for k, v in GOAL_STATE.items())
    return TrialResult(
        mode=mode, task_profile=task_profile,
        holdout=(task_profile == "spare_key_unavailable"),
        trial=trial, completed=verified, verified=verified,
        initial_strategy="direct_unlock", final_strategy=final_strategy,
        initial_attempt_success=initial_success,
        recovery_attempted=recovery_attempted, recovery_success=recovery_success,
        actions_attempted=actions_attempted, final_state=final_state,
        failure_types=failures,
    )


def _summarize(rows: list[TrialResult]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"runs": 0, "completion_rate": None, "verification_rate": None}
    recoveries = [r for r in rows if r.recovery_attempted]
    return {
        "runs": n,
        "completed": sum(r.completed for r in rows),
        "completion_rate": sum(r.completed for r in rows) / n,
        "verified": sum(r.verified for r in rows),
        "verification_rate": sum(r.verified for r in rows) / n,
        "recovery_attempts": len(recoveries),
        "recovery_attempt_rate": len(recoveries) / n,
        "recovery_successes": sum(r.recovery_success for r in recoveries),
        "recovery_success_rate_when_attempted": (
            sum(r.recovery_success for r in recoveries) / len(recoveries)
            if recoveries else None
        ),
        "mean_actions_attempted": sum(r.actions_attempted for r in rows) / n,
        "unauthorized_high_risk_actions": None,
        "harmful_duplicate_side_effects": None,
        "note": "Security and duplicate-side-effect metrics are not exercised by this benchmark.",
    }


def run_benchmark(repeats: int = 3, training_repeats: int = 3) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if training_repeats < 1:
        raise ValueError("training_repeats must be at least 1.")

    box_strategy, spare_strategy, selected, training = _train_candidates(training_repeats)
    selector_scores = {
        box_strategy.name: round(
            box_strategy.success_rate - box_strategy.average_actions * 0.01
            - box_strategy.reflection_penalty, 6
        ),
        spare_strategy.name: round(
            spare_strategy.success_rate - spare_strategy.average_actions * 0.01
            - spare_strategy.reflection_penalty, 6
        ),
    }

    results: list[TrialResult] = []
    profiles = [
        ("spare_key_available", True),
        ("spare_key_unavailable", False),
    ]
    for task_profile, spare_available in profiles:
        for trial in range(1, repeats + 1):
            for mode in ("static", "adaptive"):
                results.append(run_trial(
                    mode, task_profile, trial, selected_strategy=selected,
                    spare_key_available=spare_available,
                ))

    summaries = {}
    for mode in ("static", "adaptive"):
        mode_rows = [r for r in results if r.mode == mode]
        summaries[mode] = _summarize(mode_rows)
        summaries[mode]["by_task_profile"] = {
            profile: _summarize([
                r for r in mode_rows if r.task_profile == profile
            ])
            for profile, _available in profiles
        }
        summaries[mode]["holdout"] = _summarize([
            r for r in mode_rows if r.holdout
        ])

    static_mean = summaries["static"]["mean_actions_attempted"]
    adaptive_mean = summaries["adaptive"]["mean_actions_attempted"]
    return {
        "benchmark": "step30_multitask_generalization_diagnostic_v1",
        "purpose": (
            "measure whether the current goal-level StrategySelector generalizes "
            "from training with an available spare key to a held-out unavailable-key context"
        ),
        "task_count": 2,
        "repeats_per_mode_per_profile": repeats,
        "evaluation_runs": len(results),
        "training": {
            "context": "spare_key_available only",
            "episodes_total": training_repeats * 2,
            "repeats_per_strategy": training_repeats,
            "candidate_strategies": training,
            "selector_scores": selector_scores,
            "selected_strategy": selected.name,
            "evaluation_updates_training": False,
        },
        "task_profiles": [
            {
                "name": "spare_key_available",
                "split": "in_distribution",
                "initial_state": {"door": "locked", "box": "closed", "key": "inside_box",
                                  "spare_key": "available", "agent_location": "room"},
            },
            {
                "name": "spare_key_unavailable",
                "split": "holdout",
                "initial_state": {"door": "locked", "box": "closed", "key": "inside_box",
                                  "spare_key": "unavailable", "agent_location": "room"},
            },
        ],
        "conditions": {
            "static": "same failed direct attempt, then fixed open_box/take_key/unlock_door recovery plan",
            "adaptive": "same failed direct attempt and recovery budget; one strategy selected from actual training outcomes",
            "training_evidence": "actual RoomEnvironment execution and final-state verification; no synthetic counters",
            "holdout_policy": "the unavailable-spare-key profile is excluded from training",
        },
        "summary": {
            "static": summaries["static"],
            "adaptive": summaries["adaptive"],
            "adaptive_minus_static_mean_actions": round(adaptive_mean - static_mean, 6),
            "relative_action_reduction": round((static_mean - adaptive_mean) / static_mean, 6) if static_mean else None,
            "completion_rate_difference": round(
                summaries["adaptive"]["completion_rate"] - summaries["static"]["completion_rate"], 6
            ),
            "holdout_completion_rate_difference": round(
                summaries["adaptive"]["holdout"]["completion_rate"]
                - summaries["static"]["holdout"]["completion_rate"], 6
            ),
        },
        "trials": [asdict(row) for row in results],
        "interpretation": {
            "p0_acceptance": "NOT_EVALUATED_BY_THIS_PILOT",
            "security_metrics": "NOT_MEASURED",
            "expected_diagnostic": (
                "The global selector may choose the shorter spare-key route from training, "
                "then fail the held-out unavailable-key context. Such a result indicates that "
                "strategy selection needs context/precondition awareness; it is not a benchmark failure."
            ),
            "holdout": True,
        },
        "p0_acceptance": {
            "30_controlled_tasks_x_3_runs": False,
            "correct_completion_threshold": 0.85,
            "verification_threshold": 1.0,
            "unauthorized_high_risk_actions_allowed": 0,
            "harmful_duplicate_side_effects_allowed": 0,
            "recoverable_failure_recovery_threshold": 0.80,
            "status": "NOT_EVALUATED_BY_THIS_PILOT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--training-repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run_benchmark(repeats=args.repeats, training_repeats=args.training_repeats)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nJSON report written to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
