"""Fair Step 30.4 pilot: fixed fallback vs experience-selected fallback.

Run from the repository root:
    python experiments/step30_meta_learning_benchmark.py
    python experiments/step30_meta_learning_benchmark.py --repeats 3 --json-out artifacts/step30_benchmark.json

Both evaluation arms receive the same initial failed action and the same one-plan
recovery budget. The baseline uses a fixed fallback route. The adaptive arm uses
StrategySelector on strategy statistics earned through real training executions.
This is a controlled integration pilot, not the full P0 30-task/90-run benchmark.
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
        reason="Controlled Step 30.4 matched benchmark decision.",
        uncertainty=0.1,
        expected_information_gain=0.0,
        risk=0.1,
    )


def _build_runtime(name: str, spare_key_available: bool = True):
    environment = RoomEnvironment(
        spare_key_available=spare_key_available
    )
    agent = Agent(name=name)

    actions = [
        Action(
            name="open_box",
            description="Open the box",
            handler=lambda **kwargs: environment.execute("open_box"),
            preconditions={"box": "closed"},
            effects={"box": "open"},
        ),
        Action(
            name="take_key",
            description="Take the key from the box",
            handler=lambda **kwargs: environment.execute("take_key"),
            preconditions={"box": "open", "key": "inside_box"},
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key",
            description="Find the spare key",
            handler=lambda **kwargs: environment.execute("find_spare_key"),
            preconditions={"spare_key": "available"},
            effects={"key": "agent", "spare_key": "found"},
        ),
        Action(
            name="unlock_door",
            description="Unlock the door",
            handler=lambda **kwargs: environment.execute("unlock_door"),
            preconditions={"key": "agent", "door": "locked"},
            effects={"door": "unlocked"},
        ),
    ]
    agent.register_actions(actions)
    executor = UnifiedDecisionExecutor(explorer=ActiveExplorer())
    goal = Goal(GOAL_DESCRIPTION, target_state=dict(GOAL_STATE))
    return environment, agent, executor, goal


def _execute_route(
    *,
    route: tuple[str, ...],
    experience_id: str,
    name: str,
    spare_key_available: bool = True,
):
    environment, agent, executor, goal = _build_runtime(
        name=name,
        spare_key_available=spare_key_available,
    )
    actions = [agent.actions[action_name] for action_name in route]
    result = executor.execute_plan(
        decision=_decision(),
        agent=agent,
        environment=environment,
        goal=goal,
        actions=actions,
        world_state=environment.observe(),
        experience_id=experience_id,
    )
    final_state = environment.observe()
    verified = all(
        final_state.get(key) == value
        for key, value in GOAL_STATE.items()
    )
    return result, verified, final_state, len(
        result.plan_execution_results or ()
    )


def _train_strategy(
    strategy: Strategy,
    route: tuple[str, ...],
    *,
    trial: int,
    training_repeats: int,
) -> dict[str, Any]:
    successes = 0
    failures = 0
    action_total = 0

    for episode in range(1, training_repeats + 1):
        result, verified, _state, attempted = _execute_route(
            route=route,
            experience_id=(
                f"step30-training-{trial}-{strategy.name}-{episode}"
            ),
            name=f"step30-training-{trial}-{strategy.name}-{episode}",
            spare_key_available=True,
        )
        action_total += attempted
        event_id = f"step30-training-{trial}-{strategy.name}-{episode}"

        if result.success and verified:
            strategy.record_success(
                experience_id=event_id,
                action_count=attempted,
                progress_delta=1.0,
            )
            successes += 1
        else:
            strategy.record_failure(
                experience_id=event_id,
                action_count=attempted,
                progress_delta=0.0,
            )
            failures += 1

    return {
        "strategy": strategy.name,
        "training_episodes": training_repeats,
        "successes": successes,
        "failures": failures,
        "empirical_success_rate": (
            successes / training_repeats if training_repeats else None
        ),
        "mean_actions": (
            action_total / training_repeats if training_repeats else None
        ),
        "evidence_source": "actual RoomEnvironment execution and final-state verification",
    }


def _train_candidates(trial: int, training_repeats: int):
    box_strategy = Strategy(
        name="open_box_then_take_key",
        goal=GOAL_DESCRIPTION,
        steps=list(BOX_ROUTE),
    )
    spare_strategy = Strategy(
        name="find_spare_key",
        goal=GOAL_DESCRIPTION,
        steps=list(SPARE_ROUTE),
    )

    box_report = _train_strategy(
        box_strategy,
        BOX_ROUTE,
        trial=trial,
        training_repeats=training_repeats,
    )
    spare_report = _train_strategy(
        spare_strategy,
        SPARE_ROUTE,
        trial=trial,
        training_repeats=training_repeats,
    )

    selector = StrategySelector()
    selected = selector.select(
        [box_strategy, spare_strategy],
        goal=GOAL_DESCRIPTION,
    )
    if selected is None:
        raise RuntimeError("StrategySelector returned no trained strategy.")

    return box_strategy, spare_strategy, selected, [box_report, spare_report]


def run_trial(
    mode: str,
    trial: int,
    *,
    selected_strategy: Strategy,
    training_summary: list[dict[str, Any]],
) -> TrialResult:
    if mode not in {"static", "adaptive"}:
        raise ValueError("mode must be 'static' or 'adaptive'.")

    environment, agent, executor, goal = _build_runtime(
        name=f"step30-evaluation-{mode}-{trial}",
        spare_key_available=True,
    )

    # Both conditions receive the exact same first action and environment.
    initial_result = executor.execute_action(
        decision=_decision(),
        action=agent.actions["unlock_door"],
        agent=agent,
        environment=environment,
        goal=goal,
        experience_id=f"step30-evaluation-{mode}-{trial}-initial",
    )
    initial_success = bool(initial_result.success)
    failures: list[str] = []
    if not initial_success and isinstance(initial_result.direct_result, dict):
        failure_type = initial_result.direct_result.get("failure_type")
        if failure_type:
            failures.append(str(failure_type))

    # The baseline and adaptive arm both receive one recovery plan. The only
    # difference is route choice: fixed baseline vs learned selector output.
    if mode == "static":
        final_strategy = "fixed_fallback_open_box_then_take_key"
        selected_route = BOX_ROUTE
    else:
        final_strategy = selected_strategy.name
        selected_route = tuple(selected_strategy.steps)

    recovery_attempted = not initial_success
    recovery_success = False
    actions_attempted = 1

    if recovery_attempted:
        recovery_goal = Goal(
            GOAL_DESCRIPTION,
            target_state=dict(GOAL_STATE),
        )
        plan = [agent.actions[name] for name in selected_route]
        recovery_result = executor.execute_plan(
            decision=_decision(),
            agent=agent,
            environment=environment,
            goal=recovery_goal,
            actions=plan,
            world_state=environment.observe(),
            experience_id=f"step30-evaluation-{mode}-{trial}-recovery",
        )
        recovery_success = bool(recovery_result.success)
        actions_attempted += len(
            recovery_result.plan_execution_results or ()
        )
        for step in recovery_result.plan_execution_results or ():
            if not bool(getattr(step, "success", False)):
                result = getattr(step, "result", None)
                if isinstance(result, dict) and result.get("failure_type"):
                    failures.append(str(result["failure_type"]))

    final_state = environment.observe()
    verified = all(
        final_state.get(key) == value
        for key, value in GOAL_STATE.items()
    )
    return TrialResult(
        mode=mode,
        trial=trial,
        completed=verified,
        verified=verified,
        initial_strategy="direct_unlock",
        final_strategy=final_strategy,
        initial_attempt_success=initial_success,
        recovery_attempted=recovery_attempted,
        recovery_success=recovery_success,
        actions_attempted=actions_attempted,
        final_state=final_state,
        failure_types=failures,
    )


def summarize(results: list[TrialResult]) -> dict[str, Any]:
    count = len(results)
    if count == 0:
        return {
            "runs": 0,
            "completion_rate": None,
            "verification_rate": None,
            "mean_actions_attempted": None,
        }

    recovery_trials = [row for row in results if row.recovery_attempted]
    return {
        "runs": count,
        "completed": sum(row.completed for row in results),
        "completion_rate": sum(row.completed for row in results) / count,
        "verified": sum(row.verified for row in results),
        "verification_rate": sum(row.verified for row in results) / count,
        "initial_attempt_success_rate": (
            sum(row.initial_attempt_success for row in results) / count
        ),
        "recovery_attempts": len(recovery_trials),
        "recovery_attempt_rate": len(recovery_trials) / count,
        "recovery_successes": sum(row.recovery_success for row in recovery_trials),
        "recovery_success_rate_when_attempted": (
            sum(row.recovery_success for row in recovery_trials) / len(recovery_trials)
            if recovery_trials else None
        ),
        "mean_actions_attempted": (
            sum(row.actions_attempted for row in results) / count
        ),
        "unauthorized_high_risk_actions": None,
        "harmful_duplicate_side_effects": None,
        "note": (
            "Security and harmful-duplicate metrics are not exercised by this "
            "RoomEnvironment scenario; null means not measured, not zero."
        ),
    }


def run_benchmark(
    repeats: int = 3,
    training_repeats: int = 3,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if training_repeats < 1:
        raise ValueError("training_repeats must be at least 1.")

    results: list[TrialResult] = []
    training_reports = []
    selected_routes = []

    for trial in range(1, repeats + 1):
        # Train candidates only on separate training episodes, then freeze
        # the resulting statistics for both matched evaluation conditions.
        box_strategy, spare_strategy, selected, candidate_reports = (
            _train_candidates(trial, training_repeats)
        )
        training_reports.append({
            "trial_pair": trial,
            "candidate_strategies": candidate_reports,
            "selector_scores": {
                box_strategy.name: round(
                    box_strategy.success_rate
                    - box_strategy.average_actions * 0.01
                    - box_strategy.reflection_penalty,
                    6,
                ),
                spare_strategy.name: round(
                    spare_strategy.success_rate
                    - spare_strategy.average_actions * 0.01
                    - spare_strategy.reflection_penalty,
                    6,
                ),
            },
            "selected_strategy": selected.name,
        })
        selected_routes.append(selected.name)

        results.append(run_trial(
            "static",
            trial,
            selected_strategy=selected,
            training_summary=candidate_reports,
        ))
        results.append(run_trial(
            "adaptive",
            trial,
            selected_strategy=selected,
            training_summary=candidate_reports,
        ))

    static_results = [row for row in results if row.mode == "static"]
    adaptive_results = [row for row in results if row.mode == "adaptive"]
    static_summary = summarize(static_results)
    adaptive_summary = summarize(adaptive_results)

    return {
        "benchmark": "step30_matched_recovery_room_task_v2",
        "purpose": (
            "controlled pilot comparing fixed fallback with experience-selected "
            "fallback under equal initial attempts and equal recovery budgets"
        ),
        "task_count": 1,
        "repeats_per_mode": repeats,
        "evaluation_runs": len(results),
        "training_repeats_per_strategy_per_pair": training_repeats,
        "training_episodes_total": repeats * training_repeats * 2,
        "task": {
            "goal": GOAL_DESCRIPTION,
            "target_state": GOAL_STATE,
            "initial_state": {
                "door": "locked",
                "box": "closed",
                "key": "inside_box",
                "spare_key": "available",
                "agent_location": "room",
            },
        },
        "conditions": {
            "static": (
                "one direct attempt, then one fixed open_box/take_key/unlock_door recovery plan"
            ),
            "adaptive": (
                "same direct attempt and recovery budget; route selected by StrategySelector "
                "using strategy outcomes from separate real training executions"
            ),
            "training_evidence": (
                "actual RoomEnvironment plan executions and final-state verification; "
                "no synthetic prior counters"
            ),
            "evaluation_updates_training": False,
            "holdout": False,
        },
        "training": training_reports,
        "summary": {
            "static": static_summary,
            "adaptive": adaptive_summary,
            "adaptive_minus_static_mean_actions": round(
                adaptive_summary["mean_actions_attempted"]
                - static_summary["mean_actions_attempted"],
                6,
            ),
            "relative_action_reduction": (
                round(
                    (
                        static_summary["mean_actions_attempted"]
                        - adaptive_summary["mean_actions_attempted"]
                    ) / static_summary["mean_actions_attempted"],
                    6,
                )
                if static_summary["mean_actions_attempted"] else None
            ),
            "completion_rate_difference": round(
                adaptive_summary["completion_rate"]
                - static_summary["completion_rate"],
                6,
            ),
        },
        "trials": [asdict(row) for row in results],
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

    report = run_benchmark(
        repeats=args.repeats,
        training_repeats=args.training_repeats,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nJSON report written to: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
