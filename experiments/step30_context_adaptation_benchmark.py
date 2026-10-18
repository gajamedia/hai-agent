
"""Step 30 context-adaptation benchmark.

Compares:
1. static: fixed box route;
2. experience_only: highest learned score without evaluation context;
3. context_adaptive: feasibility filtering followed by learned-score selection.

Training and evaluation use real RoomEnvironment executions. Training statistics
are frozen before evaluation; evaluation outcomes do not update them.

Run from repository root:
    python experiments/step30_context_adaptation_benchmark.py
    python experiments/step30_context_adaptation_benchmark.py --repeats 3 --training-repeats 3
    python experiments/step30_context_adaptation_benchmark.py --json-out artifacts/step30_context_adaptation_benchmark.json

This is a controlled in-memory benchmark, not evidence of production-wide
generalization or safety.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.cognition.active_explorer import ActiveExplorer
from agent.cognition.unified_executor import UnifiedDecisionExecutor
from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector
from agent.world.room import RoomEnvironment


# Ensure the repository root is importable when this file is run directly.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
# Reuse the existing benchmark's deterministic decision and real-execution
# training procedure so this comparison remains aligned with Step 30.4.
from experiments.step30_meta_learning_benchmark import (
    BOX_ROUTE,
    GOAL_DESCRIPTION,
    GOAL_STATE,
    SPARE_ROUTE,
    _decision,
    _train_candidates,
)

MODES = ("static", "experience_only", "context_adaptive")
SCENARIOS = {
    "spare_key_available": True,
    "spare_key_unavailable": False,
}


@dataclass
class EvaluationResult:
    mode: str
    scenario: str
    trial: int
    completed: bool
    verified: bool
    initial_attempt_success: bool
    recovery_attempted: bool
    recovery_success: bool
    selected_strategy: str
    actions_attempted: int
    final_state: dict[str, Any]
    failure_types: list[str]


def _build_runtime(name: str, spare_key_available: bool):
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


def _verified(final_state: Mapping[str, Any]) -> bool:
    return all(
        final_state.get(key) == expected
        for key, expected in GOAL_STATE.items()
    )


def _route_for_mode(
    mode: str,
    *,
    box_strategy: Strategy,
    spare_strategy: Strategy,
    learned_strategy: Strategy,
    environment: RoomEnvironment,
    agent: Agent,
) -> tuple[str, tuple[str, ...]]:
    if mode == "static":
        return "fixed_box_route", BOX_ROUTE

    if mode == "experience_only":
        return learned_strategy.name, tuple(learned_strategy.steps)

    if mode == "context_adaptive":
        selector = StrategySelector()
        selected = selector.select(
            [box_strategy, spare_strategy],
            goal=GOAL_DESCRIPTION,
            world_state=environment.observe(),
            actions=agent.actions,
            goal_state=GOAL_STATE,
        )
        if selected is None:
            return "no_feasible_strategy", ()

        return selected.name, tuple(selected.steps)

    raise ValueError(f"Unsupported benchmark mode: {mode}")


def _evaluate(
    *,
    mode: str,
    scenario: str,
    trial: int,
    spare_key_available: bool,
    box_strategy: Strategy,
    spare_strategy: Strategy,
    learned_strategy: Strategy,
) -> EvaluationResult:
    environment, agent, executor, goal = _build_runtime(
        name=f"step30-context-{scenario}-{mode}-{trial}",
        spare_key_available=spare_key_available,
    )

    # Every mode receives the same initial action and recovery opportunity.
    initial_result = executor.execute_action(
        decision=_decision(),
        action=agent.actions["unlock_door"],
        agent=agent,
        environment=environment,
        goal=goal,
        experience_id=(
            f"step30-context-{scenario}-{mode}-{trial}-initial"
        ),
    )
    initial_success = bool(initial_result.success)

    failures: list[str] = []
    if not initial_success and isinstance(
        initial_result.direct_result, dict
    ):
        failure_type = initial_result.direct_result.get("failure_type")
        if failure_type:
            failures.append(str(failure_type))

    selected_name, route = _route_for_mode(
        mode,
        box_strategy=box_strategy,
        spare_strategy=spare_strategy,
        learned_strategy=learned_strategy,
        environment=environment,
        agent=agent,
    )

    recovery_attempted = not initial_success
    recovery_success = False
    actions_attempted = 1

    if recovery_attempted and route:
        plan = [agent.actions[name] for name in route]
        recovery_result = executor.execute_plan(
            decision=_decision(),
            agent=agent,
            environment=environment,
            goal=Goal(
                GOAL_DESCRIPTION,
                target_state=dict(GOAL_STATE),
            ),
            actions=plan,
            world_state=environment.observe(),
            experience_id=(
                f"step30-context-{scenario}-{mode}-{trial}-recovery"
            ),
        )
        recovery_success = bool(recovery_result.success)
        step_results = recovery_result.plan_execution_results or ()
        actions_attempted += len(step_results)

        for step in step_results:
            if not bool(getattr(step, "success", False)):
                result = getattr(step, "result", None)
                if isinstance(result, dict):
                    failure_type = result.get("failure_type")
                    if failure_type:
                        failures.append(str(failure_type))

    final_state = environment.observe()
    verified = _verified(final_state)

    return EvaluationResult(
        mode=mode,
        scenario=scenario,
        trial=trial,
        completed=verified,
        verified=verified,
        initial_attempt_success=initial_success,
        recovery_attempted=recovery_attempted,
        recovery_success=recovery_success,
        selected_strategy=selected_name,
        actions_attempted=actions_attempted,
        final_state=dict(final_state),
        failure_types=failures,
    )


def _summarize(rows: list[EvaluationResult]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {
            "runs": 0,
            "completion_rate": None,
            "verification_rate": None,
            "mean_actions_attempted": None,
        }

    recovery_rows = [row for row in rows if row.recovery_attempted]
    return {
        "runs": count,
        "completed": sum(row.completed for row in rows),
        "completion_rate": sum(row.completed for row in rows) / count,
        "verified": sum(row.verified for row in rows),
        "verification_rate": sum(row.verified for row in rows) / count,
        "recovery_attempts": len(recovery_rows),
        "recovery_successes": sum(
            row.recovery_success for row in recovery_rows
        ),
        "recovery_success_rate_when_attempted": (
            sum(row.recovery_success for row in recovery_rows)
            / len(recovery_rows)
            if recovery_rows
            else None
        ),
        "mean_actions_attempted": (
            sum(row.actions_attempted for row in rows) / count
        ),
        "runs_with_action_failures": sum(
            bool(row.failure_types) for row in rows
        ),
        # These safety metrics are not exercised by this benchmark.
        "unauthorized_high_risk_actions": None,
        "harmful_duplicate_side_effects": None,
    }


def run_benchmark(
    repeats: int = 3,
    training_repeats: int = 3,
) -> dict[str, Any]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    if (
        isinstance(training_repeats, bool)
        or not isinstance(training_repeats, int)
        or training_repeats < 1
    ):
        raise ValueError("training_repeats must be a positive integer.")

    results: list[EvaluationResult] = []
    training_reports: list[dict[str, Any]] = []

    for trial in range(1, repeats + 1):
        box_strategy, spare_strategy, learned_strategy, candidate_reports = (
            _train_candidates(trial, training_repeats)
        )

        # Capture immutable summaries before evaluation. Evaluation does not
        # record success/failure into these strategies.
        training_reports.append({
            "trial": trial,
            "candidate_strategies": candidate_reports,
            "experience_only_selected_strategy": learned_strategy.name,
            "frozen_before_evaluation": True,
        })

        for scenario, spare_key_available in SCENARIOS.items():
            for mode in MODES:
                results.append(
                    _evaluate(
                        mode=mode,
                        scenario=scenario,
                        trial=trial,
                        spare_key_available=spare_key_available,
                        box_strategy=box_strategy,
                        spare_strategy=spare_strategy,
                        learned_strategy=learned_strategy,
                    )
                )

    summaries: dict[str, Any] = {}
    for scenario in SCENARIOS:
        summaries[scenario] = {}
        for mode in MODES:
            rows = [
                row for row in results
                if row.scenario == scenario and row.mode == mode
            ]
            summaries[scenario][mode] = _summarize(rows)

    return {
        "benchmark": "step30_context_adaptation_benchmark_v1",
        "purpose": (
            "Separate fixed-route performance, experience-only strategy "
            "selection, and context-aware feasibility filtering."
        ),
        "evaluation_design": {
            "repeats_per_scenario_per_mode": repeats,
            "training_repeats_per_strategy_per_trial": training_repeats,
            "scenarios": {
                name: {"spare_key_available": available}
                for name, available in SCENARIOS.items()
            },
            "modes": list(MODES),
            "same_initial_action_for_all_modes": True,
            "same_recovery_budget_for_all_modes": True,
            "evaluation_updates_training": False,
            "training_statistics_frozen_before_evaluation": True,
            "training_evidence": (
                "Actual RoomEnvironment executions and final-state verification."
            ),
        },
        "evaluation_runs": len(results),
        "training": training_reports,
        "summary_by_scenario": summaries,
        "overall_summary": {
            mode: _summarize([
                row for row in results if row.mode == mode
            ])
            for mode in MODES
        },
        "trials": [asdict(row) for row in results],
        "scope_limits": [
            "Deterministic in-memory RoomEnvironment only.",
            "Not evidence of production-wide generalization.",
            "Security and harmful-duplicate metrics are not measured.",
            "Context-aware feasibility depends on declared action preconditions/effects.",
        ],
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
