"""Step 41: adapter-driven, multi-domain evaluation contract.

This module deliberately does not provide a fake agent or pretend that a
reference simulator is evidence of agent generalization. Applications inject a
runner that executes a scenario through the runtime they want to evaluate.
Python 3.10+, standard library only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol


class MultiScenarioEvaluationError(ValueError):
    """Raised when scenario definitions or runner outputs are invalid."""


@dataclass(frozen=True)
class ScenarioCase:
    scenario_id: str
    domain: str
    description: str
    initial_state: Mapping[str, Any]
    target_state: Mapping[str, Any]
    allowed_actions: tuple[str, ...]
    holdout: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise MultiScenarioEvaluationError("scenario_id must not be empty")
        if not self.domain.strip():
            raise MultiScenarioEvaluationError("domain must not be empty")
        if not self.description.strip():
            raise MultiScenarioEvaluationError("description must not be empty")
        if not isinstance(self.initial_state, Mapping) or not isinstance(self.target_state, Mapping):
            raise MultiScenarioEvaluationError("initial_state and target_state must be mappings")
        if not self.target_state:
            raise MultiScenarioEvaluationError("target_state must not be empty")
        if not self.allowed_actions or any(not isinstance(a, str) or not a.strip() for a in self.allowed_actions):
            raise MultiScenarioEvaluationError("allowed_actions must contain non-empty action names")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise MultiScenarioEvaluationError("allowed_actions must not contain duplicates")
        object.__setattr__(self, "initial_state", dict(self.initial_state))
        object.__setattr__(self, "target_state", dict(self.target_state))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class ScenarioExecution:
    """Facts returned by the runtime adapter, not inferred from agent prose."""
    final_state: Mapping[str, Any]
    runtime_success: bool
    actions_attempted: int
    executed_actions: tuple[str, ...] = ()
    failure_reason: str | None = None
    adapter_name: str = "unspecified"
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.final_state, Mapping):
            raise MultiScenarioEvaluationError("runner final_state must be a mapping")
        if not isinstance(self.runtime_success, bool):
            raise MultiScenarioEvaluationError("runtime_success must be bool")
        if isinstance(self.actions_attempted, bool) or not isinstance(self.actions_attempted, int) or self.actions_attempted < 0:
            raise MultiScenarioEvaluationError("actions_attempted must be a non-negative integer")
        if not self.adapter_name.strip():
            raise MultiScenarioEvaluationError("adapter_name must not be empty")
        object.__setattr__(self, "final_state", dict(self.final_state))
        object.__setattr__(self, "evidence", dict(self.evidence or {}))


class ScenarioRunner(Protocol):
    def __call__(self, scenario: ScenarioCase, run_index: int) -> ScenarioExecution: ...


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    domain: str
    holdout: bool
    run_index: int
    adapter_name: str
    runtime_success: bool
    goal_verified: bool
    correct_completion: bool
    actions_attempted: int
    executed_actions: tuple[str, ...]
    failure_reason: str | None
    final_state: Mapping[str, Any]
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiScenarioReport:
    schema_version: str
    evaluation_mode: str
    scenario_count: int
    runs_per_scenario: int
    total_runs: int
    aggregate: Mapping[str, Any]
    by_domain: Mapping[str, Mapping[str, Any]]
    holdout: Mapping[str, Any]
    results: tuple[ScenarioResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_mode": self.evaluation_mode,
            "scenario_count": self.scenario_count,
            "runs_per_scenario": self.runs_per_scenario,
            "total_runs": self.total_runs,
            "aggregate": dict(self.aggregate),
            "by_domain": {k: dict(v) for k, v in self.by_domain.items()},
            "holdout": dict(self.holdout),
            "results": [asdict(row) for row in self.results],
        }


def _metrics(rows: list[ScenarioResult]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"runs": 0, "runtime_success_rate": None, "goal_verified_rate": None,
                "correct_completion_rate": None, "mean_actions_attempted": None}
    return {
        "runs": count,
        "runtime_success_rate": round(sum(r.runtime_success for r in rows) / count, 6),
        "goal_verified_rate": round(sum(r.goal_verified for r in rows) / count, 6),
        "correct_completion_rate": round(sum(r.correct_completion for r in rows) / count, 6),
        "mean_actions_attempted": round(sum(r.actions_attempted for r in rows) / count, 6),
    }


def evaluate_scenarios(
    scenarios: Iterable[ScenarioCase],
    runner: ScenarioRunner,
    *,
    runs_per_scenario: int = 1,
    evaluation_mode: str = "real_runtime",
) -> MultiScenarioReport:
    """Run an injected adapter and verify goal state independently of its claim.

    `evaluation_mode` must identify the evidence source. Use `real_runtime` only
    when runner actually invokes the runtime under evaluation; use `fixture` for
    deterministic/reference simulations. This function never executes actions
    itself and never treats `runtime_success=True` alone as goal completion.
    """
    if not callable(runner):
        raise MultiScenarioEvaluationError("runner must be callable")
    if isinstance(runs_per_scenario, bool) or not isinstance(runs_per_scenario, int) or runs_per_scenario < 1:
        raise MultiScenarioEvaluationError("runs_per_scenario must be an integer >= 1")
    if evaluation_mode not in {"real_runtime", "fixture", "offline_replay"}:
        raise MultiScenarioEvaluationError("evaluation_mode must be real_runtime, fixture, or offline_replay")

    cases = tuple(scenarios)
    if not cases:
        raise MultiScenarioEvaluationError("at least one scenario is required")
    ids = [case.scenario_id for case in cases]
    if len(ids) != len(set(ids)):
        raise MultiScenarioEvaluationError("scenario_id values must be unique")
    if any(not isinstance(case, ScenarioCase) for case in cases):
        raise MultiScenarioEvaluationError("all scenarios must be ScenarioCase instances")

    results: list[ScenarioResult] = []
    for case in cases:
        for run_index in range(1, runs_per_scenario + 1):
            execution = runner(case, run_index)
            if not isinstance(execution, ScenarioExecution):
                raise MultiScenarioEvaluationError(
                    f"runner must return ScenarioExecution for {case.scenario_id} run {run_index}"
                )
            # Only target keys declared by the scenario are used for deterministic verification.
            verified = all(execution.final_state.get(key) == value for key, value in case.target_state.items())
            results.append(ScenarioResult(
                scenario_id=case.scenario_id,
                domain=case.domain,
                holdout=case.holdout,
                run_index=run_index,
                adapter_name=execution.adapter_name,
                runtime_success=execution.runtime_success,
                goal_verified=verified,
                correct_completion=bool(execution.runtime_success and verified),
                actions_attempted=execution.actions_attempted,
                executed_actions=tuple(execution.executed_actions),
                failure_reason=execution.failure_reason,
                final_state=dict(execution.final_state),
                evidence=dict(execution.evidence),
            ))

    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for row in results:
        grouped[row.domain].append(row)
    holdout_rows = [row for row in results if row.holdout]
    return MultiScenarioReport(
        schema_version="step41.multi-scenario.v1",
        evaluation_mode=evaluation_mode,
        scenario_count=len(cases),
        runs_per_scenario=runs_per_scenario,
        total_runs=len(results),
        aggregate=_metrics(results),
        by_domain={domain: _metrics(rows) for domain, rows in sorted(grouped.items())},
        holdout={"scenario_count": len({row.scenario_id for row in holdout_rows}), **_metrics(holdout_rows)},
        results=tuple(results),
    )


__all__ = [
    "MultiScenarioEvaluationError", "ScenarioCase", "ScenarioExecution",
    "ScenarioResult", "MultiScenarioReport", "ScenarioRunner", "evaluate_scenarios",
]
