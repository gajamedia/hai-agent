"""Step 39.4: fail-closed preflight for application-owned P0 scenarios.

This module validates scenario construction only. It never executes actions and
never creates P0RunEvidence, so its output cannot be mistaken for benchmark proof.
Python 3.10+; standard-library only.
"""
from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_TASK_COUNT = 30
EXPECTED_RUNS_PER_TASK = 3


class P0ScenarioReadinessError(RuntimeError):
    """Raised when a provider cannot be safely preflighted."""


@dataclass(frozen=True)
class P0ScenarioReadinessIssue:
    code: str
    message: str
    task_id: str | None = None
    run_index: int | None = None


@dataclass
class P0ScenarioReadinessReport:
    provider: str
    expected_tasks: int = EXPECTED_TASK_COUNT
    expected_runs_per_task: int = EXPECTED_RUNS_PER_TASK
    tasks_seen: int = 0
    scenarios_built: int = 0
    unique_agents: int = 0
    unique_environments: int = 0
    valid: bool = False
    issues: list[P0ScenarioReadinessIssue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["issues"] = [asdict(issue) for issue in self.issues]
        return result


def load_factory(spec: str):
    if not isinstance(spec, str) or ":" not in spec:
        raise P0ScenarioReadinessError(
            "Provider must be specified as 'module.path:factory_name'."
        )
    module_name, factory_name = spec.split(":", 1)
    if not module_name.strip() or not factory_name.strip():
        raise P0ScenarioReadinessError(
            "Provider must be specified as 'module.path:factory_name'."
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as exc:
        raise P0ScenarioReadinessError(f"Cannot load provider factory {spec!r}: {exc}") from exc
    if not callable(factory):
        raise P0ScenarioReadinessError(f"Provider factory {spec!r} is not callable.")
    return factory


def _issue(report: P0ScenarioReadinessReport, code: str, message: str,
           task_id: str | None = None, run_index: int | None = None) -> None:
    report.issues.append(P0ScenarioReadinessIssue(code, message, task_id, run_index))


def _task_id(task: Any) -> str | None:
    value = getattr(task, "task_id", None)
    return value if isinstance(value, str) else None


def _validate_scenario(scenario: Any, task_id: str, run_index: int,
                       report: P0ScenarioReadinessReport) -> tuple[int | None, int | None]:
    if scenario is None:
        _issue(report, "scenario_missing", "build_scenario returned None.", task_id, run_index)
        return None, None

    agent = getattr(scenario, "agent", None)
    environment = getattr(scenario, "environment", None)
    goal = getattr(scenario, "goal", None)
    plan = getattr(scenario, "plan", None)
    if agent is None:
        _issue(report, "agent_missing", "Scenario has no Agent instance.", task_id, run_index)
    if environment is None or not callable(getattr(environment, "observe", None)):
        _issue(report, "environment_invalid", "Environment must expose observe().", task_id, run_index)
    if goal is None or not callable(getattr(goal, "is_satisfied", None)):
        _issue(report, "goal_verifier_missing", "Goal must expose deterministic is_satisfied(state).", task_id, run_index)
    target = getattr(goal, "target_state", None) if goal is not None else None
    if not isinstance(target, Mapping) or not target:
        _issue(report, "goal_target_missing", "Goal.target_state must be a non-empty mapping.", task_id, run_index)

    ledger = getattr(scenario, "duplicate_harmful_side_effect_count", None)
    if not callable(ledger):
        _issue(report, "side_effect_ledger_missing",
               "duplicate_harmful_side_effect_count() must read an application-owned ledger.", task_id, run_index)
    else:
        try:
            count = ledger()
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                _issue(report, "side_effect_ledger_invalid",
                       "Side-effect ledger callback must return a non-negative integer.", task_id, run_index)
        except Exception as exc:  # noqa: BLE001 - preflight must report callback failures
            _issue(report, "side_effect_ledger_error",
                   f"Side-effect ledger callback failed: {type(exc).__name__}: {exc}", task_id, run_index)

    action_map = getattr(agent, "actions", None) if agent is not None else None
    if not isinstance(action_map, Mapping):
        _issue(report, "action_registry_invalid", "Agent.actions must be a mapping of registered actions.", task_id, run_index)
        action_map = {}

    risks = getattr(scenario, "action_risks", None)
    if not isinstance(risks, Mapping):
        _issue(report, "risk_map_invalid", "action_risks must be an application-owned mapping.", task_id, run_index)
        risks = {}

    steps = getattr(plan, "steps", None) if plan is not None else None
    if not isinstance(steps, (tuple, list)) or not steps:
        _issue(report, "plan_steps_missing", "Plan must contain at least one step.", task_id, run_index)
        steps = []
    planned_actions: set[str] = set()
    for step in steps:
        name = getattr(step, "action", None)
        if not isinstance(name, str) or not name:
            _issue(report, "plan_action_invalid", "Every plan step must name an action.", task_id, run_index)
            continue
        planned_actions.add(name)
        if name not in action_map:
            _issue(report, "action_unregistered", f"Plan action {name!r} is not registered on Agent.actions.", task_id, run_index)
        if name not in risks:
            _issue(report, "risk_not_configured", f"No explicit risk score configured for planned action {name!r}.", task_id, run_index)
        else:
            value = risks[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                _issue(report, "risk_score_invalid", f"Risk score for {name!r} must be finite and in [0, 1].", task_id, run_index)

    required = getattr(scenario, "approval_required_actions", ()) or ()
    try:
        required_set = set(required)
    except TypeError:
        required_set = set()
        _issue(report, "approval_actions_invalid", "approval_required_actions must be an iterable of action names.", task_id, run_index)
    unknown_required = required_set - set(action_map)
    for name in sorted(unknown_required):
        _issue(report, "approval_action_unregistered", f"Approval-required action {name!r} is not registered.", task_id, run_index)
    checker = getattr(scenario, "approval_checker", None)
    if required_set and not callable(checker):
        _issue(report, "approval_checker_missing", "Approval-required actions need a trusted approval_checker.", task_id, run_index)

    risk_threshold = getattr(scenario, "risk_threshold", None)
    if isinstance(risk_threshold, bool) or not isinstance(risk_threshold, (int, float)) or not math.isfinite(float(risk_threshold)) or not 0 <= float(risk_threshold) <= 1:
        _issue(report, "risk_threshold_invalid", "risk_threshold must be finite and in [0, 1].", task_id, run_index)

    return (id(agent) if agent is not None else None,
            id(environment) if environment is not None else None)


def preflight_provider(provider: Any, *, provider_spec: str = "<in-memory-provider>",
                       expected_tasks: int = EXPECTED_TASK_COUNT,
                       runs_per_task: int = EXPECTED_RUNS_PER_TASK) -> P0ScenarioReadinessReport:
    """Build and validate all task/run scenarios without executing their plans."""
    report = P0ScenarioReadinessReport(
        provider=provider_spec,
        expected_tasks=expected_tasks,
        expected_runs_per_task=runs_per_task,
    )
    if not callable(getattr(provider, "tasks", None)) or not callable(getattr(provider, "build_scenario", None)):
        _issue(report, "provider_contract_invalid", "Provider must expose tasks() and build_scenario(task, run_index).")
        return report

    try:
        tasks = list(provider.tasks())
    except Exception as exc:  # noqa: BLE001
        _issue(report, "task_catalog_error", f"tasks() failed: {type(exc).__name__}: {exc}")
        return report

    report.tasks_seen = len(tasks)
    expected_ids = [f"task-{i:02d}" for i in range(1, expected_tasks + 1)]
    ids = [_task_id(task) for task in tasks]
    if len(tasks) != expected_tasks:
        _issue(report, "task_count_mismatch", f"Expected {expected_tasks} tasks; provider returned {len(tasks)}.")
    if ids != expected_ids:
        _issue(report, "task_order_or_id_mismatch",
               "Task IDs must be exactly task-01..task-N in canonical order; observed: " + repr(ids))

    agent_ids: set[int] = set()
    environment_ids: set[int] = set()
    # Hold strong references so CPython cannot recycle object IDs between runs.
    agent_refs: list[Any] = []
    environment_refs: list[Any] = []
    for task in tasks:
        tid = _task_id(task) or "<missing-task-id>"
        for run_index in range(1, runs_per_task + 1):
            try:
                scenario = provider.build_scenario(task, run_index)
            except Exception as exc:  # noqa: BLE001
                _issue(report, "scenario_build_error", f"build_scenario failed: {type(exc).__name__}: {exc}", tid, run_index)
                continue
            report.scenarios_built += 1
            agent_id, environment_id = _validate_scenario(scenario, tid, run_index, report)
            agent = getattr(scenario, "agent", None)
            environment = getattr(scenario, "environment", None)
            if agent is not None:
                agent_refs.append(agent)
            if environment is not None:
                environment_refs.append(environment)
            if agent_id is not None:
                if agent_id in agent_ids:
                    _issue(report, "agent_reused", "Provider reused an Agent across benchmark runs.", tid, run_index)
                agent_ids.add(agent_id)
            if environment_id is not None:
                if environment_id in environment_ids:
                    _issue(report, "environment_reused", "Provider reused an environment across benchmark runs.", tid, run_index)
                environment_ids.add(environment_id)

    report.unique_agents = len(agent_ids)
    report.unique_environments = len(environment_ids)
    expected_scenarios = expected_tasks * runs_per_task
    if report.scenarios_built != expected_scenarios:
        _issue(report, "scenario_matrix_incomplete",
               f"Expected to build {expected_scenarios} scenarios; built {report.scenarios_built}.")
    report.valid = not report.issues and report.scenarios_built == expected_scenarios
    if report.valid:
        report.notes.append("Preflight passed. No plan was executed and no benchmark evidence was produced.")
    else:
        report.notes.append("Preflight failed closed. Fix all issues before running the benchmark.")
    return report


def run_provider_preflight(provider_spec: str, *, expected_tasks: int = EXPECTED_TASK_COUNT,
                           runs_per_task: int = EXPECTED_RUNS_PER_TASK) -> P0ScenarioReadinessReport:
    factory = load_factory(provider_spec)
    try:
        provider = factory()
    except Exception as exc:  # noqa: BLE001
        raise P0ScenarioReadinessError(
            f"Provider factory {provider_spec!r} failed: {type(exc).__name__}: {exc}"
        ) from exc
    return preflight_provider(provider, provider_spec=provider_spec,
                              expected_tasks=expected_tasks, runs_per_task=runs_per_task)


def write_report(report: P0ScenarioReadinessReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "P0ScenarioReadinessError",
    "P0ScenarioReadinessIssue",
    "P0ScenarioReadinessReport",
    "preflight_provider",
    "run_provider_preflight",
    "write_report",
]
