"""Step 39.3 adapter that drives the canonical Agent execution path.

The adapter owns the integration path and evidence collection. Application-specific
scenarios remain injected because only the application knows how to construct its
real environment, registered actions, goals, and approval source.
"""
from __future__ import annotations

import importlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from agent.llm.agent_canonical_single_action_api import AgentCanonicalSingleActionAPI
from agent.llm.execution_security import (
    ExecutionSecurityPolicy,
    SQLiteExecutionSecurityAuditSink,
)
from agent.llm.execution_trace_runtime_observer import ExecutionTraceRuntimeObserver
from agent.llm.execution_trace_store import ExecutionTraceStore
from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.p0_acceptance_gate import P0RunEvidence
from agent.llm.p0_runtime_benchmark import P0BenchmarkAdapter, P0BenchmarkError, P0BenchmarkTask


@dataclass(frozen=True)
class P0RuntimeScenario:
    """One real, isolated runtime scenario supplied by the application.

    `goal` must expose a deterministic `is_satisfied(final_state)` method and a
    non-empty `target_state`. `duplicate_harmful_side_effect_count` must read an
    application-owned idempotency/side-effect ledger, not infer from agent text.
    Risk configuration and approval checker must be application-owned/trusted.
    """

    agent: Any
    environment: Any
    plan: Any
    goal: Any
    decision_factory: Callable[[Any], Any]
    action_risks: Mapping[str, float]
    approval_required_actions: frozenset[str] | set[str] | tuple[str, ...] = ()
    approval_checker: Callable[[Any], bool] | None = None
    risk_threshold: float = 0.65
    duplicate_harmful_side_effect_count: Callable[[], int] | None = None
    is_recoverable_failure: Callable[[Any, Any], bool] | None = None
    recovery_plan_factory: Callable[[Any, Any, Any], Any] | None = None
    # Optional application-defined extra metadata. Never used for authorization.
    metadata: Mapping[str, Any] | None = None


class P0RuntimeScenarioProvider(Protocol):
    """Application adapter contract for a controlled 30-task matrix."""

    def tasks(self) -> Iterable[P0BenchmarkTask]: ...
    def build_scenario(
        self, task: P0BenchmarkTask, run_index: int
    ) -> P0RuntimeScenario: ...


class P0CanonicalRuntimeAdapterError(P0BenchmarkError):
    """Raised when a real runtime scenario cannot provide trustworthy evidence."""


def _import_factory(spec: str) -> Callable[[], Any]:
    if not isinstance(spec, str) or ":" not in spec:
        raise P0CanonicalRuntimeAdapterError(
            "HAI_P0_SCENARIO_PROVIDER must be 'module.path:factory_name'."
        )
    module_name, factory_name = spec.split(":", 1)
    if not module_name or not factory_name:
        raise P0CanonicalRuntimeAdapterError(
            "HAI_P0_SCENARIO_PROVIDER must be 'module.path:factory_name'."
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as exc:
        raise P0CanonicalRuntimeAdapterError(
            f"Cannot load scenario provider '{spec}': {exc}"
        ) from exc
    if not callable(factory):
        raise P0CanonicalRuntimeAdapterError("Scenario provider factory is not callable.")
    return factory


def create_adapter() -> "P0CanonicalRuntimeAdapter":
    """Factory for p0_runtime_benchmark_cli --adapter.

    Configure HAI_P0_SCENARIO_PROVIDER=your_package.module:build_provider.
    The provider must return an object with tasks() and build_scenario(task, run_index).
    """
    spec = os.environ.get("HAI_P0_SCENARIO_PROVIDER", "").strip()
    if not spec:
        raise P0CanonicalRuntimeAdapterError(
            "Set HAI_P0_SCENARIO_PROVIDER=your_package.module:build_provider first."
        )
    provider = _import_factory(spec)()
    return P0CanonicalRuntimeAdapter(provider)


class P0CanonicalRuntimeAdapter(P0BenchmarkAdapter):
    """Execute each scenario through the repository's canonical runtime chain."""

    def __init__(
        self,
        provider: P0RuntimeScenarioProvider,
        *,
        artifact_dir: str | Path = "artifacts/p0_runtime",
    ) -> None:
        if provider is None:
            raise TypeError("provider cannot be None")
        if not callable(getattr(provider, "tasks", None)):
            raise TypeError("provider must expose tasks()")
        if not callable(getattr(provider, "build_scenario", None)):
            raise TypeError("provider must expose build_scenario(task, run_index)")
        self.provider = provider
        self.artifact_dir = Path(artifact_dir)
        # Keep strong references so Python cannot recycle an object ID between runs.
        self._seen_agents: list[Any] = []
        self._seen_environments: list[Any] = []

    def tasks(self) -> Iterable[P0BenchmarkTask]:
        return self.provider.tasks()

    @staticmethod
    def _validate_scenario(scenario: P0RuntimeScenario) -> None:
        if not isinstance(scenario, P0RuntimeScenario):
            raise P0CanonicalRuntimeAdapterError(
                "build_scenario must return P0RuntimeScenario."
            )
        if scenario.agent is None or scenario.environment is None:
            raise P0CanonicalRuntimeAdapterError("Scenario must provide a real Agent and environment.")
        if not callable(scenario.decision_factory):
            raise P0CanonicalRuntimeAdapterError("Scenario decision_factory must be callable.")
        if not isinstance(scenario.action_risks, Mapping):
            raise P0CanonicalRuntimeAdapterError("Scenario action_risks must be an application-owned mapping.")
        if not callable(getattr(scenario.environment, "observe", None)):
            raise P0CanonicalRuntimeAdapterError("Environment must expose observe().")
        if not callable(getattr(scenario.goal, "is_satisfied", None)):
            raise P0CanonicalRuntimeAdapterError(
                "Goal must expose deterministic is_satisfied(final_state)."
            )
        target_state = getattr(scenario.goal, "target_state", None)
        if not isinstance(target_state, Mapping) or not target_state:
            raise P0CanonicalRuntimeAdapterError(
                "Goal.target_state must be a non-empty mapping for deterministic verification."
            )
        if not callable(scenario.duplicate_harmful_side_effect_count):
            raise P0CanonicalRuntimeAdapterError(
                "Provide duplicate_harmful_side_effect_count() backed by the application's side-effect ledger."
            )
        if scenario.is_recoverable_failure is not None and not callable(scenario.is_recoverable_failure):
            raise P0CanonicalRuntimeAdapterError("is_recoverable_failure must be callable.")
        if scenario.recovery_plan_factory is not None and not callable(scenario.recovery_plan_factory):
            raise P0CanonicalRuntimeAdapterError("recovery_plan_factory must be callable.")

    def _build_runtime(self, scenario: P0RuntimeScenario, trace_path: Path, audit_path: Path):
        audit_sink = SQLiteExecutionSecurityAuditSink(audit_path)
        policy = ExecutionSecurityPolicy(
            action_risks=dict(scenario.action_risks),
            risk_threshold=scenario.risk_threshold,
            approval_required_actions=frozenset(scenario.approval_required_actions),
            approval_checker=scenario.approval_checker,
            audit_sink=audit_sink,
        )
        # Keep the same trusted policy visible to the Agent and the canonical API.
        scenario.agent.execution_security_policy = policy
        api = AgentCanonicalSingleActionAPI(
            scenario.agent,
            decision_factory=scenario.decision_factory,
            security_policy=policy,
        )
        bridge = LLMPlanSingleActionBridge.for_api(api)
        trace_store = ExecutionTraceStore(trace_path)
        trace_observer = ExecutionTraceRuntimeObserver(
            trace_store,
            raise_on_error=True,
            include_world_state=False,
        )
        runtime = LLMPlanRuntimeExecutor.for_bridge(
            bridge,
            runtime_event_observer=trace_observer,
        )
        # Verify the chain identity; do not silently substitute another executor.
        if runtime.bridge is not bridge or bridge.api is not api:
            raise P0CanonicalRuntimeAdapterError("Canonical runtime chain binding failed.")
        if api.executor.binding.agent is not scenario.agent:
            raise P0CanonicalRuntimeAdapterError("Canonical binding is not using the scenario Agent.")
        return runtime, api, policy, audit_sink, trace_store, trace_observer

    @staticmethod
    def _audit_evidence(policy: ExecutionSecurityPolicy, audit_sink: SQLiteExecutionSecurityAuditSink):
        """Validate durable audit records and authorization-to-outcome linkage."""
        durable = audit_sink.read_all()
        in_memory = policy.audit_log
        complete = bool(durable) and len(durable) == len(in_memory)
        approvals_complete = complete
        allowed_authorizations: dict[tuple[str, int], int] = {}
        for event in durable:
            key = (event.action, event.position)
            if event.event_type == "authorization":
                # Every attempted authorization must have an explicit decision.
                if not isinstance(event.allowed, bool) or not event.reason:
                    complete = False
                    approvals_complete = False
                if event.approval_required and not event.reason:
                    approvals_complete = False
                if event.allowed:
                    allowed_authorizations[key] = allowed_authorizations.get(key, 0) + 1
            elif event.event_type == "execution_outcome":
                if allowed_authorizations.get(key, 0) <= 0:
                    complete = False
                    approvals_complete = False
                else:
                    allowed_authorizations[key] -= 1
                if event.execution_success is None:
                    complete = False
        return complete, approvals_complete, durable

    def execute(self, task: P0BenchmarkTask, run_index: int) -> P0RunEvidence:
        attempt_id = uuid.uuid4().hex[:12]
        execution_id = f"{task.task_id}-run-{run_index:02d}-{attempt_id}"
        # Keep repeated benchmark invocations isolated; never append new audit
        # rows to a prior attempt's database or overwrite prior evidence.
        run_dir = self.artifact_dir / task.task_id / f"run-{run_index:02d}" / f"attempt-{attempt_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = run_dir / "execution_trace.jsonl"
        audit_path = run_dir / "security_audit.sqlite3"

        try:
            scenario = self.provider.build_scenario(task, run_index)
        except Exception as exc:
            raise P0CanonicalRuntimeAdapterError(
                f"Scenario construction failed for {task.task_id} run {run_index}: {exc}"
            ) from exc
        self._validate_scenario(scenario)

        # Require fresh objects per run to prevent state/trace contamination.
        if any(previous is scenario.agent for previous in self._seen_agents):
            raise P0CanonicalRuntimeAdapterError("Provider reused an Agent across benchmark runs.")
        if any(previous is scenario.environment for previous in self._seen_environments):
            raise P0CanonicalRuntimeAdapterError("Provider reused an environment across benchmark runs.")
        self._seen_agents.append(scenario.agent)
        self._seen_environments.append(scenario.environment)

        before_duplicates = int(scenario.duplicate_harmful_side_effect_count())
        if before_duplicates < 0:
            raise P0CanonicalRuntimeAdapterError("Side-effect ledger returned a negative count.")

        runtime, api, policy, audit_sink, trace_store, trace_observer = self._build_runtime(
            scenario, trace_path, audit_path
        )
        metadata = {
            "execution_id": execution_id,
            "generation": 1,
            "task_id": task.task_id,
            "run_index": run_index,
            "runtime_adapter": "p0_canonical_runtime_adapter",
        }
        if scenario.metadata:
            # Metadata is for trace context only; never used for authorization.
            metadata["scenario_metadata"] = dict(scenario.metadata)

        initial_state = scenario.environment.observe()
        report = runtime.execute(
            scenario.plan,
            goal=scenario.goal,
            world_state=initial_state,
            environment=scenario.environment,
            metadata=metadata,
        )
        final_state = scenario.environment.observe()
        deterministic_verification_available = (
            callable(getattr(scenario.goal, "is_satisfied", None))
            and bool(getattr(scenario.goal, "target_state", None))
        )
        goal_satisfied = (
            bool(scenario.goal.is_satisfied(final_state))
            if deterministic_verification_available
            else False
        )
        recoverable_failure = False
        recovery_succeeded = False

        if not (bool(report.success) and goal_satisfied) and scenario.is_recoverable_failure:
            recoverable_failure = bool(scenario.is_recoverable_failure(report, final_state))
            if recoverable_failure:
                if scenario.recovery_plan_factory is None:
                    raise P0CanonicalRuntimeAdapterError(
                        "Recoverable failure was identified but no explicit recovery_plan_factory was provided."
                    )
                recovery_plan = scenario.recovery_plan_factory(scenario, report, final_state)
                if recovery_plan is None:
                    raise P0CanonicalRuntimeAdapterError("recovery_plan_factory returned None.")
                recovery_report = runtime.execute(
                    recovery_plan,
                    goal=scenario.goal,
                    world_state=scenario.environment.observe(),
                    environment=scenario.environment,
                    metadata={
                        **metadata,
                        "generation": 2,
                        "recovery_of": execution_id,
                    },
                )
                report = recovery_report
                final_state = scenario.environment.observe()
                goal_satisfied = bool(scenario.goal.is_satisfied(final_state))
                recovery_succeeded = bool(recovery_report.success and goal_satisfied)

        after_duplicates = int(scenario.duplicate_harmful_side_effect_count())
        if after_duplicates < before_duplicates:
            raise P0CanonicalRuntimeAdapterError("Side-effect ledger count decreased during execution.")
        duplicate_side_effects = after_duplicates - before_duplicates

        audit_complete, approval_complete, durable_audit = self._audit_evidence(policy, audit_sink)
        trace_verification = trace_store.verify()
        trace_rows = trace_store.replay(execution_id)
        trace_complete = (
            isinstance(trace_verification, Mapping)
            and trace_verification.get("valid") is True
            and bool(trace_rows)
            and trace_observer.error_count == 0
            and trace_observer.persisted_count >= len(trace_rows)
        )

        # An outcome row must be preceded by a durable allowed authorization for
        # the same action/position. A denied request that never executes is not
        # counted as an unauthorized *executed* high-risk action.
        outstanding_allowed: dict[tuple[str, int], int] = {}
        unauthorized_executed = 0
        for event in durable_audit:
            key = (event.action, event.position)
            if event.event_type == "authorization" and event.allowed:
                outstanding_allowed[key] = outstanding_allowed.get(key, 0) + 1
            elif event.event_type == "execution_outcome":
                if outstanding_allowed.get(key, 0) <= 0:
                    unauthorized_executed += 1
                else:
                    outstanding_allowed[key] -= 1

        return P0RunEvidence(
            task_id=task.task_id,
            run_index=run_index,
            correct_completion=bool(report.success and goal_satisfied),
            deterministic_verification_passed=bool(deterministic_verification_available),
            unauthorized_high_risk_actions=unauthorized_executed,
            duplicate_harmful_side_effects=duplicate_side_effects,
            recoverable_failure=recoverable_failure,
            recovery_succeeded=recovery_succeeded,
            audit_evidence_complete=audit_complete,
            required_approval_evidence_complete=approval_complete,
            trace_integrity_passed=trace_complete,
            notes=(
                f"execution_id={execution_id}; goal_satisfied={goal_satisfied}; "
                f"runtime_success={bool(report.success)}; audit_events={len(durable_audit)}; "
                f"trace_events={len(trace_rows)}; trace_valid={trace_verification.get('valid') if isinstance(trace_verification, Mapping) else False}; "
                f"recovery={recovery_succeeded if recoverable_failure else 'not_required'}"
            ),
        )


__all__ = [
    "P0CanonicalRuntimeAdapter",
    "P0CanonicalRuntimeAdapterError",
    "P0RuntimeScenario",
    "P0RuntimeScenarioProvider",
    "create_adapter",
]
