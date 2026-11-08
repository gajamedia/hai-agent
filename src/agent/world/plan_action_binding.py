"""Bind LLM plan steps to capabilities discovered for one environment.

This module performs validation and produces immutable binding metadata only.
It never executes an action, exposes handlers, registers capabilities, or
changes the plan, Agent, Planner, or canonical execution boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Mapping

from agent.llm.planning import LLMPlan
from agent.world.environment_capability_discovery import (
    EnvironmentCapabilityDiscoveryReport,
    EnvironmentCapabilityState,
)


class PlanActionBindingError(RuntimeError):
    """Base error for plan-to-capability binding."""


class PlanActionBindingValidationError(PlanActionBindingError):
    """Raised when a plan cannot be safely bound to an environment."""


class PlanActionBindingState(str, Enum):
    BOUND = "bound"
    UNBOUND = "unbound"


@dataclass(frozen=True, slots=True)
class BoundPlanStep:
    step_index: int
    action_name: str
    capability_id: str
    environment_instance_id: str
    registry_version: int
    risk_level: str
    permissions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanActionBindingReport:
    plan_goal: str
    environment_instance_id: str
    environment_type: str
    registry_version: int
    steps: tuple[BoundPlanStep, ...]
    unbound_steps: tuple[tuple[int, str, str], ...]
    state: PlanActionBindingState

    @property
    def is_complete(self) -> bool:
        return self.state == PlanActionBindingState.BOUND and not self.unbound_steps

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(step.capability_id for step in self.steps)


class PlanActionBinder:
    """Resolve plan action names against one discovery snapshot.

    A plan step's action string is treated as an untrusted requested action.
    It is bound only if exactly one AVAILABLE discovery candidate matches it.
    Missing, inactive, ambiguous, or unlisted actions remain unbound.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self.last_report: PlanActionBindingReport | None = None

    @staticmethod
    def _validate_inputs(
        plan: LLMPlan,
        discovery: EnvironmentCapabilityDiscoveryReport,
    ) -> None:
        if not isinstance(plan, LLMPlan):
            raise PlanActionBindingValidationError("plan must be an LLMPlan.")
        if not isinstance(discovery, EnvironmentCapabilityDiscoveryReport):
            raise PlanActionBindingValidationError(
                "discovery must be an EnvironmentCapabilityDiscoveryReport."
            )
        if not isinstance(plan.goal, str) or not plan.goal.strip():
            raise PlanActionBindingValidationError(
                "Plan goal must be a non-empty string."
            )
        if not plan.steps:
            raise PlanActionBindingValidationError("Plan must contain at least one step.")

        indices = set()
        for step in plan.steps:
            if isinstance(step.index, bool) or not isinstance(step.index, int) or step.index < 1:
                raise PlanActionBindingValidationError(
                    "Plan step indices must be positive integers."
                )
            if step.index in indices:
                raise PlanActionBindingValidationError(
                    f"Duplicate plan step index: {step.index}."
                )
            indices.add(step.index)
            if not isinstance(step.action, str) or not step.action.strip():
                raise PlanActionBindingValidationError(
                    f"Plan step {step.index} has an invalid action name."
                )

        if not discovery.environment_instance_id or not discovery.environment_type:
            raise PlanActionBindingValidationError(
                "Discovery report has invalid environment identity."
            )

    def bind(
        self,
        plan: LLMPlan,
        discovery: EnvironmentCapabilityDiscoveryReport,
        *,
        require_complete: bool = True,
    ) -> PlanActionBindingReport:
        if not isinstance(require_complete, bool):
            raise PlanActionBindingValidationError(
                "require_complete must be a boolean."
            )

        self._validate_inputs(plan, discovery)

        with self._lock:
            by_action: dict[str, list] = {}
            for candidate in discovery.candidates:
                if candidate.state == EnvironmentCapabilityState.AVAILABLE:
                    by_action.setdefault(candidate.action_name, []).append(candidate)

            bound = []
            unbound = []
            for step in plan.steps:
                matches = by_action.get(step.action, [])
                if len(matches) != 1:
                    reason = "capability_not_available" if not matches else "ambiguous_action_binding"
                    unbound.append((step.index, step.action, reason))
                    continue

                candidate = matches[0]
                if (
                    candidate.environment_instance_id
                    != discovery.environment_instance_id
                    or candidate.registry_version != discovery.registry_version
                ):
                    raise PlanActionBindingValidationError(
                        "Discovery candidate identity/version mismatch."
                    )

                bound.append(BoundPlanStep(
                    step_index=step.index,
                    action_name=step.action,
                    capability_id=candidate.capability_id,
                    environment_instance_id=discovery.environment_instance_id,
                    registry_version=discovery.registry_version,
                    risk_level=candidate.risk_level,
                    permissions=tuple(candidate.permissions),
                ))

            report = PlanActionBindingReport(
                plan_goal=plan.goal,
                environment_instance_id=discovery.environment_instance_id,
                environment_type=discovery.environment_type,
                registry_version=discovery.registry_version,
                steps=tuple(bound),
                unbound_steps=tuple(unbound),
                state=(
                    PlanActionBindingState.BOUND
                    if not unbound
                    else PlanActionBindingState.UNBOUND
                ),
            )
            self.last_report = report

            if require_complete and not report.is_complete:
                details = ", ".join(
                    f"{index}:{action}({reason})"
                    for index, action, reason in report.unbound_steps
                )
                raise PlanActionBindingValidationError(
                    "Plan contains steps without one available trusted capability: "
                    + details
                )
            return report


__all__ = [
    "PlanActionBindingError",
    "PlanActionBindingValidationError",
    "PlanActionBindingState",
    "BoundPlanStep",
    "PlanActionBindingReport",
    "PlanActionBinder",
]
