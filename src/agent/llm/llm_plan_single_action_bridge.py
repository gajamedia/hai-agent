from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .canonical_single_action import (
    CanonicalSingleActionRequest,
    CanonicalSingleActionResult,
)
from .agent_canonical_single_action_api import AgentCanonicalSingleActionAPI
from .planning import LLMPlan, LLMPlanStep


class LLMPlanSingleActionBridgeError(RuntimeError):
    """Base error for LLM-plan-to-single-action bridging."""


class LLMPlanSingleActionBridgeConfigurationError(
    LLMPlanSingleActionBridgeError
):
    """Raised when the bridge is configured with an invalid API."""


class LLMPlanSingleActionBridgeRequestError(
    LLMPlanSingleActionBridgeError
):
    """Raised when an LLM plan step cannot become a canonical request."""


@dataclass(frozen=True)
class LLMPlanSingleActionExecutionRequest:
    """Traceable pairing of an LLM plan step and its canonical request."""

    plan: LLMPlan
    step: LLMPlanStep
    request: CanonicalSingleActionRequest

    @property
    def action(self) -> str:
        return self.step.action

    @property
    def position(self) -> int:
        return self.step.index


class LLMPlanSingleActionBridge:
    """
    Translate exactly one LLM plan step into the canonical Agent action API.

    This layer is deliberately narrow:

    * one plan step becomes one CanonicalSingleActionRequest;
    * it never executes the whole plan;
    * it never invokes an LLM;
    * it never validates action availability itself;
    * action availability remains the responsibility of the Agent binding;
    * it never calls Agent.decide_and_act().
    """

    def __init__(
        self,
        api: AgentCanonicalSingleActionAPI,
    ) -> None:
        if api is None:
            raise LLMPlanSingleActionBridgeConfigurationError(
                "api cannot be None."
            )
        if not callable(getattr(api, "execute", None)):
            raise LLMPlanSingleActionBridgeConfigurationError(
                "api must expose callable execute(...)."
            )
        self.api = api
        self.last_execution_request: LLMPlanSingleActionExecutionRequest | None = None
        self.last_result: CanonicalSingleActionResult | None = None

    @classmethod
    def for_api(
        cls,
        api: AgentCanonicalSingleActionAPI,
    ) -> "LLMPlanSingleActionBridge":
        return cls(api)

    @staticmethod
    def _coerce_step(
        plan: LLMPlan,
        step: LLMPlanStep | int,
    ) -> LLMPlanStep:
        if not isinstance(plan, LLMPlan):
            raise TypeError("plan must be an LLMPlan.")

        if isinstance(step, bool):
            raise LLMPlanSingleActionBridgeRequestError(
                "step must be an LLMPlanStep or positive step index."
            )

        if isinstance(step, int):
            for candidate in plan.steps:
                if candidate.index == step:
                    return candidate
            raise LLMPlanSingleActionBridgeRequestError(
                f"Plan step index {step} does not exist."
            )

        if not isinstance(step, LLMPlanStep):
            raise TypeError("step must be an LLMPlanStep or positive step index.")

        for candidate in plan.steps:
            if candidate.index == step.index:
                if candidate != step:
                    raise LLMPlanSingleActionBridgeRequestError(
                        f"Plan step index {step.index} does not match the supplied step."
                    )
                return candidate

        raise LLMPlanSingleActionBridgeRequestError(
            f"Plan does not contain step index {step.index}."
        )

    @staticmethod
    def build_request(
        plan: LLMPlan,
        step: LLMPlanStep | int,
        *,
        goal: Any = None,
        world_state: Any = None,
        environment: Any = None,
        strategies: Any = None,
        information_gain: Mapping[str, float] | None = None,
        action_risk: Mapping[str, float] | None = None,
        exploration_risk: float = 0.0,
        action_cost: Mapping[str, float] | None = None,
        experience_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlanSingleActionExecutionRequest:
        selected = LLMPlanSingleActionBridge._coerce_step(plan, step)
        resolved_goal = plan.goal if goal is None else goal

        bridge_metadata = {
            "source": "llm_plan_single_action_bridge",
            "llm_plan_goal": plan.goal,
            "llm_plan_confidence": plan.confidence,
            "llm_plan_provider": plan.provider,
            "llm_plan_model": plan.model,
            "llm_plan_step_index": selected.index,
            "llm_plan_step_action": selected.action,
            "llm_plan_step_purpose": selected.purpose,
            "llm_plan_expected_observation": selected.expected_observation,
            "llm_plan_success_criteria": selected.success_criteria,
            "llm_plan_step_risk": selected.risk,
        }
        bridge_metadata.update(dict(metadata or {}))

        request = CanonicalSingleActionRequest(
            action=selected.action,
            step=selected,
            position=selected.index,
            goal=resolved_goal,
            world_state=world_state,
            environment=environment,
            strategies=strategies,
            information_gain=dict(information_gain or {}),
            action_risk=dict(action_risk or {}),
            exploration_risk=exploration_risk,
            action_cost=dict(action_cost or {}),
            experience_id=experience_id,
            metadata=bridge_metadata,
        )

        return LLMPlanSingleActionExecutionRequest(
            plan=plan,
            step=selected,
            request=request,
        )

    def execute_step(
        self,
        plan: LLMPlan,
        step: LLMPlanStep | int,
        *,
        goal: Any = None,
        world_state: Any = None,
        environment: Any = None,
        strategies: Any = None,
        information_gain: Mapping[str, float] | None = None,
        action_risk: Mapping[str, float] | None = None,
        exploration_risk: float = 0.0,
        action_cost: Mapping[str, float] | None = None,
        experience_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CanonicalSingleActionResult:
        execution_request = self.build_request(
            plan,
            step,
            goal=goal,
            world_state=world_state,
            environment=environment,
            strategies=strategies,
            information_gain=information_gain,
            action_risk=action_risk,
            exploration_risk=exploration_risk,
            action_cost=action_cost,
            experience_id=experience_id,
            metadata=metadata,
        )

        result = self.api.execute(execution_request.request)
        self.last_execution_request = execution_request
        self.last_result = result
        return result

    def execute_step_index(
        self,
        plan: LLMPlan,
        step_index: int,
        **kwargs: Any,
    ) -> CanonicalSingleActionResult:
        return self.execute_step(plan, step_index, **kwargs)


__all__ = [
    "LLMPlanSingleActionBridge",
    "LLMPlanSingleActionBridgeConfigurationError",
    "LLMPlanSingleActionBridgeError",
    "LLMPlanSingleActionBridgeRequestError",
    "LLMPlanSingleActionExecutionRequest",
]
