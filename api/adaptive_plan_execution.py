
"""Closed-loop adaptive execution using the existing plan runtime and planner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent.llm.llm_plan_adaptive_replanning import LLMPlanAdaptiveReplanner
from agent.llm.llm_plan_adaptive_runtime import (
    LLMPlanAdaptiveRuntime,
    LLMPlanAdaptiveRuntimeExecutionReport,
)
from agent.llm.llm_plan_agent_experience_integration_adaptive import (
    LLMPlanAgentExperienceIntegrationResult,
    LLMPlanAgentExperienceIntegrator,
)
from agent.llm.llm_planner_adaptive_replanning import (
    LLMPlannerAdaptiveReplanningProvider,
)
from agent.llm.planning import LLMPlan


@dataclass(frozen=True)
class AdaptivePlanExecution:
    """Combined experience-integrator result and adaptive runtime report."""

    integration: LLMPlanAgentExperienceIntegrationResult
    adaptive_report: LLMPlanAdaptiveRuntimeExecutionReport
    replanning_provider: LLMPlannerAdaptiveReplanningProvider


def execute_adaptive_plan(
    *,
    agent: Any,
    live_runtime: Any,
    planner: Any,
    feedback_bridge: Any,
    plan: LLMPlan,
    goal: str,
    world_state: Mapping[str, Any],
    available_actions: Any,
    environment: Any = None,
    metadata: Mapping[str, Any] | None = None,
    validator: Any = None,
    experience_integrator_options: Mapping[str, Any] | None = None,
    max_replans: int = 1,
) -> AdaptivePlanExecution:
    """Execute bounded replanning while recording every generation's steps.

    The existing live runtime remains the only action executor. The supplied
    validator is applied by the adaptive replanner before replacement plans run.

    The same environment instance is passed through the experience integrator
    and adaptive runtime to every plan generation.
    """
    if not isinstance(plan, LLMPlan):
        raise TypeError("plan must be an LLMPlan")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a non-empty string")
    if isinstance(max_replans, bool) or not isinstance(max_replans, int) or max_replans < 0:
        raise ValueError("max_replans must be a non-negative integer")

    provider = LLMPlannerAdaptiveReplanningProvider(
        planner,
        metadata={
            "source": "execution_plan_run",
            "execution_mode": "isolated_simulation",
            "adaptive_replanning": True,
        },
    )
    replanner = LLMPlanAdaptiveReplanner(
        provider,
        validator=validator,
        max_replans=max_replans,
    )
    adaptive_runtime = LLMPlanAdaptiveRuntime(
        live_runtime,
        replanner,
        feedback_source=feedback_bridge,
        max_segments=max_replans + 1,
    )

    integrator_options = dict(experience_integrator_options or {})
    integrator = LLMPlanAgentExperienceIntegrator(
        agent,
        adaptive_runtime,
        experience_observer=feedback_bridge.observe_experience,
        project_agent_action=integrator_options.pop("project_agent_action", True),
        project_agent_observation=integrator_options.pop(
            "project_agent_observation", True
        ),
        advance_agent_cycle=integrator_options.pop("advance_agent_cycle", False),
        **integrator_options,
    )

    runtime_kwargs: dict[str, Any] = {
        "goal": goal,
        "world_state": dict(world_state),
        "available_actions": available_actions,
        "metadata": dict(metadata or {}),
    }
    if environment is not None:
        runtime_kwargs["environment"] = environment

    integration_result = integrator.execute(plan, **runtime_kwargs)

    adaptive_report = integration_result.runtime_result
    if not isinstance(adaptive_report, LLMPlanAdaptiveRuntimeExecutionReport):
        raise RuntimeError(
            "Adaptive runtime returned an unexpected report type: "
            f"{type(adaptive_report).__name__}"
        )

    return AdaptivePlanExecution(
        integration=integration_result,
        adaptive_report=adaptive_report,
        replanning_provider=provider,
    )


__all__ = ["AdaptivePlanExecution", "execute_adaptive_plan"]
