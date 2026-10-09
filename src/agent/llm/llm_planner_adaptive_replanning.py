from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .context import AgentContext, AgentContextBuilder
from .llm_plan_adaptive_replanning import (
    LLMPlanAdaptiveReplanningResult,
    LLMPlanAdaptiveReplanner,
    LLMPlanReplanningRequest,
)
from .planning import LLMPlan, LLMPlanStep, LLMPlanner
from .reasoning import LLMReasoning


class LLMPlannerAdaptiveReplanningError(RuntimeError):
    """Base error for Stage 24.15 LLM planner integration."""


class LLMPlannerAdaptiveReplanningConfigurationError(
    LLMPlannerAdaptiveReplanningError
):
    """Raised when the LLM planner integration is misconfigured."""


class LLMPlannerAdaptiveReplanningRequestError(
    LLMPlannerAdaptiveReplanningError
):
    """Raised when an adaptive replanning request cannot be normalized."""


class LLMPlannerAdaptiveReplanningBuildError(LLMPlannerAdaptiveReplanningError):
    """Raised when context/reasoning cannot be built for a replan request."""


@dataclass(frozen=True)
class LLMPlannerAdaptiveReplanningContext:
    """Diagnostics describing the context sent to the Stage 23 planner."""

    goal: Any
    world_state: Any
    preserved_steps: tuple[LLMPlanStep, ...]
    remaining_steps: tuple[LLMPlanStep, ...]
    failed_step_index: int | None
    next_step_index: int
    generation: int
    feedback_step_index: int
    feedback_action: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def preserved_step_indices(self) -> tuple[int, ...]:
        return tuple(step.index for step in self.preserved_steps)

    @property
    def remaining_step_indices(self) -> tuple[int, ...]:
        return tuple(step.index for step in self.remaining_steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "world_state": self.world_state,
            "preserved_steps": [step.to_dict() for step in self.preserved_steps],
            "remaining_steps": [step.to_dict() for step in self.remaining_steps],
            "failed_step_index": self.failed_step_index,
            "next_step_index": self.next_step_index,
            "generation": self.generation,
            "feedback_step_index": self.feedback_step_index,
            "feedback_action": self.feedback_action,
            "metadata": dict(self.metadata),
        }


AdaptiveReplanningContextBuilder = Callable[
    [LLMPlanReplanningRequest], AgentContext
]
AdaptiveReplanningReasoningBuilder = Callable[
    [LLMPlanReplanningRequest], LLMReasoning
]
AdaptiveReplanningRiskResolver = Callable[[str | None], float]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, numeric))


def _step_as_dict(step: LLMPlanStep) -> dict[str, Any]:
    return step.to_dict()


def _default_context_builder(
    request: LLMPlanReplanningRequest,
    *,
    builder: AgentContextBuilder,
) -> AgentContext:
    """Build a provider-neutral context from the information available in a replan request."""

    feedback = request.feedback
    runtime_snapshot = {
        "feedback": {
            "step_index": feedback.step_index,
            "action": feedback.action,
            "success": feedback.success,
            "progress_delta": feedback.progress_delta,
            "evidence_quality": feedback.evidence_quality,
            "confidence": feedback.confidence,
            "diagnosis": feedback.diagnosis,
            "lesson": feedback.lesson,
            "next_step": feedback.next_step,
            "change_required": feedback.change_required,
            "continue_recommended": feedback.continue_recommended,
            "exploration_recommended": feedback.exploration_recommended,
            "risk_level": feedback.risk_level,
            "risk_score": feedback.risk_score,
            "information_gain": feedback.information_gain,
        },
        "failed_step_index": request.failed_step_index,
        "next_runtime_step_index": request.next_step_index,
        "generation": request.generation,
    }

    planning_memory = (
        {
            "type": "preserved_successful_step",
            "step": _step_as_dict(step),
        }
        for step in request.preserved_steps
    )
    remaining_memory = (
        {
            "type": "remaining_original_step",
            "step": _step_as_dict(step),
        }
        for step in request.remaining_steps
    )

    return builder.build(
        goal=request.goal,
        state={
            "runtime_generation": request.generation,
            "next_step_index": request.next_step_index,
            "failed_step_index": request.failed_step_index,
            "feedback_step_index": feedback.step_index,
        },
        world_model=request.world_state,
        relevant_memory=tuple(planning_memory) + tuple(remaining_memory),
        strategy={
            "mode": "adaptive_replanning",
            "direction": feedback.next_step,
            "preserve_completed_successful_steps": True,
        },
        self_awareness={
            "feedback_confidence": _clamp(feedback.confidence),
            "evidence_quality": _clamp(feedback.evidence_quality),
        },
        calibration={
            "risk_level": feedback.risk_level,
            "risk_score": _clamp(feedback.risk_score),
        },
        metacognition=runtime_snapshot,
        metadata={
            **dict(request.metadata),
            "operation": "adaptive_replanning",
            "stage": "24.15",
        },
    )


def _default_reasoning_builder(request: LLMPlanReplanningRequest) -> LLMReasoning:
    """Create a deterministic high-level reasoning envelope for the planner."""

    feedback = request.feedback
    conclusions: list[str] = []
    alternatives: list[str] = []
    risks: list[str] = []
    information_gaps: list[str] = []

    if feedback.success is False:
        conclusions.append(
            f"The previous step '{feedback.action}' failed and should not be repeated automatically."
        )
    elif feedback.progress_delta is not None and feedback.progress_delta <= 0.0:
        conclusions.append(
            f"The previous step '{feedback.action}' produced insufficient progress."
        )
    else:
        conclusions.append(
            f"The previous step '{feedback.action}' completed with available evidence."
        )

    conclusions.append(
        f"The runtime is at adaptive planning generation {request.generation}."
    )

    if request.remaining_steps:
        conclusions.append(
            f"There are {len(request.remaining_steps)} original remaining step(s) that may need replacement."
        )
    else:
        conclusions.append("There are no original remaining steps; the replanner must propose recovery work only if needed.")

    if feedback.exploration_recommended:
        alternatives.append(
            "Prefer gathering additional information before committing to a high-risk continuation."
        )
    if feedback.continue_recommended:
        alternatives.append(
            "A conservative continuation may still be appropriate when it directly addresses the feedback."
        )

    if feedback.risk_level:
        risks.append(
            f"Runtime feedback currently classifies risk as {feedback.risk_level}."
        )
    if feedback.change_required:
        risks.append("The previous strategy is marked as requiring behavioral change.")

    if feedback.next_step:
        information_gaps.append(feedback.next_step)
    if feedback.diagnosis:
        information_gaps.append(f"Diagnosis: {feedback.diagnosis}")

    direction = (
        "Replan the remaining work from the current world state, preserve completed successful work, "
        "and prefer the safest evidence-backed route."
    )

    if feedback.success is False:
        summary = (
            "Adaptive replanning is required after the previous step failed because "
            f"runtime feedback indicates {feedback.diagnosis or 'a change in execution conditions'}."
        )
    else:
        summary = (
            "Adaptive replanning is required because runtime feedback indicates "
            f"{feedback.diagnosis or 'a change in execution conditions'}."
        )

    return LLMReasoning(
        summary=summary,
        conclusions=tuple(conclusions),
        alternatives=tuple(alternatives),
        risks=tuple(risks),
        information_gaps=tuple(information_gaps),
        recommended_direction=direction,
        confidence=_clamp(feedback.confidence),
        provider="adaptive-runtime",
        model="deterministic-feedback-envelope",
    )


def default_risk_resolver(risk: str | None) -> float:
    """Map Stage 23 textual step risk into the bounded numeric risk contract used by Stage 24 validation."""

    if risk is None or not str(risk).strip():
        return 0.0
    text = str(risk).casefold()
    if any(token in text for token in ("critical", "danger", "unsafe", "severe", "high")):
        return 0.75
    if any(token in text for token in ("low", "safe", "minimal")):
        return 0.25
    return 0.5


def _normalize_plan_risks(
    plan: LLMPlan,
    risk_resolver: AdaptiveReplanningRiskResolver,
) -> LLMPlan:
    """Normalize parsed textual risks without changing the canonical Stage 23 schema."""

    normalized_steps: list[LLMPlanStep] = []
    changed = False
    for step in plan.steps:
        raw_risk = step.risk
        if raw_risk is None or isinstance(raw_risk, (int, float)):
            normalized_risk = raw_risk
        else:
            normalized_risk = risk_resolver(raw_risk)
            changed = True
        normalized_steps.append(
            LLMPlanStep(
                index=step.index,
                action=step.action,
                purpose=step.purpose,
                preconditions=step.preconditions,
                expected_observation=step.expected_observation,
                success_criteria=step.success_criteria,
                risk=normalized_risk,
            )
        )

    if not changed:
        return plan

    return LLMPlan(
        goal=plan.goal,
        rationale=plan.rationale,
        steps=tuple(normalized_steps),
        assumptions=plan.assumptions,
        constraints=plan.constraints,
        stop_conditions=plan.stop_conditions,
        confidence=plan.confidence,
        provider=plan.provider,
        model=plan.model,
        raw_content=plan.raw_content,
    )


class LLMPlannerAdaptiveReplanningProvider:
    """
    Adapt Stage 23.6 ``LLMPlanner`` to Stage 24.13's replan-provider contract.

    The class deliberately reuses the canonical LLMPlanner. It does not create
    another LLM transport, parser, or plan schema. The returned ``LLMPlan`` is
    therefore parsed by the same Stage 23 planner and validated again by the
    existing Stage 24.13 ``LLMPlanAdaptiveReplanner``.
    """

    def __init__(
        self,
        planner: LLMPlanner,
        *,
        context_builder: AgentContextBuilder | AdaptiveReplanningContextBuilder | None = None,
        reasoning_builder: AdaptiveReplanningReasoningBuilder | None = None,
        risk_resolver: AdaptiveReplanningRiskResolver | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(planner, LLMPlanner):
            raise LLMPlannerAdaptiveReplanningConfigurationError(
                "planner must be an LLMPlanner instance."
            )

        if context_builder is not None and not (
            isinstance(context_builder, AgentContextBuilder)
            or callable(context_builder)
        ):
            raise LLMPlannerAdaptiveReplanningConfigurationError(
                "context_builder must be an AgentContextBuilder or callable."
            )
        if reasoning_builder is not None and not callable(reasoning_builder):
            raise LLMPlannerAdaptiveReplanningConfigurationError(
                "reasoning_builder must be callable."
            )
        if risk_resolver is not None and not callable(risk_resolver):
            raise LLMPlannerAdaptiveReplanningConfigurationError(
                "risk_resolver must be callable."
            )

        self.planner = planner
        self.context_builder = context_builder or AgentContextBuilder()
        self.reasoning_builder = reasoning_builder or _default_reasoning_builder
        self.risk_resolver = risk_resolver or default_risk_resolver
        self.metadata = dict(metadata or {})
        self.last_request: LLMPlanReplanningRequest | None = None
        self.last_context: AgentContext | None = None
        self.last_reasoning: LLMReasoning | None = None
        self.last_plan: LLMPlan | None = None

    def _build_context(self, request: LLMPlanReplanningRequest) -> AgentContext:
        try:
            if isinstance(self.context_builder, AgentContextBuilder):
                return _default_context_builder(
                    request,
                    builder=self.context_builder,
                )
            context = self.context_builder(request)
        except Exception as exc:
            raise LLMPlannerAdaptiveReplanningBuildError(
                f"Failed to build adaptive replanning context: {exc}"
            ) from exc

        if not isinstance(context, AgentContext):
            raise LLMPlannerAdaptiveReplanningBuildError(
                "context_builder must return an AgentContext."
            )
        return context

    def _build_reasoning(self, request: LLMPlanReplanningRequest) -> LLMReasoning:
        try:
            reasoning = self.reasoning_builder(request)
        except Exception as exc:
            raise LLMPlannerAdaptiveReplanningBuildError(
                f"Failed to build adaptive replanning reasoning: {exc}"
            ) from exc

        if not isinstance(reasoning, LLMReasoning):
            raise LLMPlannerAdaptiveReplanningBuildError(
                "reasoning_builder must return an LLMReasoning."
            )
        return reasoning

    def replan(self, request: LLMPlanReplanningRequest) -> LLMPlan:
        if not isinstance(request, LLMPlanReplanningRequest):
            raise LLMPlannerAdaptiveReplanningRequestError(
                "request must be an LLMPlanReplanningRequest."
            )
        if not isinstance(request.goal, str) or not request.goal.strip():
            raise LLMPlannerAdaptiveReplanningRequestError(
                "request.goal must be a non-empty string."
            )
        if request.next_step_index < 1:
            raise LLMPlannerAdaptiveReplanningRequestError(
                "request.next_step_index must be >= 1."
            )
        if request.generation < 1:
            raise LLMPlannerAdaptiveReplanningRequestError(
                "request.generation must be >= 1."
            )

        context = self._build_context(request)
        reasoning = self._build_reasoning(request)

        metadata = {
            **self.metadata,
            **dict(request.metadata),
            "stage": "24.15",
            "operation": "adaptive_replanning",
            "generation": request.generation,
            "feedback_step_index": request.feedback.step_index,
            "failed_step_index": request.failed_step_index,
            "next_step_index": request.next_step_index,
            "preserved_step_indices": request.preserved_step_indices,
            "remaining_original_step_indices": request.remaining_step_indices,
            "original_plan_goal": request.original_plan.goal,
        }

        try:
            plan = self.planner.plan(
                reasoning,
                context,
                metadata=metadata,
            )
        except Exception:
            raise

        if not isinstance(plan, LLMPlan):
            raise LLMPlannerAdaptiveReplanningBuildError(
                "LLMPlanner must return an LLMPlan."
            )

        try:
            normalized_plan = _normalize_plan_risks(
                plan,
                self.risk_resolver,
            )
            for step in normalized_plan.steps:
                if step.risk is not None:
                    if isinstance(step.risk, bool):
                        raise ValueError("risk must be numeric")
                    numeric_risk = float(step.risk)
                    if not 0.0 <= numeric_risk <= 1.0:
                        raise ValueError("risk must be between 0 and 1")
        except Exception as exc:
            raise LLMPlannerAdaptiveReplanningBuildError(
                f"Failed to normalize LLM plan risk values: {exc}"
            ) from exc

        self.last_request = request
        self.last_context = context
        self.last_reasoning = reasoning
        self.last_plan = normalized_plan
        return normalized_plan

    def __call__(self, request: LLMPlanReplanningRequest) -> LLMPlan:
        return self.replan(request)

    @classmethod
    def for_planner(
        cls,
        planner: LLMPlanner,
        **kwargs: Any,
    ) -> "LLMPlannerAdaptiveReplanningProvider":
        return cls(planner, **kwargs)

    def clear_trace(self) -> None:
        self.last_request = None
        self.last_context = None
        self.last_reasoning = None
        self.last_plan = None
        self.planner.clear_trace()


@dataclass(frozen=True)
class LLMPlannerAdaptiveReplanningResult:
    """Trace tying one Stage 24.13 replan result back to Stage 23 planning."""

    replanning: LLMPlanAdaptiveReplanningResult
    planner: LLMPlannerAdaptiveReplanningProvider
    llm_provider: str | None
    llm_model: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def replanned(self) -> bool:
        return self.replanning.replanned

    @property
    def replacement_plan(self) -> LLMPlan | None:
        return self.replanning.replacement_plan

    @property
    def llm_called(self) -> bool:
        return self.planner.last_request is not None


def build_llm_adaptive_replanner(
    planner: LLMPlanner,
    *,
    validator: Any = None,
    max_replans: int = 3,
    **kwargs: Any,
) -> tuple[LLMPlanAdaptiveReplanner, LLMPlannerAdaptiveReplanningProvider]:
    """Convenience factory producing the canonical Stage 24.13 replanner + Stage 23 planner adapter."""

    provider = LLMPlannerAdaptiveReplanningProvider(planner, **kwargs)
    replanner = LLMPlanAdaptiveReplanner(
        provider,
        validator=validator,
        max_replans=max_replans,
    )
    return replanner, provider


__all__ = [
    "AdaptiveReplanningContextBuilder",
    "AdaptiveReplanningReasoningBuilder",
    "AdaptiveReplanningRiskResolver",
    "LLMPlannerAdaptiveReplanningBuildError",
    "LLMPlannerAdaptiveReplanningConfigurationError",
    "LLMPlannerAdaptiveReplanningError",
    "LLMPlannerAdaptiveReplanningProvider",
    "LLMPlannerAdaptiveReplanningRequestError",
    "LLMPlannerAdaptiveReplanningContext",
    "LLMPlannerAdaptiveReplanningResult",
    "build_llm_adaptive_replanner",
    "default_risk_resolver",
]
