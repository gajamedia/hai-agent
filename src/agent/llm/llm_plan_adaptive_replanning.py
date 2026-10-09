from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedback,
)
from .plan_validation import LLMPlanValidationResult, LLMPlanValidator
from .planning import LLMPlan, LLMPlanStep


class LLMPlanAdaptiveReplanningError(RuntimeError):
    """Base error for adaptive runtime replanning."""


class LLMPlanAdaptiveReplanningConfigurationError(
    LLMPlanAdaptiveReplanningError
):
    """Raised when adaptive replanning is misconfigured."""


class LLMPlanAdaptiveReplanningRequestError(
    LLMPlanAdaptiveReplanningError
):
    """Raised when a replan request cannot be constructed safely."""


class LLMPlanAdaptiveReplanningValidationError(
    LLMPlanAdaptiveReplanningError
):
    """Raised when the replacement plan is invalid."""

    def __init__(self, validation: LLMPlanValidationResult) -> None:
        self.validation = validation
        message = "; ".join(issue.message for issue in validation.issues)
        super().__init__(message or "Replacement LLM plan validation failed.")


class LLMPlanAdaptiveReplanningSafetyError(
    LLMPlanAdaptiveReplanningError
):
    """Raised when a replacement plan would repeat a completed step."""


LLMPlanReplannerProvider = Callable[["LLMPlanReplanningRequest"], LLMPlan]


@dataclass(frozen=True)
class LLMPlanReplanningTrigger:
    """Immutable explanation of why a runtime replan is or is not needed."""

    should_replan: bool
    reason: str
    feedback_step_index: int | None
    progress_delta: float
    risk_level: str
    change_required: bool
    continue_recommended: bool
    exploration_recommended: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMPlanReplanningRequest:
    """Context supplied to an injected adaptive replan provider."""

    original_plan: LLMPlan
    goal: Any
    world_state: Any
    feedback: LLMPlanRuntimeLearningFeedback
    preserved_steps: tuple[LLMPlanStep, ...]
    remaining_steps: tuple[LLMPlanStep, ...]
    failed_step_index: int | None
    next_step_index: int
    generation: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def preserved_step_indices(self) -> tuple[int, ...]:
        return tuple(step.index for step in self.preserved_steps)

    @property
    def remaining_step_indices(self) -> tuple[int, ...]:
        return tuple(step.index for step in self.remaining_steps)


@dataclass(frozen=True)
class LLMPlanAdaptiveReplanningResult:
    """Result of one replan decision and, when triggered, its validated replacement."""

    trigger: LLMPlanReplanningTrigger
    original_plan: LLMPlan
    preserved_step_indices: tuple[int, ...]
    remaining_original_step_indices: tuple[int, ...]
    next_step_index: int
    generation: int
    request: LLMPlanReplanningRequest | None
    replacement_plan: LLMPlan | None
    validation: LLMPlanValidationResult | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def replanned(self) -> bool:
        return self.replacement_plan is not None

    @property
    def replacement_step_indices(self) -> tuple[int, ...]:
        if self.replacement_plan is None:
            return ()
        return tuple(step.index for step in self.replacement_plan.steps)

    @property
    def completed_successfully(self) -> tuple[int, ...]:
        return self.preserved_step_indices



def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, numeric))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _step_signature(step: LLMPlanStep) -> tuple[Any, ...]:
    """Semantic identity used only to prevent exact completed-step repetition."""

    return (
        _text(step.action).casefold(),
        _text(step.purpose).casefold(),
        tuple(_text(item).casefold() for item in step.preconditions),
        None
        if step.expected_observation is None
        else _text(step.expected_observation).casefold(),
        None
        if step.success_criteria is None
        else _text(step.success_criteria).casefold(),
    )


class LLMPlanAdaptiveReplanner:
    """
    Decide when runtime learning feedback requires a replan and safely build
    a validated replacement plan for the *remaining* work.

    This component deliberately does not execute actions, mutate Agent state,
    invoke ``Agent.decide_and_act()``, or call an LLM directly. The actual
    replan provider is injected as a callback, keeping the runtime independent
    from OpenCode, a future custom LLM, or any other provider.
    """

    def __init__(
        self,
        replanner: LLMPlanReplannerProvider,
        *,
        validator: LLMPlanValidator | None = None,
        max_replans: int = 3,
        replan_on_stagnation: bool = True,
        replan_on_high_risk: bool = True,
        min_progress_delta: float = 0.0,
        preserve_successful_steps: bool = True,
        prevent_exact_completed_step_repetition: bool = True,
        max_replacement_steps: int | None = None,
    ) -> None:
        if not callable(replanner):
            raise LLMPlanAdaptiveReplanningConfigurationError(
                "replanner must be callable."
            )
        if validator is not None and not callable(getattr(validator, "validate", None)):
            raise LLMPlanAdaptiveReplanningConfigurationError(
                "validator must expose callable validate(...)."
            )
        if max_replans < 0:
            raise LLMPlanAdaptiveReplanningConfigurationError(
                "max_replans cannot be negative."
            )
        if min_progress_delta < 0.0 or min_progress_delta > 1.0:
            raise LLMPlanAdaptiveReplanningConfigurationError(
                "min_progress_delta must be between 0 and 1."
            )
        if max_replacement_steps is not None and max_replacement_steps < 1:
            raise LLMPlanAdaptiveReplanningConfigurationError(
                "max_replacement_steps must be >= 1 when provided."
            )

        self.replanner = replanner
        self.validator = validator or LLMPlanValidator()
        self.max_replans = int(max_replans)
        self.replan_on_stagnation = bool(replan_on_stagnation)
        self.replan_on_high_risk = bool(replan_on_high_risk)
        self.min_progress_delta = float(min_progress_delta)
        self.preserve_successful_steps = bool(preserve_successful_steps)
        self.prevent_exact_completed_step_repetition = bool(
            prevent_exact_completed_step_repetition
        )
        self.max_replacement_steps = max_replacement_steps
        self.replan_count = 0
        self.last_trigger: LLMPlanReplanningTrigger | None = None
        self.last_result: LLMPlanAdaptiveReplanningResult | None = None

    @property
    def can_replan(self) -> bool:
        return self.replan_count < self.max_replans

    def evaluate_trigger(
        self,
        feedback: LLMPlanRuntimeLearningFeedback,
    ) -> LLMPlanReplanningTrigger:
        if feedback is None:
            raise LLMPlanAdaptiveReplanningRequestError(
                "feedback cannot be None."
            )
        progress_delta = _clamp(feedback.progress_delta, -1.0, 1.0)
        risk_level = _text(feedback.risk_level).casefold() or "low"
        change_required = bool(feedback.change_required)
        continue_recommended = bool(feedback.continue_recommended)
        exploration_recommended = bool(feedback.exploration_recommended)

        reasons: list[str] = []
        if change_required:
            reasons.append("change_required")
        if not continue_recommended:
            reasons.append("continue_not_recommended")
        if self.replan_on_stagnation and progress_delta <= self.min_progress_delta:
            reasons.append("progress_stagnation")
        if self.replan_on_high_risk and risk_level == "high":
            reasons.append("high_risk")

        should_replan = bool(reasons)
        reason = "|".join(reasons) if reasons else "no_replan_trigger"
        if should_replan and not self.can_replan:
            reason = f"replan_limit_reached:{reason}"
            should_replan = False

        trigger = LLMPlanReplanningTrigger(
            should_replan=should_replan,
            reason=reason,
            feedback_step_index=int(feedback.step_index),
            progress_delta=progress_delta,
            risk_level=risk_level,
            change_required=change_required,
            continue_recommended=continue_recommended,
            exploration_recommended=exploration_recommended,
            metadata={
                "replan_count": self.replan_count,
                "max_replans": self.max_replans,
            },
        )
        self.last_trigger = trigger
        return trigger

    @staticmethod
    def _normalize_successful_indices(
        original_plan: LLMPlan,
        successful_step_indices: Sequence[int],
    ) -> tuple[int, ...]:
        original_indices = {step.index for step in original_plan.steps}
        normalized: list[int] = []
        for value in successful_step_indices:
            if isinstance(value, bool):
                raise LLMPlanAdaptiveReplanningRequestError(
                    "successful_step_indices must contain integers."
                )
            try:
                index = int(value)
            except (TypeError, ValueError) as exc:
                raise LLMPlanAdaptiveReplanningRequestError(
                    "successful_step_indices must contain integers."
                ) from exc
            if index not in original_indices:
                raise LLMPlanAdaptiveReplanningRequestError(
                    f"Successful step index {index} is not part of the original plan."
                )
            normalized.append(index)
        return tuple(sorted(set(normalized)))

    @staticmethod
    def _remaining_steps(
        original_plan: LLMPlan,
        preserved_step_indices: Sequence[int],
    ) -> tuple[LLMPlanStep, ...]:
        preserved = set(preserved_step_indices)
        return tuple(
            step
            for step in sorted(original_plan.steps, key=lambda item: item.index)
            if step.index not in preserved
        )

    @staticmethod
    def _rebase_plan(
        candidate: LLMPlan,
        *,
        next_step_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlan:
        if next_step_index < 1:
            raise LLMPlanAdaptiveReplanningRequestError(
                "next_step_index must be >= 1."
            )
        ordered = tuple(sorted(candidate.steps, key=lambda item: item.index))
        steps = tuple(
            replace(step, index=next_step_index + offset)
            for offset, step in enumerate(ordered)
        )
        assumptions = tuple(candidate.assumptions)
        constraints = tuple(candidate.constraints)
        merged_constraints = list(constraints)
        marker = "preserve_completed_successful_steps"
        if marker not in merged_constraints:
            merged_constraints.append(marker)
        for item in tuple(
            _text(v) for v in (metadata or {}).get("replan_constraints", ())
        ):
            if item and item not in merged_constraints:
                merged_constraints.append(item)

        return replace(
            candidate,
            steps=steps,
            constraints=tuple(merged_constraints),
            raw_content=candidate.raw_content,
        )

    def _validate_replacement(
        self,
        replacement_plan: LLMPlan,
        *,
        goal: Any,
        available_actions: Any,
    ) -> LLMPlanValidationResult:
        validation = self.validator.validate(
            replacement_plan,
            goal=goal,
            available_actions=available_actions,
        )
        if not validation.valid:
            raise LLMPlanAdaptiveReplanningValidationError(validation)
        return validation

    def _enforce_repetition_safety(
        self,
        replacement_plan: LLMPlan,
        preserved_steps: Sequence[LLMPlanStep],
    ) -> None:
        if not self.prevent_exact_completed_step_repetition:
            return
        preserved_signatures = {
            _step_signature(step)
            for step in preserved_steps
        }
        for step in replacement_plan.steps:
            if _step_signature(step) in preserved_signatures:
                raise LLMPlanAdaptiveReplanningSafetyError(
                    "Replacement plan repeats a completed successful step "
                    f"semantically: action={step.action!r}, purpose={step.purpose!r}."
                )

    def replan(
        self,
        original_plan: LLMPlan,
        *,
        feedback: LLMPlanRuntimeLearningFeedback,
        successful_step_indices: Sequence[int] = (),
        goal: Any = None,
        world_state: Any = None,
        failed_step_index: int | None = None,
        next_step_index: int | None = None,
        generation: int | None = None,
        available_actions: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlanAdaptiveReplanningResult:
        if not isinstance(original_plan, LLMPlan):
            raise TypeError("original_plan must be an LLMPlan.")
        trigger = self.evaluate_trigger(feedback)
        preserved_indices = self._normalize_successful_indices(
            original_plan,
            successful_step_indices,
        )
        if not self.preserve_successful_steps:
            preserved_indices = ()

        remaining_steps = self._remaining_steps(
            original_plan,
            preserved_indices,
        )
        remaining_indices = tuple(step.index for step in remaining_steps)
        resolved_goal = original_plan.goal if goal is None else goal
        if next_step_index is None:
            resolved_next_index = (
                min(remaining_indices)
                if remaining_indices
                else max((step.index for step in original_plan.steps), default=0) + 1
            )
        else:
            resolved_next_index = int(next_step_index)
        if resolved_next_index < 1:
            raise LLMPlanAdaptiveReplanningRequestError(
                "next_step_index must be >= 1."
            )
        resolved_generation = (
            self.replan_count + 1 if generation is None else int(generation)
        )
        if resolved_generation < 1:
            raise LLMPlanAdaptiveReplanningRequestError(
                "generation must be >= 1."
            )
        if failed_step_index is not None and failed_step_index not in {
            step.index for step in original_plan.steps
        }:
            raise LLMPlanAdaptiveReplanningRequestError(
                "failed_step_index must belong to the original plan."
            )

        base_metadata = dict(metadata or {})
        base_metadata.update(
            {
                "stage": "24.13",
                "replanning": True,
                "trigger_reason": trigger.reason,
                "preserved_step_indices": preserved_indices,
                "remaining_original_step_indices": remaining_indices,
                "generation": resolved_generation,
            }
        )

        if not trigger.should_replan:
            result = LLMPlanAdaptiveReplanningResult(
                trigger=trigger,
                original_plan=original_plan,
                preserved_step_indices=preserved_indices,
                remaining_original_step_indices=remaining_indices,
                next_step_index=resolved_next_index,
                generation=resolved_generation,
                request=None,
                replacement_plan=None,
                validation=None,
                metadata=base_metadata,
            )
            self.last_result = result
            return result

        request = LLMPlanReplanningRequest(
            original_plan=original_plan,
            goal=resolved_goal,
            world_state=world_state,
            feedback=feedback,
            preserved_steps=tuple(
                step
                for step in sorted(original_plan.steps, key=lambda item: item.index)
                if step.index in set(preserved_indices)
            ),
            remaining_steps=remaining_steps,
            failed_step_index=failed_step_index,
            next_step_index=resolved_next_index,
            generation=resolved_generation,
            metadata=base_metadata,
        )

        candidate = self.replanner(request)
        if not isinstance(candidate, LLMPlan):
            raise LLMPlanAdaptiveReplanningRequestError(
                "replanner must return an LLMPlan."
            )
        if not candidate.steps:
            raise LLMPlanAdaptiveReplanningRequestError(
                "replanner returned an empty replacement plan."
            )
        if candidate.goal.casefold() != _text(resolved_goal).casefold():
            raise LLMPlanAdaptiveReplanningValidationError(
                self._invalid_goal_result(candidate, resolved_goal)
            )
        if (
            self.max_replacement_steps is not None
            and len(candidate.steps) > self.max_replacement_steps
        ):
            raise LLMPlanAdaptiveReplanningRequestError(
                "Replacement plan exceeds max_replacement_steps."
            )

        replacement = self._rebase_plan(
            candidate,
            next_step_index=resolved_next_index,
            metadata={
                "replan_constraints": (
                    "do_not_repeat_completed_successful_steps",
                )
            },
        )
        self._enforce_repetition_safety(
            replacement,
            request.preserved_steps,
        )
        validation = self._validate_replacement(
            replacement,
            goal=resolved_goal,
            available_actions=available_actions,
        )

        self.replan_count += 1
        result_metadata = dict(base_metadata)
        result_metadata.update(
            {
                "replacement_step_indices": tuple(
                    step.index for step in replacement.steps
                ),
                "replan_count": self.replan_count,
            }
        )
        result = LLMPlanAdaptiveReplanningResult(
            trigger=trigger,
            original_plan=original_plan,
            preserved_step_indices=preserved_indices,
            remaining_original_step_indices=remaining_indices,
            next_step_index=resolved_next_index,
            generation=resolved_generation,
            request=request,
            replacement_plan=replacement,
            validation=validation,
            metadata=result_metadata,
        )
        self.last_result = result
        return result

    @staticmethod
    def _invalid_goal_result(
        candidate: LLMPlan,
        goal: Any,
    ) -> LLMPlanValidationResult:
        from .plan_validation import PlanValidationIssue

        return LLMPlanValidationResult(
            valid=False,
            issues=(
                PlanValidationIssue(
                    code="goal_mismatch",
                    message=(
                        "Replacement plan goal does not match the active goal: "
                        f"{candidate.goal!r} != {_text(goal)!r}."
                    ),
                ),
            ),
            checked_step_count=len(candidate.steps),
            executable_step_indices=(),
        )

    def reset(self) -> None:
        self.replan_count = 0
        self.last_trigger = None
        self.last_result = None


__all__ = [
    "LLMPlanAdaptiveReplanningError",
    "LLMPlanAdaptiveReplanningConfigurationError",
    "LLMPlanAdaptiveReplanningRequestError",
    "LLMPlanAdaptiveReplanningValidationError",
    "LLMPlanAdaptiveReplanningSafetyError",
    "LLMPlanReplannerProvider",
    "LLMPlanReplanningTrigger",
    "LLMPlanReplanningRequest",
    "LLMPlanAdaptiveReplanningResult",
    "LLMPlanAdaptiveReplanner",
]
