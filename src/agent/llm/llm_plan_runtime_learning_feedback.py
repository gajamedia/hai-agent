from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from threading import RLock
from typing import Any


class LLMPlanRuntimeLearningFeedbackError(RuntimeError):
    """Base error for runtime learning feedback."""


class LLMPlanRuntimeLearningFeedbackConfigurationError(
    LLMPlanRuntimeLearningFeedbackError
):
    """Raised when runtime learning feedback is misconfigured."""


class LLMPlanRuntimeLearningFeedbackInputError(
    LLMPlanRuntimeLearningFeedbackError
):
    """Raised when an experience or reflection cannot be consumed."""


ReflectionProvider = Callable[[Any], Any | None]
FeedbackObserver = Callable[["LLMPlanRuntimeLearningFeedback"], None]
RuntimeEvaluationProvider = Callable[
    ["LLMPlanRuntimeLearningFeedback", Any], Any
]


@dataclass(frozen=True)
class LLMPlanRuntimeLearningFeedback:
    """Immutable feedback derived from one observed runtime experience."""

    step_index: int
    action: str
    success: bool | None
    progress_delta: float | None
    evidence_quality: float
    confidence: float
    diagnosis: str
    lesson: str
    next_step: str
    change_required: bool
    continue_recommended: bool
    exploration_recommended: bool
    risk_level: str
    risk_score: float
    information_gain: float
    reflection: Any = None
    episode: Any = None
    runtime_evaluation: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def should_change_behavior(self) -> bool:
        return self.change_required or not self.continue_recommended

    @property
    def recovery_recommended(self) -> bool:
        return self.success is False

    @property
    def is_positive_progress(self) -> bool:
        return self.progress_delta is not None and self.progress_delta > 0.0


def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return minimum
    if not isfinite(numeric):
        return minimum
    return max(minimum, min(maximum, numeric))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _read_value(source: Any, *names: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return None
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return None


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


class LLMPlanRuntimeLearningFeedbackEngine:
    """
    Convert deterministic metacognitive reflection into bounded runtime
    feedback. This layer never executes actions and never mutates the Agent.
    """

    _RISK_LABELS = {
        "low": 0.25,
        "medium": 0.50,
        "high": 0.75,
    }

    def build(
        self,
        experience: Any,
        reflection: Any,
        *,
        runtime_evaluation: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlanRuntimeLearningFeedback:
        if experience is None:
            raise LLMPlanRuntimeLearningFeedbackInputError(
                "experience cannot be None."
            )
        if reflection is None:
            raise LLMPlanRuntimeLearningFeedbackInputError(
                "reflection cannot be None."
            )

        step_index_value = _read_value(experience, "step_index", "index")
        try:
            step_index = int(step_index_value)
        except (TypeError, ValueError):
            raise LLMPlanRuntimeLearningFeedbackInputError(
                "experience must expose an integer step_index."
            ) from None
        if step_index < 0:
            raise LLMPlanRuntimeLearningFeedbackInputError(
                "step_index cannot be negative."
            )

        action = _text(_read_value(experience, "action", "action_name"))
        if not action.strip():
            raise LLMPlanRuntimeLearningFeedbackInputError(
                "experience must expose a non-empty action."
            )

        success_value = _read_value(reflection, "success")
        if success_value is None:
            success_value = _read_value(
                experience,
                "success",
            )
        success = None if success_value is None else bool(success_value)

        progress_delta = _optional_float(
            _read_value(reflection, "progress_delta")
        )

        evidence_quality = _clamp(
            _read_value(reflection, "evidence_quality")
        )
        confidence = _clamp(
            _read_value(reflection, "confidence")
        )
        diagnosis = _text(
            _read_value(reflection, "diagnosis"),
            default="unspecified",
        )
        lesson = _text(
            _read_value(reflection, "lesson"),
            default="No lesson recorded.",
        )
        next_step = _text(
            _read_value(reflection, "next_step"),
            default="Reassess before continuing.",
        )
        change_required = bool(
            _read_value(reflection, "change_required")
        )

        if success is False:
            exploration_recommended = True
            continue_recommended = False
            risk_level = "high"
        elif progress_delta is not None and progress_delta <= 0.0:
            exploration_recommended = True
            continue_recommended = not change_required
            risk_level = "medium"
        elif success is True:
            exploration_recommended = False
            continue_recommended = not change_required
            risk_level = "low"
        else:
            exploration_recommended = True
            continue_recommended = not change_required
            risk_level = "medium"

        if change_required:
            risk_level = "high" if success is not True else risk_level

        risk_score = self._RISK_LABELS[risk_level]
        information_gain = _clamp(
            max(progress_delta, 0.0) if progress_delta is not None else 0.0
        )

        merged_metadata = dict(metadata or {})
        merged_metadata.update(
            {
                "feedback_source": "metacognitive_reflection",
                "step_index": step_index,
                "action": action,
                "risk_level": risk_level,
            }
        )

        return LLMPlanRuntimeLearningFeedback(
            step_index=step_index,
            action=action,
            success=success,
            progress_delta=progress_delta,
            evidence_quality=evidence_quality,
            confidence=confidence,
            diagnosis=diagnosis,
            lesson=lesson,
            next_step=next_step,
            change_required=change_required,
            continue_recommended=continue_recommended,
            exploration_recommended=exploration_recommended,
            risk_level=risk_level,
            risk_score=risk_score,
            information_gain=information_gain,
            reflection=reflection,
            episode=experience,
            runtime_evaluation=runtime_evaluation,
            metadata=merged_metadata,
        )


class LLMPlanRuntimeLearningFeedbackBridge:
    """
    Consume the existing Stage 24.10 experience observer contract and turn
    each experience into runtime learning feedback.

    Intended wiring:

        LLMPlanAgentExperienceIntegrator(
            ..., experience_observer=bridge.observe_experience
        )

    The bridge does not replace Agent.record_experience(), metacognition,
    the runtime executor, or the Agent decision lifecycle.
    """

    def __init__(
        self,
        *,
        reflection_provider: ReflectionProvider | None = None,
        feedback_engine: LLMPlanRuntimeLearningFeedbackEngine | None = None,
        feedback_observer: FeedbackObserver | None = None,
        runtime_evaluation_provider: RuntimeEvaluationProvider | None = None,
    ) -> None:
        if reflection_provider is not None and not callable(reflection_provider):
            raise LLMPlanRuntimeLearningFeedbackConfigurationError(
                "reflection_provider must be callable."
            )
        if feedback_engine is not None and not callable(
            getattr(feedback_engine, "build", None)
        ):
            raise LLMPlanRuntimeLearningFeedbackConfigurationError(
                "feedback_engine must expose callable build(...)."
            )
        if feedback_observer is not None and not callable(feedback_observer):
            raise LLMPlanRuntimeLearningFeedbackConfigurationError(
                "feedback_observer must be callable."
            )
        if runtime_evaluation_provider is not None and not callable(
            runtime_evaluation_provider
        ):
            raise LLMPlanRuntimeLearningFeedbackConfigurationError(
                "runtime_evaluation_provider must be callable."
            )

        self.reflection_provider = reflection_provider
        self.feedback_engine = feedback_engine or LLMPlanRuntimeLearningFeedbackEngine()
        self.feedback_observer = feedback_observer
        self.runtime_evaluation_provider = runtime_evaluation_provider
        self._lock = RLock()
        self._feedback_history: list[LLMPlanRuntimeLearningFeedback] = []
        self._feedback_by_step: dict[int, LLMPlanRuntimeLearningFeedback] = {}
        self._last_feedback: LLMPlanRuntimeLearningFeedback | None = None

    @property
    def last_feedback(self) -> LLMPlanRuntimeLearningFeedback | None:
        return self._last_feedback

    @property
    def feedback_history(self) -> tuple[LLMPlanRuntimeLearningFeedback, ...]:
        return tuple(self._feedback_history)

    @property
    def feedback_by_step(self) -> Mapping[int, LLMPlanRuntimeLearningFeedback]:
        return dict(self._feedback_by_step)

    def get_feedback(self, step_index: int) -> LLMPlanRuntimeLearningFeedback | None:
        return self._feedback_by_step.get(int(step_index))

    def observe_experience(
        self,
        episode: Any,
        step_index: int,
        action: str,
        metadata: Mapping[str, Any],
    ) -> LLMPlanRuntimeLearningFeedback:
        with self._lock:
            if self.reflection_provider is None:
                reflection = _read_value(episode, "reflection")
                if reflection is None:
                    raise LLMPlanRuntimeLearningFeedbackInputError(
                        "No reflection_provider is configured and the episode has no reflection."
                    )
            else:
                reflection = self.reflection_provider(episode)

            if reflection is None:
                raise LLMPlanRuntimeLearningFeedbackInputError(
                    f"No metacognitive reflection is available for step {step_index}."
                )

            runtime_evaluation = None
            provisional_metadata = dict(metadata or {})
            provisional_metadata.update(
                {
                    "observer_step_index": int(step_index),
                    "observer_action": action,
                }
            )

            feedback = self.feedback_engine.build(
                episode,
                reflection,
                metadata=provisional_metadata,
            )

            if self.runtime_evaluation_provider is not None:
                runtime_evaluation = self.runtime_evaluation_provider(
                    feedback,
                    episode,
                )
                feedback = self.feedback_engine.build(
                    episode,
                    reflection,
                    runtime_evaluation=runtime_evaluation,
                    metadata=provisional_metadata,
                )

            existing = self._feedback_by_step.get(feedback.step_index)
            if existing is not None:
                return existing

            self._feedback_history.append(feedback)
            self._feedback_by_step[feedback.step_index] = feedback
            self._last_feedback = feedback

            if self.feedback_observer is not None:
                self.feedback_observer(feedback)

            return feedback

    def reset(self) -> None:
        with self._lock:
            self._feedback_history = []
            self._feedback_by_step = {}
            self._last_feedback = None

    def latest_runtime_guidance(self) -> dict[str, Any] | None:
        feedback = self._last_feedback
        if feedback is None:
            return None
        return {
            "step_index": feedback.step_index,
            "action": feedback.action,
            "continue_recommended": feedback.continue_recommended,
            "exploration_recommended": feedback.exploration_recommended,
            "risk_level": feedback.risk_level,
            "risk_score": feedback.risk_score,
            "information_gain": feedback.information_gain,
            "lesson": feedback.lesson,
            "next_step": feedback.next_step,
            "change_required": feedback.change_required,
        }


__all__ = [
    "LLMPlanRuntimeLearningFeedbackError",
    "LLMPlanRuntimeLearningFeedbackConfigurationError",
    "LLMPlanRuntimeLearningFeedbackInputError",
    "LLMPlanRuntimeLearningFeedback",
    "LLMPlanRuntimeLearningFeedbackEngine",
    "LLMPlanRuntimeLearningFeedbackBridge",
]
