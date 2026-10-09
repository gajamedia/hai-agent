from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_EPSILON = 1e-9


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _read_value(source: Any, *names: str) -> Any:
    if source is None:
        return None

    if isinstance(source, dict):
        for name in names:
            if name in source:
                return source[name]
        return None

    for name in names:
        if hasattr(source, name):
            return getattr(source, name)

    return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    name = _read_value(value, "name", "action_name", "description")
    if name is not None:
        return str(name)

    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MetacognitiveReflection:
    """
    Structured meta-level interpretation of one episode.

    The object deliberately separates:

        outcome evidence
            from
        causal interpretation

    so the Agent does not treat a plausible diagnosis as a
    proven cause.
    """

    episode_id: str | None
    goal: str | None
    action: str | None
    success: bool | None

    progress_before: float | None
    progress_after: float | None
    progress_delta: float | None

    diagnosis: str
    lesson: str
    next_step: str

    evidence_quality: float
    confidence: float
    change_required: bool


class MetacognitionEngine:
    """
    Convert an observed episode into a deterministic
    metacognitive reflection.

    This is intentionally model-free for the first layer.
    It does not call an LLM and it does not mutate Agent state.
    """

    def reflect_on_episode(
        self,
        episode: Any,
        *,
        progress_before: float | None = None,
        progress_after: float | None = None,
    ) -> MetacognitiveReflection:
        """
        Analyze an episode and produce a structured reflection.

        Explicit progress arguments take precedence over values
        already present on the episode.
        """

        result = _read_value(episode, "result", "outcome")

        success_value = _read_value(
            result,
            "success",
            "succeeded",
        )

        if success_value is None:
            success_value = _read_value(
                episode,
                "success",
                "succeeded",
            )

        success: bool | None

        if success_value is None:
            success = None
        else:
            success = bool(success_value)

        resolved_episode_id = _read_value(
            episode,
            "episode_id",
            "experience_id",
            "id",
        )

        episode_id = (
            str(resolved_episode_id)
            if resolved_episode_id is not None
            else None
        )

        goal = _as_text(
            _read_value(
                episode,
                "goal",
                "goal_description",
            )
        )

        action = _as_text(
            _read_value(
                episode,
                "action",
                "action_name",
            )
        )

        episode_progress_before = _optional_float(
            _read_value(
                episode,
                "progress_before",
            )
        )

        episode_progress_after = _optional_float(
            _read_value(
                episode,
                "progress_after",
            )
        )

        if progress_before is None:
            progress_before = episode_progress_before
        else:
            progress_before = _optional_float(
                progress_before
            )

        if progress_after is None:
            progress_after = episode_progress_after
        else:
            progress_after = _optional_float(
                progress_after
            )

        progress_before = (
            _clamp(progress_before)
            if progress_before is not None
            else None
        )

        progress_after = (
            _clamp(progress_after)
            if progress_after is not None
            else None
        )

        progress_delta = None

        if (
            progress_before is not None
            and progress_after is not None
        ):
            progress_delta = round(
                progress_after
                - progress_before,
                12,
            )

        has_outcome_evidence = (
            success is not None
        )

        has_progress_evidence = (
            progress_before is not None
            and progress_after is not None
        )

        evidence_quality = 0.0

        if has_outcome_evidence:
            evidence_quality += 0.60

        if has_progress_evidence:
            evidence_quality += 0.40

        evidence_quality = _clamp(
            evidence_quality
        )

        confidence = round(
            _clamp(
                0.20
                + 0.80 * evidence_quality
            ),
            12,
        )

        diagnosis = self._diagnose(
            success=success,
            progress_delta=progress_delta,
            result=result,
        )

        lesson = self._build_lesson(
            success=success,
            progress_delta=progress_delta,
        )

        next_step = self._build_next_step(
            success=success,
            progress_delta=progress_delta,
        )

        change_required = self._requires_change(
            success=success,
            progress_delta=progress_delta,
        )

        return MetacognitiveReflection(
            episode_id=episode_id,
            goal=goal,
            action=action,
            success=success,
            progress_before=progress_before,
            progress_after=progress_after,
            progress_delta=progress_delta,
            diagnosis=diagnosis,
            lesson=lesson,
            next_step=next_step,
            evidence_quality=evidence_quality,
            confidence=confidence,
            change_required=change_required,
        )

    def _diagnose(
        self,
        *,
        success: bool | None,
        progress_delta: float | None,
        result: Any,
    ) -> str:
        """
        Produce an evidence-grounded diagnosis label.

        These are descriptive labels, not claims of proven
        causality.
        """

        if success is None:
            return "outcome_evidence_missing"

        error = _read_value(
            result,
            "error",
            "exception",
            "failure_reason",
        )

        if not success:
            if (
                progress_delta is not None
                and progress_delta > _EPSILON
            ):
                return "failure_with_partial_progress"

            if error is not None:
                return "failure_with_reported_error"

            return "failure_without_measured_progress"

        if (
            progress_delta is not None
            and progress_delta < -_EPSILON
        ):
            return "successful_outcome_but_progress_regressed"

        if (
            progress_delta is not None
            and progress_delta > _EPSILON
        ):
            return "successful_outcome_with_progress"

        return "successful_outcome_without_measured_progress"

    def _build_lesson(
        self,
        *,
        success: bool | None,
        progress_delta: float | None,
    ) -> str:
        if success is None:
            return (
                "The outcome evidence is incomplete; gather clearer "
                "observations before changing the strategy."
            )

        if not success:
            if (
                progress_delta is not None
                and progress_delta > _EPSILON
            ):
                return (
                    "The action failed, but it still produced measurable "
                    "progress; preserve the useful part and revise the failing part."
                )

            return (
                "The action did not produce a successful outcome; revisit "
                "the assumptions behind the action before repeating it unchanged."
            )

        if (
            progress_delta is not None
            and progress_delta < -_EPSILON
        ):
            return (
                "A successful outcome alone is not sufficient evidence "
                "when measured goal progress regresses."
            )

        if (
            progress_delta is not None
            and progress_delta > _EPSILON
        ):
            return (
                "The observed action is supported by both outcome and "
                "measured progress; preserve this pattern while monitoring future evidence."
            )

        return (
            "The outcome was successful, but goal progress was not measured "
            "clearly enough to confirm broader effectiveness."
        )

    def _build_next_step(
        self,
        *,
        success: bool | None,
        progress_delta: float | None,
    ) -> str:
        if success is None:
            return (
                "Collect a clearer observation and outcome signal "
                "before changing strategy."
            )

        if not success:
            if (
                progress_delta is not None
                and progress_delta > _EPSILON
            ):
                return (
                    "Retain the progress-producing component and change "
                    "the component associated with the failure."
                )

            return (
                "Change the action or strategy before the next comparable attempt."
            )

        if (
            progress_delta is not None
            and progress_delta < -_EPSILON
        ):
            return (
                "Re-check the success criterion and investigate why measured "
                "goal progress moved backward."
            )

        if (
            progress_delta is not None
            and progress_delta > _EPSILON
        ):
            return (
                "Reuse the successful pattern and verify that it continues "
                "to improve goal progress."
            )

        return (
            "Reuse cautiously and collect explicit goal-progress evidence on the next cycle."
        )

    def _requires_change(
        self,
        *,
        success: bool | None,
        progress_delta: float | None,
    ) -> bool:
        if success is None:
            return False

        if not success:
            return True

        if (
            progress_delta is not None
            and progress_delta < -_EPSILON
        ):
            return True

        return False
