"""
Meta-learning feedback policy.

Converts MetaLearningFeedback into a bounded policy decision.

The policy does not execute actions and does not directly modify
AgentDecisionMaker. It only determines whether the feedback is
strong enough to influence the next decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.learning.meta_learning_feedback import (
    MetaLearningFeedback,
)


@dataclass(frozen=True)
class FeedbackDecision:
    """
    Decision produced by the feedback policy.

    mode_bias:
        Optional preferred mode: "explore" or "exploit".

    bias_strength:
        Bounded influence in the range 0..1.

    apply:
        Whether the feedback is strong enough to be consumed.

    request_adaptation:
        Whether adaptation should be considered.

    reason:
        Human-readable explanation of the policy decision.
    """

    mode_bias: Optional[str] = None
    bias_strength: float = 0.0
    apply: bool = False
    request_adaptation: bool = False
    reason: str = ""

    @property
    def has_mode_bias(self) -> bool:
        return self.mode_bias is not None

    @property
    def is_neutral(self) -> bool:
        return not self.apply

    @property
    def should_explore(self) -> bool:
        return (
            self.apply
            and self.mode_bias == "explore"
        )

    @property
    def should_exploit(self) -> bool:
        return (
            self.apply
            and self.mode_bias == "exploit"
        )


class MetaLearningFeedbackPolicy:
    """
    Decide whether MetaLearningFeedback should influence behavior.

    Default principles:

    1. Weak evidence must not override the decision system.
    2. Feedback needs sufficient confidence.
    3. A mode bias needs a meaningful pressure signal.
    4. Adaptation pressure is exposed separately.
    5. Final execution remains the responsibility of the
       decision/execution layers.
    """

    def __init__(
        self,
        minimum_confidence: float = 0.50,
        minimum_pressure: float = 0.10,
        maximum_bias: float = 0.75,
        minimum_adaptation_pressure: float = 0.50,
    ) -> None:

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )

        if not 0.0 <= minimum_pressure <= 1.0:
            raise ValueError(
                "minimum_pressure must be between 0 and 1."
            )

        if not 0.0 <= maximum_bias <= 1.0:
            raise ValueError(
                "maximum_bias must be between 0 and 1."
            )

        if not 0.0 <= minimum_adaptation_pressure <= 1.0:
            raise ValueError(
                "minimum_adaptation_pressure must be "
                "between 0 and 1."
            )

        self.minimum_confidence = (
            minimum_confidence
        )

        self.minimum_pressure = (
            minimum_pressure
        )

        self.maximum_bias = (
            maximum_bias
        )

        self.minimum_adaptation_pressure = (
            minimum_adaptation_pressure
        )

    # =====================================================
    # Public API
    # =====================================================

    def evaluate(
        self,
        feedback: MetaLearningFeedback,
    ) -> FeedbackDecision:

        if feedback is None:
            raise ValueError(
                "feedback cannot be None."
            )

        # -------------------------------------------------
        # Confidence gate
        # -------------------------------------------------

        if (
            feedback.confidence
            < self.minimum_confidence
        ):
            return FeedbackDecision(
                reason=(
                    "feedback confidence is below "
                    "the minimum threshold"
                ),
            )

        # -------------------------------------------------
        # Adaptation signal
        # -------------------------------------------------

        request_adaptation = (
            feedback.adaptation_pressure
            >= self.minimum_adaptation_pressure
        )

        # -------------------------------------------------
        # No preferred mode
        # -------------------------------------------------

        if feedback.preferred_mode is None:
            if request_adaptation:
                return FeedbackDecision(
                    apply=True,
                    request_adaptation=True,
                    reason=(
                        "adaptation pressure is strong "
                        "but no mode has sufficient preference"
                    ),
                )

            return FeedbackDecision(
                apply=False,
                request_adaptation=False,
                reason=(
                    "feedback does not provide a strong "
                    "behavioral preference"
                ),
            )

        # -------------------------------------------------
        # Determine mode pressure
        # -------------------------------------------------

        if feedback.preferred_mode == "explore":
            pressure = feedback.exploration_pressure

        elif feedback.preferred_mode == "exploit":
            pressure = feedback.exploitation_pressure

        else:
            return FeedbackDecision(
                reason=(
                    "unsupported preferred mode"
                ),
            )

        # -------------------------------------------------
        # Pressure gate
        # -------------------------------------------------

        if pressure < self.minimum_pressure:

            return FeedbackDecision(
                apply=(
                    request_adaptation
                ),
                request_adaptation=(
                    request_adaptation
                ),
                reason=(
                    "mode preference is too weak "
                    "to influence behavior"
                ),
            )

        # -------------------------------------------------
        # Convert pressure + confidence into bounded bias.
        # -------------------------------------------------

        bias_strength = min(
            self.maximum_bias,
            pressure * feedback.confidence,
        )

        # -------------------------------------------------
        # Apply feedback.
        # -------------------------------------------------

        return FeedbackDecision(
            mode_bias=feedback.preferred_mode,
            bias_strength=round(
                bias_strength,
                6,
            ),
            apply=True,
            request_adaptation=(
                request_adaptation
            ),
            reason=(
                f"apply {feedback.preferred_mode} "
                f"preference from meta-learning evidence"
            ),
        )
