"""
Meta-learning decision integration.

Integrates FeedbackDecision with an existing agent decision
without allowing meta-learning to override hard decision
constraints.

The base AgentDecision remains authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.learning.meta_learning_feedback_policy import (
    FeedbackDecision,
)


@dataclass(frozen=True)
class IntegratedDecision:
    """
    Result of integrating a base decision with meta-learning.

    The original decision mode remains authoritative.

    feedback_mode:
        Optional soft preference coming from meta-learning.

    feedback_bias:
        Strength of that preference.

    adaptation_requested:
        Whether meta-learning asks the adaptation layer
        to reconsider behavior.
    """

    base_mode: str
    final_mode: str

    base_reason: str

    feedback_mode: Optional[str] = None
    feedback_bias: float = 0.0

    adaptation_requested: bool = False

    feedback_applied: bool = False

    reason: str = ""

    @property
    def mode_changed(self) -> bool:
        return self.base_mode != self.final_mode

    @property
    def is_explore(self) -> bool:
        return self.final_mode == "explore"

    @property
    def is_exploit(self) -> bool:
        return self.final_mode == "exploit"


class MetaLearningDecisionIntegrator:
    """
    Integrates meta-learning feedback into a decision.

    Safety principle:

        Meta-learning may bias a decision,
        but cannot override the authority of the
        base decision system.

    Therefore the default integration keeps:

        final_mode == base_mode

    and exposes feedback as a soft signal.

    A later policy layer can explicitly decide whether
    a feedback bias is allowed to influence candidate
    ranking, while still preserving hard constraints.
    """

    def integrate(
        self,
        base_mode: str,
        base_reason: str,
        feedback_decision: FeedbackDecision,
    ) -> IntegratedDecision:

        if not base_mode:
            raise ValueError(
                "base_mode cannot be empty."
            )

        if feedback_decision is None:
            raise ValueError(
                "feedback_decision cannot be None."
            )

        normalized_mode = (
            base_mode.lower()
        )

        if normalized_mode not in {
            "explore",
            "exploit",
        }:
            raise ValueError(
                "base_mode must be 'explore' or 'exploit'."
            )

        # -------------------------------------------------
        # No usable feedback.
        # -------------------------------------------------

        if not feedback_decision.apply:

            return IntegratedDecision(
                base_mode=normalized_mode,
                final_mode=normalized_mode,
                base_reason=base_reason,
                feedback_mode=None,
                feedback_bias=0.0,
                adaptation_requested=(
                    feedback_decision.request_adaptation
                ),
                feedback_applied=False,
                reason=(
                    "base decision retained; "
                    "meta-learning feedback is not strong enough"
                ),
            )

        # -------------------------------------------------
        # Feedback is deliberately NOT allowed to
        # override the current decision.
        # -------------------------------------------------

        feedback_mode = (
            feedback_decision.mode_bias
        )

        return IntegratedDecision(
            base_mode=normalized_mode,
            final_mode=normalized_mode,
            base_reason=base_reason,
            feedback_mode=feedback_mode,
            feedback_bias=(
                feedback_decision.bias_strength
            ),
            adaptation_requested=(
                feedback_decision.request_adaptation
            ),
            feedback_applied=True,
            reason=(
                f"base decision retained; "
                f"meta-learning suggests "
                f"{feedback_mode}"
                if feedback_mode is not None
                else (
                    "base decision retained; "
                    "meta-learning requests adaptation"
                )
            ),
        )
