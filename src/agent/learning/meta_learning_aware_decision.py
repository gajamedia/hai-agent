"""
Meta-learning aware decision integration.

Provides a backward-compatible integration layer between the
existing AgentDecisionMaker and adaptive meta-learning signals.

The existing decision maker remains the source of the base decision.
Meta-learning may influence the final mode only through the
AdaptiveDecisionPolicy.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

from agent.learning.adaptive_decision_policy import (
    AdaptiveDecisionPolicy,
)
from agent.learning.meta_learning_feedback_policy import (
    FeedbackDecision,
)


@dataclass(frozen=True)
class MetaLearningAwareDecision:
    """
    Decision returned by the meta-learning aware layer.

    base_decision contains the original AgentDecision produced by
    AgentDecisionMaker. This preserves compatibility with downstream
    components such as UnifiedDecisionExecutor.
    """

    mode: str
    base_mode: str

    uncertainty: float
    information_gain: float
    risk: float

    base_reason: str
    reason: str

    # Original decision object produced by the base decision maker.
    #
    # Kept as object to avoid coupling this learning layer directly
    # to agent.cognition.agent_decision.AgentDecision.
    base_decision: object = None

    feedback_applied: bool = False
    adaptation_requested: bool = False

    @property
    def is_explore(self) -> bool:
        return self.mode == "explore"

    @property
    def is_exploit(self) -> bool:
        return self.mode == "exploit"

    @property
    def mode_changed(self) -> bool:
        return self.mode != self.base_mode


class MetaLearningAwareDecisionMaker:
    """
    Integrate an existing base decision with meta-learning.

    The base decision maker remains authoritative for the initial
    decision. AdaptiveDecisionPolicy then evaluates the same
    decision signals together with optional meta-learning feedback.

    Compatibility note:

    Existing decision makers may call the information-gain argument
    either:

        information_gain

    or:

        expected_information_gain

    The adapter detects the supported parameter name instead of
    assuming one concrete implementation.
    """

    def __init__(
        self,
        base_decision_maker,
        adaptive_policy: Optional[
            AdaptiveDecisionPolicy
        ] = None,
    ) -> None:

        if base_decision_maker is None:
            raise ValueError(
                "base_decision_maker cannot be None."
            )

        self.base_decision_maker = (
            base_decision_maker
        )

        self.adaptive_policy = (
            adaptive_policy
            or AdaptiveDecisionPolicy()
        )

    def decide(
        self,
        uncertainty: float,
        information_gain: float,
        risk: float,
        feedback_decision: Optional[
            FeedbackDecision
        ] = None,
        explore_allowed: bool = True,
        exploit_allowed: bool = True,
    ) -> MetaLearningAwareDecision:

        # -------------------------------------------------
        # First obtain the original AgentDecision.
        # -------------------------------------------------

        base_decision = (
            self._call_base_decision_maker(
                uncertainty=uncertainty,
                information_gain=information_gain,
                risk=risk,
            )
        )

        # -------------------------------------------------
        # Extract the original mode/reason.
        # -------------------------------------------------

        base_mode = self._extract_mode(
            base_decision
        )

        base_reason = self._extract_reason(
            base_decision
        )

        # -------------------------------------------------
        # Apply adaptive meta-learning policy.
        # -------------------------------------------------

        adaptive_decision = (
            self.adaptive_policy.evaluate(
                uncertainty=uncertainty,
                information_gain=information_gain,
                risk=risk,
                feedback_decision=feedback_decision,
                explore_allowed=explore_allowed,
                exploit_allowed=exploit_allowed,
            )
        )

        # -------------------------------------------------
        # Keep both:
        #
        #   adaptive mode
        #   original AgentDecision
        #
        # The original decision object is required by the
        # downstream execution system.
        # -------------------------------------------------

        return MetaLearningAwareDecision(
            mode=adaptive_decision.mode,
            base_mode=base_mode,
            uncertainty=uncertainty,
            information_gain=information_gain,
            risk=risk,
            base_reason=base_reason,
            reason=adaptive_decision.reason,
            base_decision=base_decision,
            feedback_applied=(
                adaptive_decision.feedback_applied
            ),
            adaptation_requested=(
                adaptive_decision.adaptation_requested
            ),
        )

    # =====================================================
    # Base decision compatibility
    # =====================================================

    def _call_base_decision_maker(
        self,
        uncertainty: float,
        information_gain: float,
        risk: float,
    ):
        """
        Call the base decision maker using the API it exposes.

        Current AgentDecisionMaker uses:

            expected_information_gain

        Some test doubles / older implementations may use:

            information_gain
        """

        decide = getattr(
            self.base_decision_maker,
            "decide",
            None,
        )

        if decide is None:
            raise AttributeError(
                "base_decision_maker must provide a "
                "'decide' method."
            )

        parameters = inspect.signature(
            decide
        ).parameters

        if (
            "expected_information_gain"
            in parameters
        ):
            return decide(
                uncertainty=uncertainty,
                expected_information_gain=(
                    information_gain
                ),
                risk=risk,
            )

        if "information_gain" in parameters:
            return decide(
                uncertainty=uncertainty,
                information_gain=(
                    information_gain
                ),
                risk=risk,
            )

        raise TypeError(
            "base_decision_maker.decide() must "
            "accept either 'information_gain' or "
            "'expected_information_gain'."
        )

    # =====================================================
    # Decision extraction
    # =====================================================

    @staticmethod
    def _extract_mode(
        decision,
    ) -> str:

        mode = getattr(
            decision,
            "mode",
            None,
        )

        # AgentDecision.mode is a DecisionMode enum.
        if hasattr(
            mode,
            "value",
        ):
            mode = mode.value

        if not isinstance(
            mode,
            str,
        ):
            raise ValueError(
                "base decision must expose a valid mode."
            )

        normalized = mode.lower()

        if normalized not in {
            "explore",
            "exploit",
        }:
            raise ValueError(
                "base decision mode must be "
                "'explore' or 'exploit'."
            )

        return normalized

    @staticmethod
    def _extract_reason(
        decision,
    ) -> str:

        reason = getattr(
            decision,
            "reason",
            "",
        )

        return str(reason)
