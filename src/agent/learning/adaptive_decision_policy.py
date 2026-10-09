"""
Adaptive decision policy.

Combines the agent's base decision signals with a bounded
meta-learning feedback bias.

Meta-learning can influence mode preference, but cannot
override hard constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.learning.meta_learning_feedback_policy import (
    FeedbackDecision,
)


@dataclass(frozen=True)
class AdaptiveModeScore:
    """
    Score for one decision mode.
    """

    mode: str
    base_score: float
    feedback_bonus: float
    final_score: float

    @property
    def is_explore(self) -> bool:
        return self.mode == "explore"

    @property
    def is_exploit(self) -> bool:
        return self.mode == "exploit"


@dataclass(frozen=True)
class AdaptiveDecision:
    """
    Final adaptive decision produced by the policy.
    """

    mode: str
    explore_score: float
    exploit_score: float

    feedback_applied: bool = False
    adaptation_requested: bool = False

    reason: str = ""

    @property
    def is_explore(self) -> bool:
        return self.mode == "explore"

    @property
    def is_exploit(self) -> bool:
        return self.mode == "exploit"


class AdaptiveDecisionPolicy:
    """
    Adaptive EXPLORE/EXPLOIT policy.

    Base scoring:

        exploration_base =
            exploration_weight * information_gain
            + uncertainty_weight * uncertainty
            - risk_weight * risk

        exploitation_base =
            exploitation_weight * (1 - uncertainty)
            + exploitation_stability_weight * (1 - risk)

    Meta-learning feedback contributes only a bounded bonus.

    Hard constraints:

        explore_allowed=False
            -> EXPLORE score becomes -infinity

        exploit_allowed=False
            -> EXPLOIT score becomes -infinity

    Thus feedback cannot usefully select a forbidden mode.
    """

    def __init__(
        self,
        exploration_weight: float = 0.50,
        uncertainty_weight: float = 0.30,
        risk_weight: float = 0.20,
        exploitation_weight: float = 0.50,
        exploitation_stability_weight: float = 0.50,
        maximum_feedback_bonus: float = 0.25,
    ) -> None:

        weights = {
            "exploration_weight": exploration_weight,
            "uncertainty_weight": uncertainty_weight,
            "risk_weight": risk_weight,
            "exploitation_weight": exploitation_weight,
            "exploitation_stability_weight": (
                exploitation_stability_weight
            ),
            "maximum_feedback_bonus": (
                maximum_feedback_bonus
            ),
        }

        for name, value in weights.items():
            if value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        if maximum_feedback_bonus > 1.0:
            raise ValueError(
                "maximum_feedback_bonus must be <= 1."
            )

        self.exploration_weight = exploration_weight
        self.uncertainty_weight = uncertainty_weight
        self.risk_weight = risk_weight

        self.exploitation_weight = exploitation_weight
        self.exploitation_stability_weight = (
            exploitation_stability_weight
        )

        self.maximum_feedback_bonus = (
            maximum_feedback_bonus
        )

    # =====================================================
    # Public API
    # =====================================================

    def evaluate(
        self,
        uncertainty: float,
        information_gain: float,
        risk: float,
        feedback_decision: Optional[
            FeedbackDecision
        ] = None,
        explore_allowed: bool = True,
        exploit_allowed: bool = True,
    ) -> AdaptiveDecision:

        self._validate_signal(
            uncertainty,
            "uncertainty",
        )

        self._validate_signal(
            information_gain,
            "information_gain",
        )

        self._validate_signal(
            risk,
            "risk",
        )

        if not explore_allowed and not exploit_allowed:
            raise ValueError(
                "at least one decision mode must be allowed."
            )

        explore_base = (
            self.exploration_weight
            * information_gain
            + self.uncertainty_weight
            * uncertainty
            - self.risk_weight
            * risk
        )

        exploit_base = (
            self.exploitation_weight
            * (1.0 - uncertainty)
            + self.exploitation_stability_weight
            * (1.0 - risk)
        )

        explore_bonus = 0.0
        exploit_bonus = 0.0

        feedback_applied = False
        adaptation_requested = False

        if feedback_decision is not None:
            adaptation_requested = (
                feedback_decision.request_adaptation
            )

            if feedback_decision.apply:
                feedback_applied = True

                feedback_bonus = min(
                    self.maximum_feedback_bonus,
                    max(
                        0.0,
                        feedback_decision.bias_strength,
                    ),
                )

                if (
                    feedback_decision.mode_bias
                    == "explore"
                ):
                    explore_bonus = feedback_bonus

                elif (
                    feedback_decision.mode_bias
                    == "exploit"
                ):
                    exploit_bonus = feedback_bonus

        # -------------------------------------------------
        # Hard feasibility constraints.
        # -------------------------------------------------

        if not explore_allowed:
            explore_score = float("-inf")
        else:
            explore_score = (
                explore_base
                + explore_bonus
            )

        if not exploit_allowed:
            exploit_score = float("-inf")
        else:
            exploit_score = (
                exploit_base
                + exploit_bonus
            )

        # -------------------------------------------------
        # Deterministic tie-breaker:
        # EXPLOIT wins ties.
        #
        # This prevents adaptation noise from causing
        # unnecessary exploration when evidence is equal.
        # -------------------------------------------------

        if explore_score > exploit_score:
            mode = "explore"
        else:
            mode = "exploit"

        reason = self._build_reason(
            mode=mode,
            feedback_decision=feedback_decision,
            explore_score=explore_score,
            exploit_score=exploit_score,
        )

        return AdaptiveDecision(
            mode=mode,
            explore_score=round(
                explore_score,
                6,
            ),
            exploit_score=round(
                exploit_score,
                6,
            ),
            feedback_applied=feedback_applied,
            adaptation_requested=adaptation_requested,
            reason=reason,
        )

    def score_modes(
        self,
        uncertainty: float,
        information_gain: float,
        risk: float,
        feedback_decision: Optional[
            FeedbackDecision
        ] = None,
        explore_allowed: bool = True,
        exploit_allowed: bool = True,
    ) -> tuple[
        AdaptiveModeScore,
        AdaptiveModeScore,
    ]:

        decision = self.evaluate(
            uncertainty=uncertainty,
            information_gain=information_gain,
            risk=risk,
            feedback_decision=feedback_decision,
            explore_allowed=explore_allowed,
            exploit_allowed=exploit_allowed,
        )

        explore_bonus = 0.0
        exploit_bonus = 0.0

        if (
            feedback_decision is not None
            and feedback_decision.apply
        ):
            bonus = min(
                self.maximum_feedback_bonus,
                max(
                    0.0,
                    feedback_decision.bias_strength,
                ),
            )

            if feedback_decision.mode_bias == "explore":
                explore_bonus = bonus

            elif feedback_decision.mode_bias == "exploit":
                exploit_bonus = bonus

        explore_base = (
            decision.explore_score
            - explore_bonus
            if decision.explore_score
            != float("-inf")
            else float("-inf")
        )

        exploit_base = (
            decision.exploit_score
            - exploit_bonus
            if decision.exploit_score
            != float("-inf")
            else float("-inf")
        )

        return (
            AdaptiveModeScore(
                mode="explore",
                base_score=round(
                    explore_base,
                    6,
                ),
                feedback_bonus=round(
                    explore_bonus,
                    6,
                ),
                final_score=decision.explore_score,
            ),
            AdaptiveModeScore(
                mode="exploit",
                base_score=round(
                    exploit_base,
                    6,
                ),
                feedback_bonus=round(
                    exploit_bonus,
                    6,
                ),
                final_score=decision.exploit_score,
            ),
        )

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _validate_signal(
        value: float,
        name: str,
    ) -> None:

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    # =====================================================
    # Explanation
    # =====================================================

    @staticmethod
    def _build_reason(
        mode: str,
        feedback_decision: Optional[
            FeedbackDecision
        ],
        explore_score: float,
        exploit_score: float,
    ) -> str:

        if (
            feedback_decision is not None
            and feedback_decision.apply
            and feedback_decision.mode_bias
        ):
            return (
                f"{mode} selected after combining base "
                f"decision signals with bounded "
                f"{feedback_decision.mode_bias} meta-learning bias"
            )

        return (
            f"{mode} selected from base decision signals"
        )
