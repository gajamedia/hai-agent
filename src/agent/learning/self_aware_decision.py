from __future__ import annotations

from dataclasses import dataclass, replace

from agent.cognition.agent_decision import (
    AgentDecision,
    DecisionMode,
)

from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SELF_MODEL_MODE_NEUTRAL,
    SelfModelDecisionInfluence,
)


# ============================================================
# Result
# ============================================================


@dataclass(frozen=True)
class SelfAwareDecisionResult:
    """
    Result of combining the existing AgentDecision with
    self-model decision influence.

    The original AgentDecision is never mutated.
    """

    base_decision: AgentDecision
    influence: SelfModelDecisionInfluence

    base_explore_score: float
    base_exploit_score: float

    final_explore_score: float
    final_exploit_score: float

    selected_mode: DecisionMode

    applied: bool

    reason: str

    @property
    def mode_changed(self) -> bool:
        return (
            self.selected_mode
            != self.base_decision.mode
        )

    @property
    def self_influence_active(self) -> bool:
        return self.applied

    @property
    def final_margin(self) -> float:
        return (
            self.final_explore_score
            - self.final_exploit_score
        )

    @property
    def base_margin(self) -> float:
        return (
            self.base_explore_score
            - self.base_exploit_score
        )

    @property
    def exploration_selected(self) -> bool:
        return (
            self.selected_mode
            == DecisionMode.EXPLORE
        )

    @property
    def exploitation_selected(self) -> bool:
        return (
            self.selected_mode
            == DecisionMode.EXPLOIT
        )

    @property
    def decision(self) -> AgentDecision:
        """
        Return a new AgentDecision containing the selected
        mode and explanation.

        All original decision measurements remain unchanged.
        """

        if not self.mode_changed:
            return self.base_decision

        return replace(
            self.base_decision,
            mode=self.selected_mode,
            reason=self.reason,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "base_decision": {
                "mode": str(
                    self.base_decision.mode
                ),
                "reason": (
                    self.base_decision.reason
                ),
                "uncertainty": (
                    self.base_decision.uncertainty
                ),
                "expected_information_gain": (
                    self.base_decision
                    .expected_information_gain
                ),
                "risk": (
                    self.base_decision.risk
                ),
            },
            "influence": (
                self.influence.snapshot()
            ),
            "base_explore_score": (
                self.base_explore_score
            ),
            "base_exploit_score": (
                self.base_exploit_score
            ),
            "final_explore_score": (
                self.final_explore_score
            ),
            "final_exploit_score": (
                self.final_exploit_score
            ),
            "selected_mode": str(
                self.selected_mode
            ),
            "applied": self.applied,
            "reason": self.reason,
            "mode_changed": self.mode_changed,
            "self_influence_active": (
                self.self_influence_active
            ),
            "final_margin": self.final_margin,
            "base_margin": self.base_margin,
            "exploration_selected": (
                self.exploration_selected
            ),
            "exploitation_selected": (
                self.exploitation_selected
            ),
        }


# ============================================================
# Arbiter
# ============================================================


class SelfAwareDecisionArbiter:
    """
    Combines an existing AgentDecision with the agent's
    self-model decision influence.

    Base decision formula:

        explore =
            0.50 * information_gain
            + 0.30 * uncertainty
            - 0.20 * risk

        exploit =
            0.50 * (1 - uncertainty)
            + 0.50 * (1 - risk)

    Self-model influence:

        final_explore =
            base_explore + explore_bonus

        final_exploit =
            base_exploit + exploit_bonus

    Tie-break:

        EXPLOIT

    The arbiter does not execute actions.
    """

    def __init__(
        self,
        *,
        enable_self_influence: bool = True,
    ) -> None:

        self.enable_self_influence = bool(
            enable_self_influence
        )

    # ========================================================
    # Public API
    # ========================================================

    def evaluate(
        self,
        base_decision: AgentDecision,
        influence: SelfModelDecisionInfluence,
    ) -> SelfAwareDecisionResult:

        if base_decision is None:
            raise ValueError(
                "base_decision cannot be None."
            )

        if not isinstance(
            base_decision,
            AgentDecision,
        ):
            raise ValueError(
                "base_decision must be AgentDecision."
            )

        if influence is None:
            raise ValueError(
                "influence cannot be None."
            )

        if not isinstance(
            influence,
            SelfModelDecisionInfluence,
        ):
            raise ValueError(
                "influence must be "
                "SelfModelDecisionInfluence."
            )

        uncertainty = self._clamp(
            base_decision.uncertainty
        )

        information_gain = self._clamp(
            base_decision.expected_information_gain
        )

        risk = self._clamp(
            base_decision.risk
        )

        # ----------------------------------------------------
        # Base scores
        # ----------------------------------------------------

        base_explore_score = self._clamp(
            (
                0.50 * information_gain
                + 0.30 * uncertainty
                - 0.20 * risk
            )
        )

        base_exploit_score = self._clamp(
            (
                0.50 * (1.0 - uncertainty)
                + 0.50 * (1.0 - risk)
            )
        )

        # ----------------------------------------------------
        # Self-model influence
        # ----------------------------------------------------

        apply_influence = (
            self.enable_self_influence
            and influence.applied
            and not influence.is_neutral
        )

        if apply_influence:
            explore_bonus = (
                influence.explore_bonus
            )

            exploit_bonus = (
                influence.exploit_bonus
            )

        else:
            explore_bonus = 0.0
            exploit_bonus = 0.0

        final_explore_score = (
            base_explore_score
            + explore_bonus
        )

        final_exploit_score = (
            base_exploit_score
            + exploit_bonus
        )

        # ----------------------------------------------------
        # Determine final mode
        # ----------------------------------------------------

        if (
            final_explore_score
            > final_exploit_score
        ):
            selected_mode = (
                DecisionMode.EXPLORE
            )

        else:
            # Explicit tie-break to exploit.
            selected_mode = (
                DecisionMode.EXPLOIT
            )

        # ----------------------------------------------------
        # Reason
        # ----------------------------------------------------

        if not apply_influence:

            reason = (
                "Self-model influence was not applied; "
                "the base decision is preserved."
            )

        elif selected_mode == base_decision.mode:

            reason = (
                "Self-model influence was applied, "
                "but it did not change the selected mode."
            )

        elif (
            selected_mode
            == DecisionMode.EXPLORE
        ):

            reason = (
                "Self-model influence shifted the "
                "decision toward exploration."
            )

        else:

            reason = (
                "Self-model influence shifted the "
                "decision toward exploitation."
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return SelfAwareDecisionResult(
            base_decision=base_decision,
            influence=influence,
            base_explore_score=round(
                base_explore_score,
                6,
            ),
            base_exploit_score=round(
                base_exploit_score,
                6,
            ),
            final_explore_score=round(
                final_explore_score,
                6,
            ),
            final_exploit_score=round(
                final_exploit_score,
                6,
            ),
            selected_mode=selected_mode,
            applied=apply_influence,
            reason=reason,
        )

    def arbitrate(
        self,
        base_decision: AgentDecision,
        influence: SelfModelDecisionInfluence,
    ) -> SelfAwareDecisionResult:

        return self.evaluate(
            base_decision,
            influence,
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )


# ============================================================
# Convenience Builder
# ============================================================


def build_self_aware_decision_arbiter(
    *,
    enable_self_influence: bool = True,
) -> SelfAwareDecisionArbiter:

    return SelfAwareDecisionArbiter(
        enable_self_influence=enable_self_influence,
    )
