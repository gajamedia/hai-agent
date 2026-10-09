from __future__ import annotations

from dataclasses import dataclass

from agent.learning.self_uncertainty import (
    SelfUncertaintyResult,
)


# ============================================================
# Constants
# ============================================================


SELF_MODEL_MODE_EXPLORE = "explore"
SELF_MODEL_MODE_EXPLOIT = "exploit"
SELF_MODEL_MODE_NEUTRAL = "neutral"


# ============================================================
# Result Model
# ============================================================


@dataclass(frozen=True)
class SelfModelDecisionInfluence:
    """
    Represents the influence of the agent's self-model
    on exploration vs exploitation.

    This is an advisory signal.

    It does not execute actions and does not directly change
    AgentDecisionMaker behavior.
    """

    preferred_mode: str

    explore_bonus: float
    exploit_bonus: float

    confidence: float
    uncertainty: float

    strength: float

    applied: bool

    reason: str

    @property
    def should_explore(self) -> bool:
        return (
            self.applied
            and self.preferred_mode
            == SELF_MODEL_MODE_EXPLORE
        )

    @property
    def should_exploit(self) -> bool:
        return (
            self.applied
            and self.preferred_mode
            == SELF_MODEL_MODE_EXPLOIT
        )

    @property
    def is_neutral(self) -> bool:
        return (
            self.preferred_mode
            == SELF_MODEL_MODE_NEUTRAL
        )

    @property
    def net_bonus(self) -> float:
        return (
            self.explore_bonus
            - self.exploit_bonus
        )

    @property
    def strongest_bonus(self) -> float:
        return max(
            self.explore_bonus,
            self.exploit_bonus,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "preferred_mode": self.preferred_mode,
            "explore_bonus": self.explore_bonus,
            "exploit_bonus": self.exploit_bonus,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "strength": self.strength,
            "applied": self.applied,
            "reason": self.reason,
            "should_explore": self.should_explore,
            "should_exploit": self.should_exploit,
            "is_neutral": self.is_neutral,
            "net_bonus": self.net_bonus,
            "strongest_bonus": self.strongest_bonus,
        }


# ============================================================
# Integrator
# ============================================================


class SelfModelDecisionInfluenceIntegrator:
    """
    Converts aggregate self-uncertainty into a bounded
    decision influence.

    Exploration influence is driven primarily by uncertainty.

    Exploitation influence is driven by confidence, but is
    reduced when the agent does not have sufficient evidence
    supporting that confidence.

    Formula:

        explore_strength = uncertainty

        exploit_strength =
            confidence * evidence_strength

        explore_bonus =
            max_bonus * explore_strength

        exploit_bonus =
            max_bonus * exploit_strength

    This keeps the influence bounded and independent of the
    environment-level uncertainty used by the existing
    decision system.
    """

    def __init__(
        self,
        *,
        max_bonus: float = 0.25,
        minimum_pressure: float = 0.05,
        minimum_mode_gap: float = 0.05,
    ) -> None:

        self._validate_range(
            max_bonus,
            "max_bonus",
            minimum=0.0,
            maximum=1.0,
        )

        self._validate_range(
            minimum_pressure,
            "minimum_pressure",
            minimum=0.0,
            maximum=1.0,
        )

        self._validate_range(
            minimum_mode_gap,
            "minimum_mode_gap",
            minimum=0.0,
            maximum=1.0,
        )

        if minimum_pressure > max_bonus:
            raise ValueError(
                "minimum_pressure cannot exceed max_bonus."
            )

        self.max_bonus = float(
            max_bonus
        )

        self.minimum_pressure = float(
            minimum_pressure
        )

        self.minimum_mode_gap = float(
            minimum_mode_gap
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def assess(
        self,
        result: SelfUncertaintyResult,
    ) -> SelfModelDecisionInfluence:

        if result is None:
            raise ValueError(
                "result cannot be None."
            )

        if not isinstance(
            result,
            SelfUncertaintyResult,
        ):
            raise ValueError(
                "result must be "
                "SelfUncertaintyResult."
            )

        uncertainty = self._clamp(
            result.uncertainty
        )

        confidence = self._clamp(
            result.confidence
        )

        evidence_strength = self._clamp(
            result.evidence_strength
        )

        # ----------------------------------------------------
        # No self-model evidence
        # ----------------------------------------------------

        if not result.has_evidence:
            explore_bonus = self.max_bonus
            exploit_bonus = 0.0

            return SelfModelDecisionInfluence(
                preferred_mode=(
                    SELF_MODEL_MODE_EXPLORE
                ),
                explore_bonus=round(
                    explore_bonus,
                    6,
                ),
                exploit_bonus=round(
                    exploit_bonus,
                    6,
                ),
                confidence=round(
                    confidence,
                    6,
                ),
                uncertainty=round(
                    uncertainty,
                    6,
                ),
                strength=1.0,
                applied=True,
                reason=(
                    "Self-model has no evidence; "
                    "exploration is preferred."
                ),
            )

        # ----------------------------------------------------
        # Continuous influence
        # ----------------------------------------------------

        explore_strength = uncertainty

        exploit_strength = (
            confidence
            * evidence_strength
        )

        explore_bonus = self._clamp(
            self.max_bonus
            * explore_strength
        )

        exploit_bonus = self._clamp(
            self.max_bonus
            * exploit_strength
        )

        explore_bonus = round(
            explore_bonus,
            6,
        )

        exploit_bonus = round(
            exploit_bonus,
            6,
        )

        gap = abs(
            explore_bonus
            - exploit_bonus
        )

        strongest_bonus = max(
            explore_bonus,
            exploit_bonus,
        )

        # ----------------------------------------------------
        # Determine preferred mode
        # ----------------------------------------------------

        if (
            strongest_bonus
            < self.minimum_pressure
        ):
            preferred_mode = (
                SELF_MODEL_MODE_NEUTRAL
            )
            applied = False
            reason = (
                "Self-model influence is below "
                "the minimum pressure threshold."
            )

        elif (
            gap
            < self.minimum_mode_gap
        ):
            preferred_mode = (
                SELF_MODEL_MODE_NEUTRAL
            )
            applied = False
            reason = (
                "Exploration and exploitation "
                "influence are too close."
            )

        elif (
            explore_bonus
            > exploit_bonus
        ):
            preferred_mode = (
                SELF_MODEL_MODE_EXPLORE
            )
            applied = True
            reason = (
                "Self-uncertainty is stronger "
                "than self-supported confidence."
            )

        else:
            preferred_mode = (
                SELF_MODEL_MODE_EXPLOIT
            )
            applied = True
            reason = (
                "Self-confidence supported by "
                "evidence favors exploitation."
            )

        # ----------------------------------------------------
        # Influence strength
        # ----------------------------------------------------

        strength = (
            strongest_bonus
            / self.max_bonus
            if self.max_bonus > 0.0
            else 0.0
        )

        strength = round(
            self._clamp(strength),
            6,
        )

        return SelfModelDecisionInfluence(
            preferred_mode=preferred_mode,
            explore_bonus=explore_bonus,
            exploit_bonus=exploit_bonus,
            confidence=round(
                confidence,
                6,
            ),
            uncertainty=round(
                uncertainty,
                6,
            ),
            strength=strength,
            applied=applied,
            reason=reason,
        )

    # --------------------------------------------------------
    # Convenience API
    # --------------------------------------------------------

    def influence_for(
        self,
        result: SelfUncertaintyResult,
    ) -> SelfModelDecisionInfluence:

        return self.assess(
            result
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

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

    @staticmethod
    def _validate_range(
        value: float,
        name: str,
        *,
        minimum: float,
        maximum: float,
    ) -> None:

        if not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f"{name} must be numeric."
            )

        if value < minimum or value > maximum:
            raise ValueError(
                f"{name} must be between "
                f"{minimum} and {maximum}."
            )


# ============================================================
# Convenience Builder
# ============================================================


def build_self_model_decision_influence(
    result: SelfUncertaintyResult,
) -> SelfModelDecisionInfluence:

    return (
        SelfModelDecisionInfluenceIntegrator()
        .assess(result)
    )
