from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from agent.learning.self_awareness_calibration import (
    SelfAwarenessCalibrationResult,
)
from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SelfModelDecisionInfluence,
)


@dataclass(frozen=True)
class CalibratedSelfAwareInfluenceResult:
    """
    Result of applying self-awareness calibration to the
    decision influence that will be used by the next decision
    cycle.
    """

    influence: SelfModelDecisionInfluence

    raw_confidence: float
    calibrated_confidence: float

    raw_uncertainty: float
    calibrated_uncertainty: float

    confidence_delta: float

    raw_explore_bonus: float
    calibrated_explore_bonus: float

    raw_exploit_bonus: float
    calibrated_exploit_bonus: float

    adjusted: bool

    @property
    def explore_bonus_delta(self) -> float:
        return round(
            self.calibrated_explore_bonus
            - self.raw_explore_bonus,
            6,
        )

    @property
    def exploit_bonus_delta(self) -> float:
        return round(
            self.calibrated_exploit_bonus
            - self.raw_exploit_bonus,
            6,
        )

    @property
    def preferred_mode(self) -> str:
        return self.influence.preferred_mode

    @property
    def confidence_shifted_down(self) -> bool:
        return self.confidence_delta < 0.0

    @property
    def confidence_shifted_up(self) -> bool:
        return self.confidence_delta > 0.0

    @property
    def snapshot(self) -> dict[str, Any]:
        return {
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": (
                self.calibrated_confidence
            ),
            "raw_uncertainty": self.raw_uncertainty,
            "calibrated_uncertainty": (
                self.calibrated_uncertainty
            ),
            "confidence_delta": self.confidence_delta,
            "raw_explore_bonus": (
                self.raw_explore_bonus
            ),
            "calibrated_explore_bonus": (
                self.calibrated_explore_bonus
            ),
            "explore_bonus_delta": (
                self.explore_bonus_delta
            ),
            "raw_exploit_bonus": (
                self.raw_exploit_bonus
            ),
            "calibrated_exploit_bonus": (
                self.calibrated_exploit_bonus
            ),
            "exploit_bonus_delta": (
                self.exploit_bonus_delta
            ),
            "adjusted": self.adjusted,
            "preferred_mode": self.preferred_mode,
        }


class CalibratedSelfAwareInfluenceApplier:
    """
    Apply calibration correction to an existing
    SelfModelDecisionInfluence.

    The original self-awareness influence already contains
    bounded explore/exploit bonuses.

    Calibration moves those bonuses according to the
    confidence correction:

        confidence decreases
            → explore bonus increases
            → exploit bonus decreases

        confidence increases
            → explore bonus decreases
            → exploit bonus increases

    The adjustment is bounded by max_bonus.

    This component intentionally does not execute actions and
    does not modify the original influence object.
    """

    def __init__(
        self,
        *,
        max_bonus: float = 0.25,
    ) -> None:

        if isinstance(
            max_bonus,
            bool,
        ) or not isinstance(
            max_bonus,
            (int, float, Decimal),
        ):
            raise TypeError(
                "max_bonus must be numeric."
            )

        normalized = float(
            max_bonus
        )

        if not 0.0 < normalized <= 1.0:
            raise ValueError(
                "max_bonus must be greater than 0 "
                "and at most 1."
            )

        self.max_bonus = normalized

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    # =====================================================
    # Apply
    # =====================================================

    def apply(
        self,
        influence: SelfModelDecisionInfluence,
        calibration: SelfAwarenessCalibrationResult,
    ) -> CalibratedSelfAwareInfluenceResult:
        """
        Apply calibration to an existing self-model
        decision influence.
        """

        if not isinstance(
            influence,
            SelfModelDecisionInfluence,
        ):
            raise TypeError(
                "influence must be a "
                "SelfModelDecisionInfluence."
            )

        if not isinstance(
            calibration,
            SelfAwarenessCalibrationResult,
        ):
            raise TypeError(
                "calibration must be a "
                "SelfAwarenessCalibrationResult."
            )

        raw_confidence = float(
            influence.confidence
        )

        calibrated_confidence = (
            self._clamp(
                calibration.calibrated_confidence
            )
        )

        raw_uncertainty = float(
            influence.uncertainty
        )

        calibrated_uncertainty = (
            self._clamp(
                1.0
                - calibrated_confidence
            )
        )

        confidence_delta = (
            calibrated_confidence
            - raw_confidence
        )

        # Move bounded influence from one side to the other.
        #
        # Negative confidence delta:
        #     more exploration
        #     less exploitation
        #
        # Positive confidence delta:
        #     less exploration
        #     more exploitation

        explore_bonus = (
            float(
                influence.explore_bonus
            )
            - (
                self.max_bonus
                * confidence_delta
            )
        )

        exploit_bonus = (
            float(
                influence.exploit_bonus
            )
            + (
                self.max_bonus
                * confidence_delta
            )
        )

        explore_bonus = max(
            0.0,
            min(
                self.max_bonus,
                explore_bonus,
            ),
        )

        exploit_bonus = max(
            0.0,
            min(
                self.max_bonus,
                exploit_bonus,
            ),
        )

        # Derive the calibrated preferred mode from the
        # corrected bonuses when there is a clear direction.
        if explore_bonus > exploit_bonus:

            preferred_mode = (
                SELF_MODEL_MODE_EXPLORE
            )

        elif exploit_bonus > explore_bonus:

            preferred_mode = (
                SELF_MODEL_MODE_EXPLOIT
            )

        else:

            preferred_mode = (
                influence.preferred_mode
            )

        adjusted = (
            abs(
                confidence_delta
            )
            > 1e-12
            or abs(
                explore_bonus
                - float(
                    influence.explore_bonus
                )
            )
            > 1e-12
            or abs(
                exploit_bonus
                - float(
                    influence.exploit_bonus
                )
            )
            > 1e-12
        )

        reason = (
            influence.reason
        )

        if adjusted:

            reason = (
                f"{influence.reason}; "
                f"calibrated confidence "
                f"{raw_confidence:.6f}"
                f"->{calibrated_confidence:.6f}"
            )

        calibrated_influence = (
            SelfModelDecisionInfluence(
                preferred_mode=(
                    preferred_mode
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
                    calibrated_confidence,
                    6,
                ),
                uncertainty=round(
                    calibrated_uncertainty,
                    6,
                ),
                strength=(
                    influence.strength
                ),
                applied=(
                    influence.applied
                ),
                reason=reason,
            )
        )

        return (
            CalibratedSelfAwareInfluenceResult(
                influence=(
                    calibrated_influence
                ),
                raw_confidence=round(
                    raw_confidence,
                    6,
                ),
                calibrated_confidence=round(
                    calibrated_confidence,
                    6,
                ),
                raw_uncertainty=round(
                    raw_uncertainty,
                    6,
                ),
                calibrated_uncertainty=round(
                    calibrated_uncertainty,
                    6,
                ),
                confidence_delta=round(
                    confidence_delta,
                    6,
                ),
                raw_explore_bonus=round(
                    float(
                        influence.explore_bonus
                    ),
                    6,
                ),
                calibrated_explore_bonus=round(
                    explore_bonus,
                    6,
                ),
                raw_exploit_bonus=round(
                    float(
                        influence.exploit_bonus
                    ),
                    6,
                ),
                calibrated_exploit_bonus=round(
                    exploit_bonus,
                    6,
                ),
                adjusted=adjusted,
            )
        )
