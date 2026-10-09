"""
Adaptation threshold and stability.

Provides a bounded mechanism for deciding whether a new
behavioral bias should replace the previous bias.

This layer is intentionally independent from the final
decision policy.

Responsibilities:

    - ignore insignificant changes
    - limit the maximum adaptation step
    - apply hysteresis around zero
    - expose whether adaptation actually occurred

Non-responsibilities:

    - choosing EXPLORE or EXPLOIT
    - executing actions
    - changing MetaLearningProfile
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityResult:
    """
    Result of applying stability constraints.
    """

    previous_bias: float
    proposed_bias: float
    applied_bias: float

    change: float

    adapted: bool
    held: bool
    clamped: bool

    reason: str

    @property
    def absolute_change(self) -> float:
        return abs(self.change)

    @property
    def is_stable(self) -> bool:
        return not self.adapted

    @property
    def direction(self) -> str:
        if self.applied_bias > 0.0:
            return "positive"

        if self.applied_bias < 0.0:
            return "negative"

        return "neutral"


class AdaptationStabilityController:
    """
    Applies threshold and stability constraints to a proposed
    behavioral bias.

    The controller does not know whether the bias represents
    exploration or exploitation.

    Formula:

        delta = proposed - previous

        if abs(delta) < minimum_change:
            hold previous value

        otherwise:
            delta is limited to maximum_adaptation_step

        if the result is near zero and hysteresis prevents
        crossing, retain the previous direction.
    """

    def __init__(
        self,
        minimum_change: float = 0.02,
        hysteresis: float = 0.05,
        maximum_adaptation_step: float = 0.10,
        maximum_bias: float = 0.25,
    ) -> None:

        if not 0.0 <= minimum_change <= 1.0:
            raise ValueError(
                "minimum_change must be between 0 and 1."
            )

        if not 0.0 <= hysteresis <= 1.0:
            raise ValueError(
                "hysteresis must be between 0 and 1."
            )

        if not 0.0 < maximum_adaptation_step <= 1.0:
            raise ValueError(
                "maximum_adaptation_step must be > 0 and <= 1."
            )

        if not 0.0 <= maximum_bias <= 1.0:
            raise ValueError(
                "maximum_bias must be between 0 and 1."
            )

        if hysteresis > maximum_bias:
            raise ValueError(
                "hysteresis cannot exceed maximum_bias."
            )

        self.minimum_change = (
            minimum_change
        )

        self.hysteresis = (
            hysteresis
        )

        self.maximum_adaptation_step = (
            maximum_adaptation_step
        )

        self.maximum_bias = (
            maximum_bias
        )

    def update(
        self,
        previous_bias: float,
        proposed_bias: float,
    ) -> StabilityResult:
        """
        Apply stability constraints to a proposed bias.

        Both values are normalized to the configured bounds.
        """

        previous = self._clamp_bias(
            previous_bias
        )

        proposed = self._clamp_bias(
            proposed_bias
        )

        raw_change = (
            proposed
            - previous
        )

        # ---------------------------------------------
        # Tiny changes are ignored.
        # ---------------------------------------------

        if (
            abs(raw_change)
            < self.minimum_change
        ):

            return StabilityResult(
                previous_bias=previous,
                proposed_bias=proposed,
                applied_bias=previous,
                change=0.0,
                adapted=False,
                held=True,
                clamped=False,
                reason=(
                    "Proposed adaptation is below "
                    "the minimum change threshold."
                ),
            )

        # ---------------------------------------------
        # Hysteresis around zero.
        #
        # A directional change must overcome the
        # hysteresis band before it can flip.
        # ---------------------------------------------

        hysteresis_result = (
            self._apply_hysteresis(
                previous=previous,
                proposed=proposed,
            )
        )

        if hysteresis_result:
            return StabilityResult(
                previous_bias=previous,
                proposed_bias=proposed,
                applied_bias=previous,
                change=0.0,
                adapted=False,
                held=True,
                clamped=False,
                reason=(
                    "Hysteresis held the previous "
                    "behavioral direction."
                ),
            )

        # ---------------------------------------------
        # Limit one-step adaptation.
        # ---------------------------------------------

        bounded_delta = self._limit_step(
            raw_change
        )

        applied = (
            previous
            + bounded_delta
        )

        applied = self._clamp_bias(
            applied
        )

        actual_change = (
            applied
            - previous
        )

        clamped = (
            abs(
                actual_change
                - raw_change
            )
            > 1e-12
        )

        return StabilityResult(
            previous_bias=previous,
            proposed_bias=proposed,
            applied_bias=round(
                applied,
                6,
            ),
            change=round(
                actual_change,
                6,
            ),
            adapted=(
                abs(actual_change) > 0.0
            ),
            held=False,
            clamped=clamped,
            reason=(
                "Behavioral adaptation applied "
                "within stability limits."
                if not clamped
                else
                "Behavioral adaptation was applied "
                "with a bounded adaptation step."
            ),
        )

    def normalize(
        self,
        bias: float,
    ) -> float:
        """
        Clamp an arbitrary bias to the configured range.
        """

        return self._clamp_bias(
            bias
        )

    def _apply_hysteresis(
        self,
        previous: float,
        proposed: float,
    ) -> bool:
        """
        Return True when hysteresis should hold the
        previous behavioral state.

        Hysteresis protects against two types of small
        behavioral changes:

        1. Small changes while remaining in the same
           direction.

        2. Small crossings around zero.

        A change is allowed only when it is large enough
        to overcome the hysteresis band.
        """

        # No previous directional state.
        if previous == 0.0:
            return False

        delta = round(
            proposed - previous,
            12,
        )


        # -------------------------------------------------
        # Same-direction small adjustment.
        #
        # Example:
        #   +0.20 -> +0.18
        #
        # The direction did not change and the change is
        # too small, so keep +0.20.
        # -------------------------------------------------

        same_direction = (
            (
                previous > 0.0
                and proposed > 0.0
            )
            or
            (
                previous < 0.0
                and proposed < 0.0
            )
        )

        if (
            same_direction
            and abs(delta) < self.hysteresis
        ):
            return True

        # -------------------------------------------------
        # Small crossing around zero.
        #
        # Example:
        #   +0.10 -> -0.03
        #
        # The new negative direction is too weak to justify
        # flipping the existing positive behavioral state.
        # -------------------------------------------------

        crossing_zero = (
            (
                previous > 0.0
                and proposed < 0.0
            )
            or
            (
                previous < 0.0
                and proposed > 0.0
            )
        )

        if crossing_zero:

            if (
                abs(proposed)
                < self.hysteresis
            ):
                return True

        return False

    def _limit_step(
        self,
        delta: float,
    ) -> float:

        return max(
            -self.maximum_adaptation_step,
            min(
                self.maximum_adaptation_step,
                delta,
            ),
        )

    def _clamp_bias(
        self,
        bias: float,
    ) -> float:

        return max(
            -self.maximum_bias,
            min(
                self.maximum_bias,
                float(bias),
            ),
        )
