"""
Exploration bias adaptation.

Converts historical exploration outcomes into a bounded
behavioral bias.

This module does not decide whether the agent should
explore. It only measures how much exploration should
be encouraged or discouraged based on past outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorationBias:
    """
    Result of exploration bias adaptation.
    """

    bias: float
    success_rate: float
    evidence_strength: float
    pressure: float
    applied: bool
    reason: str

    @property
    def is_positive(self) -> bool:
        return self.bias > 0.0

    @property
    def is_negative(self) -> bool:
        return self.bias < 0.0

    @property
    def is_neutral(self) -> bool:
        return self.bias == 0.0


class ExplorationBiasAdapter:
    """
    Calculates a bounded exploration bias.

    Positive bias:
        exploration has historically worked well.

    Negative bias:
        exploration has historically worked poorly.

    The bias is based on:

        deviation from neutral success rate (0.5)
        × evidence strength

    This prevents a tiny amount of evidence from causing
    a large behavioral shift.
    """

    def __init__(
        self,
        neutral_success_rate: float = 0.50,
        minimum_evidence: int = 3,
        maximum_bias: float = 0.25,
        minimum_pressure: float = 0.05,
    ) -> None:

        if not 0.0 <= neutral_success_rate <= 1.0:
            raise ValueError(
                "neutral_success_rate must be between 0 and 1."
            )

        if minimum_evidence < 1:
            raise ValueError(
                "minimum_evidence must be >= 1."
            )

        if not 0.0 <= maximum_bias <= 1.0:
            raise ValueError(
                "maximum_bias must be between 0 and 1."
            )

        if not 0.0 <= minimum_pressure <= 1.0:
            raise ValueError(
                "minimum_pressure must be between 0 and 1."
            )

        self.neutral_success_rate = (
            neutral_success_rate
        )

        self.minimum_evidence = (
            minimum_evidence
        )

        self.maximum_bias = (
            maximum_bias
        )

        self.minimum_pressure = (
            minimum_pressure
        )

    def evaluate(
        self,
        attempts: int,
        successes: int,
        failures: int,
    ) -> ExplorationBias:
        """
        Evaluate exploration outcomes directly.

        `attempts` represents the number of exploration
        decisions.

        `successes + failures` must not exceed attempts.
        """

        self._validate_counts(
            attempts=attempts,
            successes=successes,
            failures=failures,
        )

        if attempts == 0:
            return ExplorationBias(
                bias=0.0,
                success_rate=0.0,
                evidence_strength=0.0,
                pressure=0.0,
                applied=False,
                reason=(
                    "No exploration evidence is available."
                ),
            )

        success_rate = (
            successes
            / attempts
        )

        evidence_strength = (
            self._evidence_strength(
                attempts
            )
        )

        raw_pressure = abs(
            success_rate
            - self.neutral_success_rate
        )

        pressure = (
            raw_pressure
            * evidence_strength
        )

        if pressure < self.minimum_pressure:
            return ExplorationBias(
                bias=0.0,
                success_rate=round(
                    success_rate,
                    6,
                ),
                evidence_strength=round(
                    evidence_strength,
                    6,
                ),
                pressure=round(
                    pressure,
                    6,
                ),
                applied=False,
                reason=(
                    "Exploration evidence is too weak "
                    "to change behavioral preference."
                ),
            )

        direction = (
            1.0
            if (
                success_rate
                > self.neutral_success_rate
            )
            else -1.0
        )

        bias = (
            direction
            * self.maximum_bias
            * pressure
        )

        bias = round(
            max(
                -self.maximum_bias,
                min(
                    self.maximum_bias,
                    bias,
                ),
            ),
            6,
        )

        return ExplorationBias(
            bias=bias,
            success_rate=round(
                success_rate,
                6,
            ),
            evidence_strength=round(
                evidence_strength,
                6,
            ),
            pressure=round(
                pressure,
                6,
            ),
            applied=bias != 0.0,
            reason=(
                "Exploration success history "
                "supports positive bias."
                if bias > 0
                else
                "Exploration failure history "
                "supports negative bias."
            ),
        )

    def evaluate_profile(
        self,
        profile,
    ) -> ExplorationBias:
        """
        Evaluate exploration history from a
        MetaLearningProfile.
        """

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        return self.evaluate(
            attempts=(
                profile.exploration_count
            ),
            successes=(
                profile.exploration_success_count
            ),
            failures=(
                profile.exploration_failure_count
            ),
        )

    def apply_to_score(
        self,
        exploration_score: float,
        bias: ExplorationBias,
    ) -> float:
        """
        Apply the bounded bias to an exploration score.
        """

        if bias is None:
            raise ValueError(
                "bias cannot be None."
            )

        return round(
            exploration_score
            + bias.bias,
            6,
        )

    def _evidence_strength(
        self,
        attempts: int,
    ) -> float:

        if attempts <= 0:
            return 0.0

        strength = (
            attempts
            / (
                attempts
                + self.minimum_evidence
            )
        )

        return min(
            1.0,
            max(
                0.0,
                strength,
            ),
        )

    @staticmethod
    def _validate_counts(
        attempts: int,
        successes: int,
        failures: int,
    ) -> None:

        if attempts < 0:
            raise ValueError(
                "attempts cannot be negative."
            )

        if successes < 0:
            raise ValueError(
                "successes cannot be negative."
            )

        if failures < 0:
            raise ValueError(
                "failures cannot be negative."
            )

        if (
            successes + failures
            > attempts
        ):
            raise ValueError(
                "successes + failures cannot exceed attempts."
            )
