"""
Meta-learning feedback.

Transforms aggregated meta-learning evidence into a structured
feedback signal that can later be consumed by decision-making
or adaptation policies.

This module does not directly modify AgentDecisionMaker,
strategies, planners, or thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from agent.learning.meta_learning import (
    MetaLearningProfile,
)


@dataclass(frozen=True)
class MetaLearningFeedback:
    """
    Structured feedback produced from a MetaLearningProfile.

    The feedback describes how the agent appears to be learning
    and what behavioral direction may be appropriate next.

    It is intentionally declarative: consumers decide whether
    and how to apply it.
    """

    preferred_mode: Optional[str] = None

    exploration_pressure: float = 0.0
    exploitation_pressure: float = 0.0

    adaptation_pressure: float = 0.0

    confidence: float = 0.0

    recommendations: List[str] = field(
        default_factory=list
    )

    # =====================================================
    # Derived helpers
    # =====================================================

    @property
    def should_explore(self) -> bool:
        return (
            self.preferred_mode == "explore"
        )

    @property
    def should_exploit(self) -> bool:
        return (
            self.preferred_mode == "exploit"
        )

    @property
    def is_neutral(self) -> bool:
        return (
            self.preferred_mode is None
        )

    @property
    def requires_adaptation(self) -> bool:
        return (
            self.adaptation_pressure > 0.0
        )

    @property
    def strongest_pressure(self) -> float:
        return max(
            self.exploration_pressure,
            self.exploitation_pressure,
            self.adaptation_pressure,
        )


class MetaLearningFeedbackEngine:
    """
    Analyze meta-learning evidence and produce feedback.

    Default policy:

    - strong exploration performance can support EXPLORE
    - strong exploitation performance can support EXPLOIT
    - weak decision performance increases adaptation pressure
    - insufficient evidence produces neutral feedback

    The engine does not directly change decision thresholds.
    """

    def __init__(
        self,
        min_decisions: int = 3,
        min_adaptation_events: int = 2,
        exploration_advantage: float = 0.10,
        exploitation_advantage: float = 0.10,
        weak_success_rate: float = 0.40,
        strong_success_rate: float = 0.70,
        max_pressure: float = 1.0,
    ) -> None:

        if min_decisions < 0:
            raise ValueError(
                "min_decisions cannot be negative."
            )

        if min_adaptation_events < 0:
            raise ValueError(
                "min_adaptation_events cannot be negative."
            )

        if exploration_advantage < 0:
            raise ValueError(
                "exploration_advantage cannot be negative."
            )

        if exploitation_advantage < 0:
            raise ValueError(
                "exploitation_advantage cannot be negative."
            )

        if not 0.0 <= weak_success_rate <= 1.0:
            raise ValueError(
                "weak_success_rate must be between 0 and 1."
            )

        if not 0.0 <= strong_success_rate <= 1.0:
            raise ValueError(
                "strong_success_rate must be between 0 and 1."
            )

        if weak_success_rate > strong_success_rate:
            raise ValueError(
                "weak_success_rate cannot exceed "
                "strong_success_rate."
            )

        if max_pressure <= 0.0:
            raise ValueError(
                "max_pressure must be greater than zero."
            )

        self.min_decisions = min_decisions
        self.min_adaptation_events = (
            min_adaptation_events
        )

        self.exploration_advantage = (
            exploration_advantage
        )

        self.exploitation_advantage = (
            exploitation_advantage
        )

        self.weak_success_rate = (
            weak_success_rate
        )

        self.strong_success_rate = (
            strong_success_rate
        )

        self.max_pressure = max_pressure

    # =====================================================
    # Public API
    # =====================================================

    def analyze(
        self,
        profile: MetaLearningProfile,
    ) -> MetaLearningFeedback:

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        total_decisions = (
            profile.total_decisions
        )

        total_adaptation = (
            profile.total_adaptation_events
        )

        # -------------------------------------------------
        # Insufficient evidence
        # -------------------------------------------------

        if total_decisions < self.min_decisions:
            return MetaLearningFeedback(
                preferred_mode=None,
                exploration_pressure=0.0,
                exploitation_pressure=0.0,
                adaptation_pressure=(
                    self._adaptation_pressure(
                        profile
                    )
                    if total_adaptation
                    >= self.min_adaptation_events
                    else 0.0
                ),
                confidence=0.0,
                recommendations=[
                    "collect more decision evidence"
                ],
            )

        exploration_rate = (
            profile.exploration_success_rate
        )

        exploitation_rate = (
            profile.exploitation_success_rate
        )

        # -------------------------------------------------
        # Confidence based on evidence volume
        # -------------------------------------------------

        confidence = self._confidence(
            total_decisions
        )

        # -------------------------------------------------
        # Relative performance
        # -------------------------------------------------

        difference = (
            exploitation_rate
            - exploration_rate
        )

        exploration_pressure = 0.0
        exploitation_pressure = 0.0
        preferred_mode: Optional[str] = None
        recommendations: list[str] = []

        # -------------------------------------------------
        # Strong exploitation
        # -------------------------------------------------

        if (
            exploitation_rate
            >= self.strong_success_rate
            and difference
            >= self.exploitation_advantage
        ):
            exploitation_pressure = (
                self._clamp_pressure(
                    difference
                )
            )

            preferred_mode = "exploit"

            recommendations.append(
                "prefer exploitation when "
                "exploration is not necessary"
            )

        # -------------------------------------------------
        # Strong exploration
        # -------------------------------------------------

        elif (
            exploration_rate
            >= self.strong_success_rate
            and (
                exploration_rate
                - exploitation_rate
            )
            >= self.exploration_advantage
        ):
            exploration_pressure = (
                self._clamp_pressure(
                    exploration_rate
                    - exploitation_rate
                )
            )

            preferred_mode = "explore"

            recommendations.append(
                "exploration is producing "
                "strong outcomes"
            )

        # -------------------------------------------------
        # Weak exploration
        # -------------------------------------------------

        if (
            profile.exploration_count > 0
            and exploration_rate
            <= self.weak_success_rate
        ):
            recommendations.append(
                "exploration performance is weak"
            )

        # -------------------------------------------------
        # Weak exploitation
        # -------------------------------------------------

        if (
            profile.exploitation_count > 0
            and exploitation_rate
            <= self.weak_success_rate
        ):
            recommendations.append(
                "exploitation performance is weak"
            )

        # -------------------------------------------------
        # Adaptation
        # -------------------------------------------------

        adaptation_pressure = 0.0

        if (
            total_adaptation
            >= self.min_adaptation_events
        ):
            adaptation_pressure = (
                self._adaptation_pressure(
                    profile
                )
            )

        if adaptation_pressure > 0.0:
            recommendations.append(
                "review adaptation outcomes"
            )

        # -------------------------------------------------
        # Neutral fallback
        # -------------------------------------------------

        if preferred_mode is None:
            recommendations.append(
                "maintain current decision policy"
            )

        return MetaLearningFeedback(
            preferred_mode=preferred_mode,
            exploration_pressure=(
                exploration_pressure
            ),
            exploitation_pressure=(
                exploitation_pressure
            ),
            adaptation_pressure=(
                adaptation_pressure
            ),
            confidence=confidence,
            recommendations=recommendations,
        )

    # =====================================================
    # Internal calculations
    # =====================================================

    def _confidence(
        self,
        total_decisions: int,
    ) -> float:

        if total_decisions <= 0:
            return 0.0

        return round(
            min(
                1.0,
                total_decisions
                / (
                    total_decisions
                    + self.min_decisions
                ),
            ),
            6,
        )

    def _adaptation_pressure(
        self,
        profile: MetaLearningProfile,
    ) -> float:

        rate = (
            profile.overall_adaptation_success_rate
        )

        if rate >= self.strong_success_rate:
            return 0.0

        if rate <= self.weak_success_rate:
            return self.max_pressure

        span = (
            self.strong_success_rate
            - self.weak_success_rate
        )

        pressure = (
            (
                self.strong_success_rate
                - rate
            )
            / span
        )

        return round(
            self._clamp_pressure(
                pressure
            ),
            6,
        )

    def _clamp_pressure(
        self,
        value: float,
    ) -> float:

        return round(
            max(
                0.0,
                min(
                    self.max_pressure,
                    value,
                ),
            ),
            6,
        )
