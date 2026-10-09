"""
Behavioral evidence model.

This module converts raw meta-learning profile statistics
into explicit behavioral evidence that can later be used
by adaptive decision policies.

The model intentionally does not make decisions.
It only measures evidence.

Architecture:

    MetaLearningProfile
            |
            v
    BehavioralEvidence
            |
            v
    future adaptive policy
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class BehavioralModeEvidence:
    """
    Behavioral evidence for one decision mode.

    mode:
        "explore" or "exploit"

    attempts:
        Number of recorded decisions.

    successes:
        Number of successful decisions.

    failures:
        Number of failed decisions.

    success_rate:
        Raw observed success rate.

    evidence_strength:
        Confidence-like measure based on sample size.

    reliability:
        Success rate adjusted by evidence strength.

    stability:
        Stability estimate based on observed success rate.
    """

    mode: str
    attempts: int
    successes: int
    failures: int
    success_rate: float
    evidence_strength: float
    reliability: float
    stability: float

    @property
    def has_evidence(self) -> bool:
        return self.attempts > 0

    @property
    def balanced(self) -> bool:
        return (
            self.successes > 0
            and self.failures > 0
        )


@dataclass(frozen=True)
class BehavioralEvidence:
    """
    Combined behavioral evidence for the agent.

    The model is deliberately descriptive rather than
    prescriptive. It answers what the evidence currently
    says, not what the agent should do.
    """

    exploration: BehavioralModeEvidence
    exploitation: BehavioralModeEvidence

    preferred_mode: str | None
    preference_strength: float

    total_attempts: int
    total_successes: int
    total_failures: int

    overall_success_rate: float

    adaptation_signal: float

    @property
    def has_evidence(self) -> bool:
        return self.total_attempts > 0

    @property
    def is_balanced(self) -> bool:
        return (
            self.exploration.attempts > 0
            and self.exploitation.attempts > 0
        )

    @property
    def has_strong_preference(self) -> bool:
        return (
            self.preferred_mode is not None
            and self.preference_strength >= 0.20
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "exploration": {
                "attempts": self.exploration.attempts,
                "successes": self.exploration.successes,
                "failures": self.exploration.failures,
                "success_rate": (
                    self.exploration.success_rate
                ),
                "evidence_strength": (
                    self.exploration.evidence_strength
                ),
                "reliability": (
                    self.exploration.reliability
                ),
                "stability": (
                    self.exploration.stability
                ),
            },
            "exploitation": {
                "attempts": self.exploitation.attempts,
                "successes": self.exploitation.successes,
                "failures": self.exploitation.failures,
                "success_rate": (
                    self.exploitation.success_rate
                ),
                "evidence_strength": (
                    self.exploitation.evidence_strength
                ),
                "reliability": (
                    self.exploitation.reliability
                ),
                "stability": (
                    self.exploitation.stability
                ),
            },
            "preferred_mode": self.preferred_mode,
            "preference_strength": (
                self.preference_strength
            ),
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "overall_success_rate": (
                self.overall_success_rate
            ),
            "adaptation_signal": (
                self.adaptation_signal
            ),
        }


class BehavioralEvidenceModel:
    """
    Builds BehavioralEvidence from a MetaLearningProfile.

    No decision-making occurs here.
    """

    def __init__(
        self,
        minimum_evidence: int = 3,
        preference_threshold: float = 0.10,
    ) -> None:

        if minimum_evidence < 1:
            raise ValueError(
                "minimum_evidence must be >= 1."
            )

        if not 0.0 <= preference_threshold <= 1.0:
            raise ValueError(
                "preference_threshold must be between 0 and 1."
            )

        self.minimum_evidence = (
            minimum_evidence
        )

        self.preference_threshold = (
            preference_threshold
        )

    def analyze(
        self,
        profile,
    ) -> BehavioralEvidence:

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        exploration = (
            self._build_mode_evidence(
                mode="explore",
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
        )

        exploitation = (
            self._build_mode_evidence(
                mode="exploit",
                attempts=(
                    profile.exploitation_count
                ),
                successes=(
                    profile.exploitation_success_count
                ),
                failures=(
                    profile.exploitation_failure_count
                ),
            )
        )

        total_attempts = (
            exploration.attempts
            + exploitation.attempts
        )

        total_successes = (
            exploration.successes
            + exploitation.successes
        )

        total_failures = (
            exploration.failures
            + exploitation.failures
        )

        if total_attempts > 0:
            overall_success_rate = (
                total_successes
                / total_attempts
            )
        else:
            overall_success_rate = 0.0

        preferred_mode = (
            self._preferred_mode(
                exploration=exploration,
                exploitation=exploitation,
            )
        )

        preference_strength = (
            self._preference_strength(
                exploration=exploration,
                exploitation=exploitation,
            )
        )

        adaptation_signal = (
            self._adaptation_signal(
                exploration=exploration,
                exploitation=exploitation,
            )
        )

        return BehavioralEvidence(
            exploration=exploration,
            exploitation=exploitation,
            preferred_mode=preferred_mode,
            preference_strength=(
                preference_strength
            ),
            total_attempts=total_attempts,
            total_successes=total_successes,
            total_failures=total_failures,
            overall_success_rate=(
                round(
                    overall_success_rate,
                    6,
                )
            ),
            adaptation_signal=(
                adaptation_signal
            ),
        )

    def _build_mode_evidence(
        self,
        mode: str,
        attempts: int,
        successes: int,
        failures: int,
    ) -> BehavioralModeEvidence:

        if mode not in {
            "explore",
            "exploit",
        }:
            raise ValueError(
                "mode must be 'explore' or 'exploit'."
            )

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

        if successes + failures > attempts:
            raise ValueError(
                "successes + failures cannot exceed attempts."
            )

        if attempts == 0:
            success_rate = 0.0
        else:
            success_rate = (
                successes
                / attempts
            )

        evidence_strength = (
            self._evidence_strength(
                attempts
            )
        )

        reliability = (
            success_rate
            * evidence_strength
        )

        stability = (
            1.0
            - self._variance_proxy(
                success_rate
            )
        )

        return BehavioralModeEvidence(
            mode=mode,
            attempts=attempts,
            successes=successes,
            failures=failures,
            success_rate=round(
                success_rate,
                6,
            ),
            evidence_strength=round(
                evidence_strength,
                6,
            ),
            reliability=round(
                reliability,
                6,
            ),
            stability=round(
                stability,
                6,
            ),
        )

    def _evidence_strength(
        self,
        attempts: int,
    ) -> float:

        if attempts <= 0:
            return 0.0

        # Smooth asymptotic confidence.
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
    def _variance_proxy(
        success_rate: float,
    ) -> float:
        """
        Bernoulli variance normalized to 0..0.25.

        High variance means outcomes are mixed.
        Stable behavior approaches 1.0.
        """

        variance = (
            success_rate
            * (1.0 - success_rate)
        )

        normalized = (
            variance / 0.25
        )

        return min(
            1.0,
            max(
                0.0,
                normalized,
            ),
        )

    def _preferred_mode(
        self,
        exploration: BehavioralModeEvidence,
        exploitation: BehavioralModeEvidence,
    ) -> str | None:

        if not exploration.has_evidence:
            if exploitation.has_evidence:
                return "exploit"
            return None

        if not exploitation.has_evidence:
            return "explore"

        difference = (
            exploration.reliability
            - exploitation.reliability
        )

        if abs(difference) < self.preference_threshold:
            return None

        if difference > 0:
            return "explore"

        return "exploit"

    def _preference_strength(
        self,
        exploration: BehavioralModeEvidence,
        exploitation: BehavioralModeEvidence,
    ) -> float:

        difference = abs(
            exploration.reliability
            - exploitation.reliability
        )

        return round(
            min(
                1.0,
                difference,
            ),
            6,
        )

    @staticmethod
    def _adaptation_signal(
        exploration: BehavioralModeEvidence,
        exploitation: BehavioralModeEvidence,
    ) -> float:

        if not exploration.has_evidence:
            return 0.0

        if not exploitation.has_evidence:
            return 0.0

        # Large disagreement in reliability indicates
        # a stronger opportunity for behavioral adaptation.
        difference = abs(
            exploration.reliability
            - exploitation.reliability
        )

        # Mixed outcomes increase adaptation pressure.
        exploration_mixed = (
            exploration.balanced
        )

        exploitation_mixed = (
            exploitation.balanced
        )

        mixed_bonus = 0.10 * (
            int(exploration_mixed)
            + int(exploitation_mixed)
        )

        signal = min(
            1.0,
            difference + mixed_bonus,
        )

        return round(
            signal,
            6,
        )


def build_behavioral_evidence(
    profile,
    minimum_evidence: int = 3,
    preference_threshold: float = 0.10,
) -> BehavioralEvidence:
    """
    Convenience function for building behavioral evidence.
    """

    model = BehavioralEvidenceModel(
        minimum_evidence=minimum_evidence,
        preference_threshold=preference_threshold,
    )

    return model.analyze(
        profile
    )
