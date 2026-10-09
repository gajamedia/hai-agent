"""
Meta-learning behavioral influence.

This module converts behavioral evidence into a bounded
decision influence that can later be applied to the
adaptive decision policy.

Important design rule:

    Meta-learning may influence a decision,
    but must not completely replace the base
    decision mechanism.

Therefore the output is always bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.learning.behavioral_evidence import (
    BehavioralEvidence,
)


@dataclass(frozen=True)
class BehavioralDecisionInfluence:
    """
    Bounded influence produced from behavioral evidence.
    """

    preferred_mode: Optional[str]

    explore_bonus: float
    exploit_bonus: float

    confidence: float
    strength: float

    applied: bool
    reason: str

    @property
    def strongest_bonus(self) -> float:
        return max(
            self.explore_bonus,
            self.exploit_bonus,
        )

    @property
    def is_neutral(self) -> bool:
        return (
            not self.applied
            or (
                self.explore_bonus == 0.0
                and self.exploit_bonus == 0.0
            )
        )

    @property
    def should_prefer_explore(self) -> bool:
        return (
            self.applied
            and self.preferred_mode
            == "explore"
            and self.explore_bonus
            > self.exploit_bonus
        )

    @property
    def should_prefer_exploit(self) -> bool:
        return (
            self.applied
            and self.preferred_mode
            == "exploit"
            and self.exploit_bonus
            > self.explore_bonus
        )


class MetaLearningBehavioralInfluence:
    """
    Transform BehavioralEvidence into bounded score
    influence.

    This class does not select a final action.
    """

    def __init__(
        self,
        minimum_confidence: float = 0.50,
        minimum_strength: float = 0.10,
        maximum_bonus: float = 0.25,
        minimum_comparative_evidence: int = 1,
    ) -> None:

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )

        if not 0.0 <= minimum_strength <= 1.0:
            raise ValueError(
                "minimum_strength must be between 0 and 1."
            )

        if not 0.0 <= maximum_bonus <= 1.0:
            raise ValueError(
                "maximum_bonus must be between 0 and 1."
            )

        if minimum_comparative_evidence < 1:
            raise ValueError(
                "minimum_comparative_evidence must be >= 1."
            )

        self.minimum_confidence = (
            minimum_confidence
        )

        self.minimum_strength = (
            minimum_strength
        )

        self.maximum_bonus = (
            maximum_bonus
        )

        self.minimum_comparative_evidence = (
            minimum_comparative_evidence
        )

    def evaluate(
        self,
        evidence: BehavioralEvidence,
    ) -> BehavioralDecisionInfluence:

        if evidence is None:
            raise ValueError(
                "evidence cannot be None."
            )

        confidence = (
            self._confidence(
                evidence
            )
        )

        strength = min(
            1.0,
            max(
                0.0,
                evidence.preference_strength,
            ),
        )

        preferred_mode = (
            evidence.preferred_mode
        )

        # -------------------------------------------------
        # Not enough comparative evidence.
        #
        # We intentionally require evidence for BOTH
        # modes before allowing behavioral preference.
        # -------------------------------------------------

        if not evidence.is_balanced:
            return self._neutral(
                confidence=confidence,
                strength=strength,
                reason=(
                    "Comparative behavioral evidence "
                    "is not available."
                ),
            )

        if (
            evidence.exploration.attempts
            < self.minimum_comparative_evidence
            or evidence.exploitation.attempts
            < self.minimum_comparative_evidence
        ):
            return self._neutral(
                confidence=confidence,
                strength=strength,
                reason=(
                    "Insufficient comparative evidence."
                ),
            )

        if confidence < self.minimum_confidence:
            return self._neutral(
                confidence=confidence,
                strength=strength,
                reason=(
                    "Behavioral evidence confidence "
                    "is below the activation threshold."
                ),
            )

        if preferred_mode is None:
            return self._neutral(
                confidence=confidence,
                strength=strength,
                reason=(
                    "Behavioral evidence does not "
                    "show a meaningful preference."
                ),
            )

        if strength < self.minimum_strength:
            return self._neutral(
                confidence=confidence,
                strength=strength,
                reason=(
                    "Behavioral preference is too weak "
                    "to influence the decision."
                ),
            )

        # -------------------------------------------------
        # Evidence quality.
        #
        # We use the weaker of the two mode evidence
        # strengths so that one well-supported mode cannot
        # create strong influence when the comparison side
        # has almost no evidence.
        # -------------------------------------------------

        comparative_strength = min(
            evidence.exploration.evidence_strength,
            evidence.exploitation.evidence_strength,
        )

        bounded_strength = min(
            1.0,
            strength,
        )

        bonus = (
            self.maximum_bonus
            * bounded_strength
            * comparative_strength
        )

        bonus = round(
            min(
                self.maximum_bonus,
                max(
                    0.0,
                    bonus,
                ),
            ),
            6,
        )

        if preferred_mode == "explore":

            return BehavioralDecisionInfluence(
                preferred_mode="explore",
                explore_bonus=bonus,
                exploit_bonus=0.0,
                confidence=round(
                    confidence,
                    6,
                ),
                strength=round(
                    strength,
                    6,
                ),
                applied=bonus > 0.0,
                reason=(
                    "Behavioral evidence favors "
                    "exploration."
                ),
            )

        if preferred_mode == "exploit":

            return BehavioralDecisionInfluence(
                preferred_mode="exploit",
                explore_bonus=0.0,
                exploit_bonus=bonus,
                confidence=round(
                    confidence,
                    6,
                ),
                strength=round(
                    strength,
                    6,
                ),
                applied=bonus > 0.0,
                reason=(
                    "Behavioral evidence favors "
                    "exploitation."
                ),
            )

        return self._neutral(
            confidence=confidence,
            strength=strength,
            reason=(
                "Behavioral preference mode is invalid."
            ),
        )

    def apply_to_scores(
        self,
        explore_score: float,
        exploit_score: float,
        influence: BehavioralDecisionInfluence,
    ) -> tuple[float, float]:

        if influence is None:
            raise ValueError(
                "influence cannot be None."
            )

        final_explore = (
            explore_score
            + influence.explore_bonus
        )

        final_exploit = (
            exploit_score
            + influence.exploit_bonus
        )

        return (
            round(
                final_explore,
                6,
            ),
            round(
                final_exploit,
                6,
            ),
        )

    def select_mode(
        self,
        explore_score: float,
        exploit_score: float,
        influence: BehavioralDecisionInfluence,
    ) -> str:

        final_explore, final_exploit = (
            self.apply_to_scores(
                explore_score=explore_score,
                exploit_score=exploit_score,
                influence=influence,
            )
        )

        # Tie remains exploit, matching the existing
        # AdaptiveDecisionPolicy convention.
        if final_explore > final_exploit:
            return "explore"

        return "exploit"

    def _confidence(
        self,
        evidence: BehavioralEvidence,
    ) -> float:

        total_attempts = (
            evidence.total_attempts
        )

        if total_attempts <= 0:
            return 0.0

        confidence = (
            total_attempts
            / (
                total_attempts
                + 3
            )
        )

        return min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

    @staticmethod
    def _neutral(
        confidence: float,
        strength: float,
        reason: str,
    ) -> BehavioralDecisionInfluence:

        return BehavioralDecisionInfluence(
            preferred_mode=None,
            explore_bonus=0.0,
            exploit_bonus=0.0,
            confidence=round(
                confidence,
                6,
            ),
            strength=round(
                strength,
                6,
            ),
            applied=False,
            reason=reason,
        )
