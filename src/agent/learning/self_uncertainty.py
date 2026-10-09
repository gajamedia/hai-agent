from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from agent.learning.capability_confidence import (
    CapabilityConfidenceAssessment,
    CapabilityConfidenceSummary,
)


# ============================================================
# Self Uncertainty Assessment
# ============================================================


@dataclass(frozen=True)
class SelfUncertaintyAssessment:
    """
    Uncertainty assessment for a single capability.
    """

    capability: str
    uncertainty: float
    confidence: float
    evidence_strength: float

    @property
    def high_uncertainty(self) -> bool:
        return self.uncertainty >= 0.60

    @property
    def low_uncertainty(self) -> bool:
        return self.uncertainty <= 0.40

    @property
    def reliable(self) -> bool:
        return (
            self.evidence_strength >= 0.50
            and self.confidence >= 0.60
            and self.uncertainty <= 0.40
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "evidence_strength": self.evidence_strength,
            "high_uncertainty": self.high_uncertainty,
            "low_uncertainty": self.low_uncertainty,
            "reliable": self.reliable,
        }


# ============================================================
# Self Uncertainty Result
# ============================================================


@dataclass(frozen=True)
class SelfUncertaintyResult:
    """
    Aggregate self-uncertainty state for the agent.
    """

    uncertainty: float
    confidence: float
    evidence_strength: float

    average_capability_uncertainty: float
    average_capability_confidence: float
    confidence_dispersion: float

    capability_count: int
    observed_capability_count: int

    high_uncertainty_count: int
    low_confidence_count: int

    strongest_capability: str | None
    weakest_capability: str | None

    @property
    def has_evidence(self) -> bool:
        return self.observed_capability_count > 0

    @property
    def uncertainty_high(self) -> bool:
        return self.uncertainty >= 0.60

    @property
    def uncertainty_low(self) -> bool:
        return self.uncertainty <= 0.40

    @property
    def confidence_high(self) -> bool:
        return self.confidence >= 0.60

    @property
    def confidence_low(self) -> bool:
        return self.confidence < 0.40

    @property
    def self_knowledge_stable(self) -> bool:
        """
        Self-knowledge is considered stable when uncertainty is low
        and confidence is sufficiently high.
        """
        return (
            self.uncertainty <= 0.40
            and self.confidence >= 0.60
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "evidence_strength": self.evidence_strength,
            "average_capability_uncertainty": (
                self.average_capability_uncertainty
            ),
            "average_capability_confidence": (
                self.average_capability_confidence
            ),
            "confidence_dispersion": (
                self.confidence_dispersion
            ),
            "capability_count": self.capability_count,
            "observed_capability_count": (
                self.observed_capability_count
            ),
            "high_uncertainty_count": (
                self.high_uncertainty_count
            ),
            "low_confidence_count": (
                self.low_confidence_count
            ),
            "strongest_capability": (
                self.strongest_capability
            ),
            "weakest_capability": (
                self.weakest_capability
            ),
            "has_evidence": self.has_evidence,
            "uncertainty_high": self.uncertainty_high,
            "uncertainty_low": self.uncertainty_low,
            "confidence_high": self.confidence_high,
            "confidence_low": self.confidence_low,
            "self_knowledge_stable": (
                self.self_knowledge_stable
            ),
        }


# ============================================================
# Self Uncertainty Integrator
# ============================================================


class SelfUncertaintyIntegrator:
    """
    Converts capability-level confidence information into
    an aggregate estimate of the agent's self-uncertainty.

    The capability uncertainty is intentionally dominant.

    Formula:

        uncertainty =
            capability_uncertainty_weight
            * average_capability_uncertainty

            + evidence_deficit_weight
            * evidence_deficit

            + dispersion_weight
            * confidence_dispersion

    Default weights:

        capability uncertainty = 0.85
        evidence deficit       = 0.10
        confidence dispersion  = 0.05

    where:

        evidence_deficit = 1 - average_evidence_strength
    """
    
    def __init__(
        self,
        *,
        capability_uncertainty_weight: float = 0.85,
        evidence_deficit_weight: float = 0.10,
        dispersion_weight: float = 0.05,
    ) -> None:

        self._validate_weight(
            capability_uncertainty_weight,
            "capability_uncertainty_weight",
        )

        self._validate_weight(
            evidence_deficit_weight,
            "evidence_deficit_weight",
        )

        self._validate_weight(
            dispersion_weight,
            "dispersion_weight",
        )

        total_weight = (
            capability_uncertainty_weight
            + evidence_deficit_weight
            + dispersion_weight
        )

        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(
                "Self uncertainty weights must sum to 1.0."
            )

        self.capability_uncertainty_weight = (
            float(capability_uncertainty_weight)
        )

        self.evidence_deficit_weight = (
            float(evidence_deficit_weight)
        )

        self.dispersion_weight = (
            float(dispersion_weight)
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def assess(
        self,
        summary: CapabilityConfidenceSummary,
    ) -> SelfUncertaintyResult:

        if summary is None:
            raise ValueError(
                "summary cannot be None."
            )

        if not isinstance(
            summary,
            CapabilityConfidenceSummary,
        ):
            raise ValueError(
                "summary must be "
                "CapabilityConfidenceSummary."
            )

        assessments = tuple(
            summary.assessments
        )

        # ----------------------------------------------------
        # No capability evidence
        # ----------------------------------------------------

        if not assessments:
            return SelfUncertaintyResult(
                uncertainty=1.0,
                confidence=0.0,
                evidence_strength=0.0,
                average_capability_uncertainty=1.0,
                average_capability_confidence=0.0,
                confidence_dispersion=0.0,
                capability_count=(
                    summary.capability_count
                ),
                observed_capability_count=0,
                high_uncertainty_count=0,
                low_confidence_count=0,
                strongest_capability=None,
                weakest_capability=None,
            )

        # ----------------------------------------------------
        # Capability aggregates
        # ----------------------------------------------------

        average_uncertainty = (
            sum(
                item.uncertainty
                for item in assessments
            )
            / len(assessments)
        )

        average_confidence = (
            sum(
                item.confidence
                for item in assessments
            )
            / len(assessments)
        )

        average_evidence_strength = (
            sum(
                item.evidence_strength
                for item in assessments
            )
            / len(assessments)
        )

        # Mean absolute deviation from average confidence.
        dispersion = (
            sum(
                abs(
                    item.confidence
                    - average_confidence
                )
                for item in assessments
            )
            / len(assessments)
        )

        # ----------------------------------------------------
        # Evidence deficit
        # ----------------------------------------------------

        evidence_deficit = (
            1.0
            - average_evidence_strength
        )

        # ----------------------------------------------------
        # Aggregate uncertainty
        # ----------------------------------------------------

        uncertainty = (
            self.capability_uncertainty_weight
            * average_uncertainty
            + self.evidence_deficit_weight
            * evidence_deficit
            + self.dispersion_weight
            * dispersion
        )

        uncertainty = self._clamp(
            uncertainty
        )

        # Confidence is intentionally the complement
        # of aggregate self-uncertainty.
        confidence = self._clamp(
            1.0 - uncertainty
        )

        # ----------------------------------------------------
        # Capability-level diagnostics
        # ----------------------------------------------------

        high_uncertainty_count = sum(
            1
            for item in assessments
            if item.uncertainty >= 0.60
        )

        low_confidence_count = sum(
            1
            for item in assessments
            if item.confidence < 0.40
        )

        strongest = max(
            assessments,
            key=lambda item: item.confidence,
        )

        weakest = min(
            assessments,
            key=lambda item: item.confidence,
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return SelfUncertaintyResult(
            uncertainty=round(
                uncertainty,
                6,
            ),
            confidence=round(
                confidence,
                6,
            ),
            evidence_strength=round(
                average_evidence_strength,
                6,
            ),
            average_capability_uncertainty=round(
                average_uncertainty,
                6,
            ),
            average_capability_confidence=round(
                average_confidence,
                6,
            ),
            confidence_dispersion=round(
                dispersion,
                6,
            ),
            capability_count=(
                summary.capability_count
            ),
            # IMPORTANT:
            # CapabilityConfidenceSummary does not expose
            # observed_capability_count. The observed count
            # is the number of assessments actually present.
            observed_capability_count=len(
                assessments
            ),
            high_uncertainty_count=(
                high_uncertainty_count
            ),
            low_confidence_count=(
                low_confidence_count
            ),
            strongest_capability=(
                strongest.capability
            ),
            weakest_capability=(
                weakest.capability
            ),
        )

    def assess_capabilities(
        self,
        assessments: Sequence[
            CapabilityConfidenceAssessment
        ],
    ) -> SelfUncertaintyResult:

        if assessments is None:
            raise ValueError(
                "assessments cannot be None."
            )

        if isinstance(
            assessments,
            (str, bytes),
        ):
            raise ValueError(
                "assessments must be a sequence of "
                "CapabilityConfidenceAssessment."
            )

        assessment_list = list(
            assessments
        )

        for item in assessment_list:
            if not isinstance(
                item,
                CapabilityConfidenceAssessment,
            ):
                raise ValueError(
                    "Every assessment must be "
                    "CapabilityConfidenceAssessment."
                )

        summary = (
            self._summary_from_assessments(
                assessment_list
            )
        )

        return self.assess(
            summary
        )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    @staticmethod
    def _summary_from_assessments(
        assessments: Sequence[
            CapabilityConfidenceAssessment
        ],
    ) -> CapabilityConfidenceSummary:

        items = tuple(
            assessments
        )

        if not items:
            return CapabilityConfidenceSummary(
                assessments=(),
                average_confidence=0.0,
                average_uncertainty=1.0,
                highest_confidence_capability=None,
                lowest_confidence_capability=None,
            )

        average_confidence = (
            sum(
                item.confidence
                for item in items
            )
            / len(items)
        )

        average_uncertainty = (
            sum(
                item.uncertainty
                for item in items
            )
            / len(items)
        )

        highest = max(
            items,
            key=lambda item: item.confidence,
        )

        lowest = min(
            items,
            key=lambda item: item.confidence,
        )

        return CapabilityConfidenceSummary(
            assessments=items,
            average_confidence=round(
                average_confidence,
                6,
            ),
            average_uncertainty=round(
                average_uncertainty,
                6,
            ),
            highest_confidence_capability=(
                highest.capability
            ),
            lowest_confidence_capability=(
                lowest.capability
            ),
        )

    @staticmethod
    def _validate_weight(
        value: float,
        name: str,
    ) -> None:

        if not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f"{name} must be numeric."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{name} must be between 0.0 and 1.0."
            )

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


def build_self_uncertainty(
    summary: CapabilityConfidenceSummary,
) -> SelfUncertaintyResult:

    return (
        SelfUncertaintyIntegrator().assess(
            summary
        )
    )
