"""
Capability confidence integration.

Converts self-model behavioral evidence into normalized
capability confidence estimates.

This module does not modify the existing SelfModel or
ConfidenceScore classes yet.

It provides a stable integration boundary for the next
self-model stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent.learning.self_model_behavioral_evidence import (
    CapabilityBehavioralEvidence,
    SelfModelBehavioralEvidence,
)


@dataclass(frozen=True)
class CapabilityConfidenceAssessment:
    """
    Confidence assessment for a single capability.
    """

    capability: str

    confidence: float
    uncertainty: float

    success_rate: float
    evidence_strength: float

    source_confidence: float

    @property
    def high_confidence(self) -> bool:
        return self.confidence >= 0.70

    @property
    def low_confidence(self) -> bool:
        return self.confidence < 0.40

    @property
    def high_uncertainty(self) -> bool:
        return self.uncertainty >= 0.60

    @property
    def reliable(self) -> bool:
        return (
            self.confidence >= 0.50
            and self.uncertainty <= 0.50
            and self.evidence_strength >= 0.50
        )


@dataclass(frozen=True)
class CapabilityConfidenceSummary:
    """
    Aggregate confidence assessment across capabilities.
    """

    assessments: tuple[
        CapabilityConfidenceAssessment,
        ...
    ]

    average_confidence: float
    average_uncertainty: float

    highest_confidence_capability: str | None
    lowest_confidence_capability: str | None

    @property
    def capability_count(self) -> int:
        return len(self.assessments)

    @property
    def has_evidence(self) -> bool:
        return self.capability_count > 0

    @property
    def high_confidence_count(self) -> int:
        return sum(
            1
            for item in self.assessments
            if item.high_confidence
        )

    @property
    def low_confidence_count(self) -> int:
        return sum(
            1
            for item in self.assessments
            if item.low_confidence
        )

    def get(
        self,
        capability: str,
    ) -> CapabilityConfidenceAssessment | None:

        for assessment in self.assessments:
            if assessment.capability == capability:
                return assessment

        return None

    def snapshot(self) -> dict:
        return {
            "assessments": [
                {
                    "capability": item.capability,
                    "confidence": item.confidence,
                    "uncertainty": item.uncertainty,
                    "success_rate": item.success_rate,
                    "evidence_strength": (
                        item.evidence_strength
                    ),
                    "source_confidence": (
                        item.source_confidence
                    ),
                }
                for item in self.assessments
            ],
            "average_confidence": (
                self.average_confidence
            ),
            "average_uncertainty": (
                self.average_uncertainty
            ),
            "highest_confidence_capability": (
                self.highest_confidence_capability
            ),
            "lowest_confidence_capability": (
                self.lowest_confidence_capability
            ),
        }


class CapabilityConfidenceIntegrator:
    """
    Converts capability behavioral evidence into confidence.

    Default weighting:

        source confidence   0.50
        success rate        0.30
        evidence strength   0.20

    Therefore confidence reflects both:

        "How successful am I?"

    and:

        "How certain is the evidence behind that belief?"
    """

    def __init__(
        self,
        source_confidence_weight: float = 0.50,
        success_rate_weight: float = 0.30,
        evidence_strength_weight: float = 0.20,
    ) -> None:

        self._validate_weight(
            source_confidence_weight,
            "source_confidence_weight",
        )

        self._validate_weight(
            success_rate_weight,
            "success_rate_weight",
        )

        self._validate_weight(
            evidence_strength_weight,
            "evidence_strength_weight",
        )

        total = (
            source_confidence_weight
            + success_rate_weight
            + evidence_strength_weight
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "confidence weights must sum to 1."
            )

        self.source_confidence_weight = (
            source_confidence_weight
        )

        self.success_rate_weight = (
            success_rate_weight
        )

        self.evidence_strength_weight = (
            evidence_strength_weight
        )

    def assess(
        self,
        evidence: CapabilityBehavioralEvidence,
    ) -> CapabilityConfidenceAssessment:

        if evidence is None:
            raise ValueError(
                "evidence cannot be None."
            )

        if not isinstance(
            evidence,
            CapabilityBehavioralEvidence,
        ):
            raise ValueError(
                "evidence must be "
                "CapabilityBehavioralEvidence."
            )

        source_confidence = (
            evidence.confidence
        )

        confidence = (
            self.source_confidence_weight
            * source_confidence
            + self.success_rate_weight
            * evidence.success_rate
            + self.evidence_strength_weight
            * evidence.evidence_strength
        )

        confidence = self._clamp(
            confidence
        )

        # Confidence and uncertainty represent
        # complementary beliefs at this integration
        # boundary.
        uncertainty = (
            1.0
            - confidence
        )

        return CapabilityConfidenceAssessment(
            capability=evidence.capability,
            confidence=round(
                confidence,
                6,
            ),
            uncertainty=round(
                uncertainty,
                6,
            ),
            success_rate=round(
                evidence.success_rate,
                6,
            ),
            evidence_strength=round(
                evidence.evidence_strength,
                6,
            ),
            source_confidence=round(
                source_confidence,
                6,
            ),
        )

    def assess_evidence(
        self,
        evidence: SelfModelBehavioralEvidence,
    ) -> CapabilityConfidenceSummary:

        if evidence is None:
            raise ValueError(
                "evidence cannot be None."
            )

        assessments = tuple(
            self.assess(item)
            for item in evidence.capabilities
            if item.has_evidence
        )

        if not assessments:
            return CapabilityConfidenceSummary(
                assessments=tuple(),
                average_confidence=0.0,
                average_uncertainty=0.0,
                highest_confidence_capability=None,
                lowest_confidence_capability=None,
            )

        average_confidence = (
            sum(
                item.confidence
                for item in assessments
            )
            / len(assessments)
        )

        average_uncertainty = (
            sum(
                item.uncertainty
                for item in assessments
            )
            / len(assessments)
        )

        highest = max(
            assessments,
            key=lambda item: item.confidence,
        )

        lowest = min(
            assessments,
            key=lambda item: item.confidence,
        )

        return CapabilityConfidenceSummary(
            assessments=assessments,
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

    def assess_capabilities(
        self,
        capabilities: Iterable[
            CapabilityBehavioralEvidence
        ],
    ) -> CapabilityConfidenceSummary:

        if capabilities is None:
            raise ValueError(
                "capabilities cannot be None."
            )

        assessments = tuple(
            self.assess(item)
            for item in capabilities
            if item.has_evidence
        )

        if not assessments:
            return CapabilityConfidenceSummary(
                assessments=tuple(),
                average_confidence=0.0,
                average_uncertainty=0.0,
                highest_confidence_capability=None,
                lowest_confidence_capability=None,
            )

        average_confidence = (
            sum(
                item.confidence
                for item in assessments
            )
            / len(assessments)
        )

        average_uncertainty = (
            sum(
                item.uncertainty
                for item in assessments
            )
            / len(assessments)
        )

        highest = max(
            assessments,
            key=lambda item: item.confidence,
        )

        lowest = min(
            assessments,
            key=lambda item: item.confidence,
        )

        return CapabilityConfidenceSummary(
            assessments=assessments,
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

    def confidence_for(
        self,
        evidence: CapabilityBehavioralEvidence,
    ) -> float:

        return self.assess(
            evidence
        ).confidence

    def uncertainty_for(
        self,
        evidence: CapabilityBehavioralEvidence,
    ) -> float:

        return self.assess(
            evidence
        ).uncertainty

    @staticmethod
    def _validate_weight(
        value: float,
        name: str,
    ) -> None:

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return min(
            1.0,
            max(
                0.0,
                float(value),
            ),
        )
