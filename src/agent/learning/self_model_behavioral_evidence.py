"""
Self-model behavioral evidence.

This module transforms self-model capability outcomes into
explicit behavioral evidence.

It intentionally does not:
    - change the SelfModel
    - update confidence
    - choose actions
    - make decisions

It only describes what the current self-model evidence says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CapabilityBehavioralEvidence:
    """
    Behavioral evidence for a single capability.
    """

    capability: str

    attempts: int
    successes: int
    failures: int

    success_rate: float
    confidence: float
    uncertainty: float
    evidence_strength: float

    @property
    def has_evidence(self) -> bool:
        return self.attempts > 0

    @property
    def reliable(self) -> bool:
        return (
            self.evidence_strength >= 0.50
            and self.confidence >= 0.50
            and self.uncertainty <= 0.50
        )

    @property
    def weak(self) -> bool:
        return (
            self.confidence < 0.50
            or self.uncertainty > 0.50
        )


@dataclass(frozen=True)
class SelfModelBehavioralEvidence:
    """
    Aggregate behavioral evidence derived from the
    self-model's capability history.
    """

    capabilities: tuple[
        CapabilityBehavioralEvidence,
        ...
    ]

    average_confidence: float
    average_uncertainty: float
    average_success_rate: float
    evidence_strength: float

    strongest_capability: str | None
    weakest_capability: str | None

    capability_count: int
    observed_capability_count: int

    @property
    def has_evidence(self) -> bool:
        return (
            self.observed_capability_count > 0
        )

    @property
    def has_multiple_capabilities(self) -> bool:
        return (
            self.observed_capability_count >= 2
        )

    @property
    def confidence_high(self) -> bool:
        return (
            self.average_confidence >= 0.70
        )

    @property
    def uncertainty_high(self) -> bool:
        return (
            self.average_uncertainty >= 0.70
        )

    @property
    def is_stable(self) -> bool:
        return (
            self.average_confidence >= 0.50
            and self.average_uncertainty <= 0.50
        )

    def get_capability(
        self,
        capability: str,
    ) -> CapabilityBehavioralEvidence | None:
        for item in self.capabilities:
            if item.capability == capability:
                return item

        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "capabilities": [
                {
                    "capability": item.capability,
                    "attempts": item.attempts,
                    "successes": item.successes,
                    "failures": item.failures,
                    "success_rate": item.success_rate,
                    "confidence": item.confidence,
                    "uncertainty": item.uncertainty,
                    "evidence_strength": (
                        item.evidence_strength
                    ),
                }
                for item in self.capabilities
            ],
            "average_confidence": (
                self.average_confidence
            ),
            "average_uncertainty": (
                self.average_uncertainty
            ),
            "average_success_rate": (
                self.average_success_rate
            ),
            "evidence_strength": (
                self.evidence_strength
            ),
            "strongest_capability": (
                self.strongest_capability
            ),
            "weakest_capability": (
                self.weakest_capability
            ),
            "capability_count": (
                self.capability_count
            ),
            "observed_capability_count": (
                self.observed_capability_count
            ),
        }


class SelfModelBehavioralEvidenceModel:
    """
    Builds behavioral evidence from normalized self-model
    capability data.

    Input contract:

        {
            "capability_name": {
                "attempts": int,
                "successes": int,
                "failures": int,
                "confidence": float,
                "uncertainty": float,
            }
        }

    Only attempts/successes/failures are required.
    Confidence and uncertainty default to values derived
    from the observed success rate.

    This keeps the evidence model independent from the
    exact internal representation of SelfModel.
    """

    def __init__(
        self,
        minimum_evidence: int = 3,
    ) -> None:

        if minimum_evidence < 1:
            raise ValueError(
                "minimum_evidence must be >= 1."
            )

        self.minimum_evidence = (
            minimum_evidence
        )

    def analyze(
        self,
        capabilities: Mapping[str, Any],
    ) -> SelfModelBehavioralEvidence:

        if capabilities is None:
            raise ValueError(
                "capabilities cannot be None."
            )

        if not isinstance(
            capabilities,
            Mapping,
        ):
            raise ValueError(
                "capabilities must be a mapping."
            )

        items: list[
            CapabilityBehavioralEvidence
        ] = []

        for (
            capability,
            data,
        ) in capabilities.items():

            evidence = (
                self._build_capability_evidence(
                    capability=capability,
                    data=data,
                )
            )

            items.append(
                evidence
            )

        observed = [
            item
            for item in items
            if item.has_evidence
        ]

        if not observed:
            return SelfModelBehavioralEvidence(
                capabilities=tuple(items),
                average_confidence=0.0,
                average_uncertainty=0.0,
                average_success_rate=0.0,
                evidence_strength=0.0,
                strongest_capability=None,
                weakest_capability=None,
                capability_count=len(items),
                observed_capability_count=0,
            )

        total_attempts = sum(
            item.attempts
            for item in observed
        )

        if total_attempts <= 0:
            return SelfModelBehavioralEvidence(
                capabilities=tuple(items),
                average_confidence=0.0,
                average_uncertainty=0.0,
                average_success_rate=0.0,
                evidence_strength=0.0,
                strongest_capability=None,
                weakest_capability=None,
                capability_count=len(items),
                observed_capability_count=0,
            )

        weighted_confidence = sum(
            item.confidence * item.attempts
            for item in observed
        )

        weighted_uncertainty = sum(
            item.uncertainty * item.attempts
            for item in observed
        )

        weighted_success = sum(
            item.success_rate * item.attempts
            for item in observed
        )

        average_confidence = (
            weighted_confidence
            / total_attempts
        )

        average_uncertainty = (
            weighted_uncertainty
            / total_attempts
        )

        average_success_rate = (
            weighted_success
            / total_attempts
        )

        evidence_strength = (
            total_attempts
            / (
                total_attempts
                + self.minimum_evidence
            )
        )

        strongest = max(
            observed,
            key=lambda item: (
                item.success_rate,
                item.confidence,
                -item.uncertainty,
            ),
        )

        weakest = min(
            observed,
            key=lambda item: (
                item.success_rate,
                item.confidence,
                -item.uncertainty,
            ),
        )

        return SelfModelBehavioralEvidence(
            capabilities=tuple(items),
            average_confidence=round(
                average_confidence,
                6,
            ),
            average_uncertainty=round(
                average_uncertainty,
                6,
            ),
            average_success_rate=round(
                average_success_rate,
                6,
            ),
            evidence_strength=round(
                evidence_strength,
                6,
            ),
            strongest_capability=(
                strongest.capability
            ),
            weakest_capability=(
                weakest.capability
            ),
            capability_count=len(items),
            observed_capability_count=len(observed),
        )

    def _build_capability_evidence(
        self,
        capability: str,
        data: Any,
    ) -> CapabilityBehavioralEvidence:

        if not capability:
            raise ValueError(
                "capability cannot be empty."
            )

        normalized = (
            self._normalize_data(data)
        )

        attempts = int(
            normalized.get(
                "attempts",
                0,
            )
        )

        successes = int(
            normalized.get(
                "successes",
                0,
            )
        )

        failures = int(
            normalized.get(
                "failures",
                0,
            )
        )

        self._validate_counts(
            attempts=attempts,
            successes=successes,
            failures=failures,
        )

        if attempts > 0:
            success_rate = (
                successes
                / attempts
            )
        else:
            success_rate = 0.0

        confidence = normalized.get(
            "confidence",
            None,
        )

        if confidence is None:
            confidence = (
                success_rate
                if attempts > 0
                else 0.0
            )

        uncertainty = normalized.get(
            "uncertainty",
            None,
        )

        if uncertainty is None:
            uncertainty = (
                1.0
                - success_rate
                if attempts > 0
                else 1.0
            )

        confidence = self._normalize_signal(
            confidence,
            "confidence",
        )

        uncertainty = self._normalize_signal(
            uncertainty,
            "uncertainty",
        )

        evidence_strength = (
            self._evidence_strength(
                attempts
            )
        )

        return CapabilityBehavioralEvidence(
            capability=str(
                capability
            ),
            attempts=attempts,
            successes=successes,
            failures=failures,
            success_rate=round(
                success_rate,
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
            evidence_strength=round(
                evidence_strength,
                6,
            ),
        )

    @staticmethod
    def _normalize_data(
        data: Any,
    ) -> dict[str, Any]:

        if isinstance(
            data,
            Mapping,
        ):
            return dict(data)

        result: dict[str, Any] = {}

        for field in (
            "attempts",
            "successes",
            "failures",
            "confidence",
            "uncertainty",
        ):
            if hasattr(
                data,
                field,
            ):
                result[field] = getattr(
                    data,
                    field,
                )

        if not result:
            raise ValueError(
                "capability data must be a mapping "
                "or an object exposing behavioral fields."
            )

        return result

    def _evidence_strength(
        self,
        attempts: int,
    ) -> float:

        if attempts <= 0:
            return 0.0

        return min(
            1.0,
            max(
                0.0,
                attempts
                / (
                    attempts
                    + self.minimum_evidence
                ),
            ),
        )

    @staticmethod
    def _normalize_signal(
        value: float,
        name: str,
    ) -> float:

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

        return value

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
            successes
            + failures
            > attempts
        ):
            raise ValueError(
                "successes + failures cannot exceed attempts."
            )


def build_self_model_behavioral_evidence(
    capabilities: Mapping[str, Any],
    minimum_evidence: int = 3,
) -> SelfModelBehavioralEvidence:
    """
    Convenience helper.
    """

    model = (
        SelfModelBehavioralEvidenceModel(
            minimum_evidence=minimum_evidence,
        )
    )

    return model.analyze(
        capabilities
    )
