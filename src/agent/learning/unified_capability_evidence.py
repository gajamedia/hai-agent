from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class UnifiedCapabilityEvidenceResult:
    """
    Result of merging capability evidence from multiple
    sources.

    Explicit telemetry has priority over experience-derived
    evidence when the same capability exists in both sources.
    """

    capabilities: dict[str, dict[str, int]]
    telemetry_capability_count: int
    experience_capability_count: int
    overlap_count: int
    telemetry_priority: bool

    @property
    def capability_count(self) -> int:
        return len(self.capabilities)

    @property
    def has_evidence(self) -> bool:
        return self.capability_count > 0

    @property
    def has_overlap(self) -> bool:
        return self.overlap_count > 0


class UnifiedCapabilityEvidenceBuilder:
    """
    Merge capability evidence from explicit telemetry and
    experience-derived evidence.

    Evidence source priority:

        explicit telemetry
            >
        experience-derived evidence

    The builder intentionally does NOT add the two records
    together when the same capability exists in both sources.

    Example:

        experience:
            execution = 10 attempts

        telemetry:
            execution = 3 attempts

    Result:

        execution = telemetry record

    This avoids counting the same runtime behavior twice.
    """

    def __init__(
        self,
        *,
        telemetry_priority: bool = True,
    ) -> None:
        self.telemetry_priority = (
            telemetry_priority
        )

    # =====================================================
    # Normalization
    # =====================================================

    @staticmethod
    def normalize_capability(
        capability: Any,
    ) -> str:
        if capability is None:
            raise ValueError(
                "capability cannot be None."
            )

        text = str(
            capability
        ).strip().lower()

        if not text:
            raise ValueError(
                "capability cannot be empty."
            )

        normalized = (
            text
            .replace("-", "_")
            .replace(" ", "_")
        )

        while "__" in normalized:
            normalized = normalized.replace(
                "__",
                "_",
            )

        return normalized

    # =====================================================
    # Evidence validation
    # =====================================================

    @staticmethod
    def _normalize_record(
        record: Mapping[str, Any],
    ) -> dict[str, int]:
        if not isinstance(
            record,
            Mapping,
        ):
            raise TypeError(
                "capability evidence must be a mapping."
            )

        attempts = record.get(
            "attempts",
            0,
        )

        successes = record.get(
            "successes",
            0,
        )

        failures = record.get(
            "failures",
            0,
        )

        values = {
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
        }

        for key, value in values.items():

            if not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{key} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"{key} cannot be negative."
                )

        return values

    @classmethod
    def _normalize_source(
        cls,
        evidence: Mapping[
            Any,
            Mapping[str, Any],
        ],
    ) -> dict[
        str,
        dict[str, int],
    ]:
        if evidence is None:
            return {}

        if not isinstance(
            evidence,
            Mapping,
        ):
            raise TypeError(
                "evidence must be a mapping."
            )

        normalized: dict[
            str,
            dict[str, int],
        ] = {}

        for capability, record in evidence.items():

            key = cls.normalize_capability(
                capability
            )

            normalized[key] = (
                cls._normalize_record(
                    record
                )
            )

        return normalized

    # =====================================================
    # Build
    # =====================================================

    def build(
        self,
        *,
        telemetry_evidence: Mapping[
            Any,
            Mapping[str, Any],
        ] | None = None,
        experience_evidence: Mapping[
            Any,
            Mapping[str, Any],
        ] | None = None,
    ) -> UnifiedCapabilityEvidenceResult:
        """
        Merge telemetry and experience evidence.

        When telemetry_priority=True:
            explicit telemetry wins on overlap.

        When telemetry_priority=False:
            experience-derived evidence wins on overlap.
        """

        telemetry = (
            self._normalize_source(
                telemetry_evidence
                if telemetry_evidence is not None
                else {}
            )
        )

        experience = (
            self._normalize_source(
                experience_evidence
                if experience_evidence is not None
                else {}
            )
        )

        overlap = (
            set(telemetry)
            & set(experience)
        )

        if self.telemetry_priority:

            # Experience provides the base set of capabilities.
            # Explicit telemetry overwrites overlapping
            # capabilities because telemetry has priority.
            capabilities = {
                capability: dict(record)
                for capability, record
                in experience.items()
            }

            for capability, record in telemetry.items():
                capabilities[
                    capability
                ] = dict(record)

        else:

            # Telemetry provides the base set of capabilities.
            # Experience overwrites overlapping capabilities
            # because experience has priority.
            capabilities = {
                capability: dict(record)
                for capability, record
                in telemetry.items()
            }

            for capability, record in experience.items():
                capabilities[
                    capability
                ] = dict(record)

        return UnifiedCapabilityEvidenceResult(
            capabilities=capabilities,
            telemetry_capability_count=len(
                telemetry
            ),
            experience_capability_count=len(
                experience
            ),
            overlap_count=len(
                overlap
            ),
            telemetry_priority=(
                self.telemetry_priority
            ),
        )
