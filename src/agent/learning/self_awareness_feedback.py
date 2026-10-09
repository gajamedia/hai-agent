from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SelfAwarenessFeedbackRecord:
    """
    Immutable record of one self-awareness feedback cycle.
    """

    sequence: int
    uncertainty: float
    confidence: float
    preferred_mode: str | None
    selected_mode: str | None
    evidence_count: int
    mode_changed: bool

    @property
    def self_model_stable(self) -> bool:
        return (
            self.uncertainty
            + self.confidence
            == 1.0
        )


@dataclass(frozen=True)
class SelfAwarenessFeedbackSnapshot:
    """
    Snapshot of the complete self-awareness feedback trace.
    """

    records: tuple[
        SelfAwarenessFeedbackRecord,
        ...
    ]

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def has_feedback(self) -> bool:
        return bool(self.records)

    @property
    def first(self) -> SelfAwarenessFeedbackRecord | None:
        if not self.records:
            return None

        return self.records[0]

    @property
    def latest(self) -> SelfAwarenessFeedbackRecord | None:
        if not self.records:
            return None

        return self.records[-1]

    @property
    def uncertainty_delta(self) -> float:
        if len(self.records) < 2:
            return 0.0

        return round(
            self.records[-1].uncertainty
            - self.records[0].uncertainty,
            6,
        )

    @property
    def confidence_delta(self) -> float:
        if len(self.records) < 2:
            return 0.0

        return round(
            self.records[-1].confidence
            - self.records[0].confidence,
            6,
        )

    @property
    def selected_modes(self) -> tuple[str | None, ...]:
        return tuple(
            record.selected_mode
            for record in self.records
        )

    @property
    def preferred_modes(self) -> tuple[str | None, ...]:
        return tuple(
            record.preferred_mode
            for record in self.records
        )

    @property
    def changed_decision_count(self) -> int:
        return sum(
            1
            for record in self.records
            if record.mode_changed
        )


class SelfAwarenessFeedbackTrace:
    """
    Runtime trace for observing how self-awareness changes
    across decision cycles.

    This component does not modify self-awareness itself.

    It only records:

        self uncertainty
        self confidence
        preferred mode
        selected mode
        evidence count
        whether arbitration changed the mode

    Therefore it acts as an observability layer for the
    self-awareness feedback loop.
    """

    def __init__(self) -> None:
        self._records: list[
            SelfAwarenessFeedbackRecord
        ] = []

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _validate_probability(
        value: Any,
        name: str,
    ) -> float:
        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        normalized = float(value)

        if not 0.0 <= normalized <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

        return normalized

    @staticmethod
    def _validate_evidence_count(
        value: Any,
    ) -> int:
        if not isinstance(
            value,
            int,
        ):
            raise TypeError(
                "evidence_count must be an integer."
            )

        if value < 0:
            raise ValueError(
                "evidence_count cannot be negative."
            )

        return value

    @staticmethod
    def _normalize_mode(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if hasattr(
            value,
            "value",
        ):
            value = value.value

        text = str(
            value
        ).strip()

        if not text:
            return None

        return text

    # =====================================================
    # Record
    # =====================================================

    def record(
        self,
        *,
        uncertainty: float,
        confidence: float,
        preferred_mode: Any = None,
        selected_mode: Any = None,
        evidence_count: int = 0,
        mode_changed: bool = False,
    ) -> SelfAwarenessFeedbackRecord:
        """
        Record one feedback observation.
        """

        normalized_uncertainty = (
            self._validate_probability(
                uncertainty,
                "uncertainty",
            )
        )

        normalized_confidence = (
            self._validate_probability(
                confidence,
                "confidence",
            )
        )

        normalized_evidence_count = (
            self._validate_evidence_count(
                evidence_count
            )
        )

        if not isinstance(
            mode_changed,
            bool,
        ):
            raise TypeError(
                "mode_changed must be bool."
            )

        sequence = (
            len(self._records) + 1
        )

        record = (
            SelfAwarenessFeedbackRecord(
                sequence=sequence,
                uncertainty=(
                    normalized_uncertainty
                ),
                confidence=(
                    normalized_confidence
                ),
                preferred_mode=(
                    self._normalize_mode(
                        preferred_mode
                    )
                ),
                selected_mode=(
                    self._normalize_mode(
                        selected_mode
                    )
                ),
                evidence_count=(
                    normalized_evidence_count
                ),
                mode_changed=mode_changed,
            )
        )

        self._records.append(
            record
        )

        return record

    # =====================================================
    # Retrieval
    # =====================================================

    def records(
        self,
    ) -> tuple[
        SelfAwarenessFeedbackRecord,
        ...
    ]:
        return tuple(
            self._records
        )

    def latest(
        self,
    ) -> SelfAwarenessFeedbackRecord | None:
        if not self._records:
            return None

        return self._records[-1]

    def first(
        self,
    ) -> SelfAwarenessFeedbackRecord | None:
        if not self._records:
            return None

        return self._records[0]

    def count(
        self,
    ) -> int:
        return len(
            self._records
        )

    # =====================================================
    # Analysis
    # =====================================================

    def uncertainty_delta(
        self,
    ) -> float:
        snapshot = self.snapshot()

        return snapshot.uncertainty_delta

    def confidence_delta(
        self,
    ) -> float:
        snapshot = self.snapshot()

        return snapshot.confidence_delta

    def changed_decision_count(
        self,
    ) -> int:
        return sum(
            1
            for record in self._records
            if record.mode_changed
        )

    # =====================================================
    # Snapshot
    # =====================================================

    def snapshot(
        self,
    ) -> SelfAwarenessFeedbackSnapshot:
        return (
            SelfAwarenessFeedbackSnapshot(
                records=tuple(
                    self._records
                )
            )
        )

    # =====================================================
    # Lifecycle
    # =====================================================

    def clear(
        self,
    ) -> None:
        self._records.clear()

    def __len__(
        self,
    ) -> int:
        return len(
            self._records
        )
