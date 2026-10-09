from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapabilityTelemetryEvent:
    """
    Immutable record of one capability observation.
    """

    event_id: int
    capability: str
    success: bool | None
    source: str

    @property
    def has_outcome(self) -> bool:
        return self.success is not None


@dataclass(frozen=True)
class CapabilityTelemetrySnapshot:
    """
    Serializable snapshot of aggregated capability telemetry.
    """

    capabilities: dict[str, dict[str, int]]
    event_count: int
    total_attempts: int
    total_successes: int
    total_failures: int

    @property
    def capability_count(self) -> int:
        return len(self.capabilities)

    @property
    def has_evidence(self) -> bool:
        return self.capability_count > 0

    @property
    def observed_event_count(self) -> int:
        return self.event_count


class CapabilityTelemetry:
    """
    Explicit capability telemetry collector.

    The telemetry layer records direct evidence about what
    capability the Agent attempted and what happened.

    Each capability maintains:

        attempts
        successes
        failures

    An event with success=None counts as an attempt but does
    not count as success or failure.

    Example:

        telemetry.record(
            "execution",
            success=True,
            source="decide_and_act",
        )

    produces:

        {
            "execution": {
                "attempts": 1,
                "successes": 1,
                "failures": 0,
            }
        }
    """

    def __init__(self) -> None:
        self._records: dict[
            str,
            dict[str, int],
        ] = {}

        self._events: list[
            CapabilityTelemetryEvent
        ] = []

        self._next_event_id = 1

    # =====================================================
    # Validation / normalization
    # =====================================================

    @staticmethod
    def normalize_capability(
        capability: Any,
    ) -> str:
        """
        Normalize a capability name into a stable identifier.
        """

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

    @staticmethod
    def _validate_source(
        source: Any,
    ) -> str:
        if source is None:
            raise ValueError(
                "source cannot be None."
            )

        normalized = str(
            source
        ).strip()

        if not normalized:
            raise ValueError(
                "source cannot be empty."
            )

        return normalized

    @staticmethod
    def _validate_success(
        success: Any,
    ) -> bool | None:
        if success is None:
            return None

        if isinstance(
            success,
            bool,
        ):
            return success

        raise TypeError(
            "success must be bool or None."
        )

    # =====================================================
    # Recording
    # =====================================================

    def record(
        self,
        capability: Any,
        success: bool | None = None,
        *,
        source: str = "runtime",
    ) -> CapabilityTelemetryEvent:
        """
        Record one capability observation.
        """

        normalized_capability = (
            self.normalize_capability(
                capability
            )
        )

        normalized_source = (
            self._validate_source(
                source
            )
        )

        normalized_success = (
            self._validate_success(
                success
            )
        )

        record = self._records.setdefault(
            normalized_capability,
            {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
            },
        )

        record["attempts"] += 1

        if normalized_success is True:
            record["successes"] += 1

        elif normalized_success is False:
            record["failures"] += 1

        event = CapabilityTelemetryEvent(
            event_id=self._next_event_id,
            capability=normalized_capability,
            success=normalized_success,
            source=normalized_source,
        )

        self._events.append(
            event
        )

        self._next_event_id += 1

        return event

    def record_attempt(
        self,
        capability: Any,
        *,
        source: str = "runtime",
    ) -> CapabilityTelemetryEvent:
        """
        Record an attempt without a known outcome.
        """

        return self.record(
            capability,
            None,
            source=source,
        )

    def record_success(
        self,
        capability: Any,
        *,
        source: str = "runtime",
    ) -> CapabilityTelemetryEvent:
        """
        Record a successful capability attempt.
        """

        return self.record(
            capability,
            True,
            source=source,
        )

    def record_failure(
        self,
        capability: Any,
        *,
        source: str = "runtime",
    ) -> CapabilityTelemetryEvent:
        """
        Record a failed capability attempt.
        """

        return self.record(
            capability,
            False,
            source=source,
        )

    # =====================================================
    # Retrieval
    # =====================================================

    def get(
        self,
        capability: Any,
    ) -> dict[str, int] | None:
        """
        Return a copy of one capability's aggregate evidence.
        """

        normalized = (
            self.normalize_capability(
                capability
            )
        )

        record = self._records.get(
            normalized
        )

        if record is None:
            return None

        return dict(
            record
        )

    def evidence(
        self,
    ) -> dict[str, dict[str, int]]:
        """
        Return aggregated capability evidence.

        A fresh dictionary is returned so callers cannot
        mutate internal telemetry state.
        """

        return {
            capability: dict(
                record
            )
            for capability, record
            in self._records.items()
        }

    def events(
        self,
    ) -> list[CapabilityTelemetryEvent]:
        """
        Return a copy of all telemetry events.
        """

        return list(
            self._events
        )

    # =====================================================
    # Counts
    # =====================================================

    def count(
        self,
        capability: Any,
    ) -> int:
        """
        Return attempt count for a capability.
        """

        record = self.get(
            capability
        )

        if record is None:
            return 0

        return record["attempts"]

    def capability_count(
        self,
    ) -> int:
        return len(
            self._records
        )

    def event_count(
        self,
    ) -> int:
        return len(
            self._events
        )

    def total_attempts(
        self,
    ) -> int:
        return sum(
            record["attempts"]
            for record in self._records.values()
        )

    def total_successes(
        self,
    ) -> int:
        return sum(
            record["successes"]
            for record in self._records.values()
        )

    def total_failures(
        self,
    ) -> int:
        return sum(
            record["failures"]
            for record in self._records.values()
        )

    # =====================================================
    # Snapshot
    # =====================================================

    def snapshot(
        self,
    ) -> CapabilityTelemetrySnapshot:
        """
        Return a serializable telemetry snapshot.
        """

        return CapabilityTelemetrySnapshot(
            capabilities=self.evidence(),
            event_count=self.event_count(),
            total_attempts=self.total_attempts(),
            total_successes=self.total_successes(),
            total_failures=self.total_failures(),
        )

    # =====================================================
    # Lifecycle
    # =====================================================

    def clear(
        self,
    ) -> None:
        """
        Clear all telemetry and reset event numbering.
        """

        self._records.clear()
        self._events.clear()
        self._next_event_id = 1

    def __len__(
        self,
    ) -> int:
        return self.capability_count

    def __contains__(
        self,
        capability: Any,
    ) -> bool:
        normalized = (
            self.normalize_capability(
                capability
            )
        )

        return normalized in self._records
