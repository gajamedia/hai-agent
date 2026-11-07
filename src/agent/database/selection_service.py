"""User-confirmed database selection based on passive device discovery.

This service only describes readiness and records a caller's choice.  It does
not install drivers, start services, open connections, read credentials, or
create a database.  A separate, explicitly configured connection workflow is
required after a user confirms a compatible option.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from agent.device import DeviceProfile, DeviceProfiler

from .contracts import DatabaseRecommendation, WorkloadAnalysis, WorkloadRequest
from .database_detector import DatabaseDetectionReport, DatabaseDetector
from .database_selector import DatabaseSelector
from .workload_analyzer import WorkloadAnalyzer


class DatabaseSelectionError(ValueError):
    """The caller's selection cannot safely proceed to configuration."""


@dataclass(frozen=True)
class DatabaseSelectionAssessment:
    """A passive recommendation shown to the user before confirmation."""

    device: DeviceProfile
    detection: DatabaseDetectionReport
    workload: WorkloadAnalysis
    recommendation: DatabaseRecommendation

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device.to_dict(),
            "detection": self.detection.to_dict(),
            "workload": asdict(self.workload),
            "recommendation": self.recommendation.to_dict(),
        }


@dataclass(frozen=True)
class UserDatabaseSelection:
    """A validated user choice; it is not a connection configuration."""

    database: str
    user_confirmed: bool
    recommended_database: str | None
    compatible_with_device: bool
    ready_for_connection_attempt: bool
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatabaseSelectionService:
    """Assess local readiness, then require the user to choose an option."""

    def __init__(
        self,
        *,
        device_profiler: DeviceProfiler | None = None,
        database_detector: DatabaseDetector | None = None,
        workload_analyzer: WorkloadAnalyzer | None = None,
        database_selector: DatabaseSelector | None = None,
    ) -> None:
        self.device_profiler = device_profiler or DeviceProfiler()
        self.database_detector = database_detector or DatabaseDetector()
        self.workload_analyzer = workload_analyzer or WorkloadAnalyzer()
        self.database_selector = database_selector or DatabaseSelector()

    def assess(self, request: WorkloadRequest) -> DatabaseSelectionAssessment:
        if not isinstance(request, WorkloadRequest):
            raise TypeError("request must be WorkloadRequest")
        device = self.device_profiler.profile()
        detection = self.database_detector.detect()
        workload = self.workload_analyzer.analyze(request)
        recommendation = self.database_selector.recommend(request, workload, detection)
        return DatabaseSelectionAssessment(
            device=device,
            detection=detection,
            workload=workload,
            recommendation=self._with_device_advisory(recommendation, device),
        )

    @staticmethod
    def _with_device_advisory(
        recommendation: DatabaseRecommendation, device: DeviceProfile
    ) -> DatabaseRecommendation:
        reasons = list(recommendation.reasons)
        blockers = list(recommendation.blockers)
        available_memory = device.memory.available_bytes
        free_disk = device.disk.free_bytes
        if available_memory is not None and available_memory < 512 * 1024 * 1024:
            blockers.append("Available memory is below 512 MiB; avoid running a new database service on this device.")
        elif available_memory is not None:
            reasons.append("Device memory was profiled; capacity must still be confirmed for the expected dataset and concurrency.")
        if free_disk is not None and free_disk < 1024 * 1024 * 1024:
            blockers.append("Free disk space is below 1 GiB; do not create a local database before freeing space.")
        elif free_disk is not None:
            reasons.append("Device disk capacity was profiled; include database growth and backup space before provisioning.")
        return replace(recommendation, reasons=tuple(reasons), blockers=tuple(blockers))

    @staticmethod
    def confirm(assessment: DatabaseSelectionAssessment, database: str) -> UserDatabaseSelection:
        if not isinstance(assessment, DatabaseSelectionAssessment):
            raise TypeError("assessment must be DatabaseSelectionAssessment")
        if not isinstance(database, str) or not database.strip():
            raise DatabaseSelectionError("database choice must be non-empty text")
        selected = database.strip().lower()
        options = {
            str(option.get("database", "")).lower(): option
            for option in assessment.recommendation.detected_options
        }
        option = options.get(selected)
        if option is None:
            raise DatabaseSelectionError(
                "The selected database is not one of the agent's detected options."
            )
        ready = bool(option.get("ready_for_connection_attempt", option.get("driver_available", False)))
        if not ready:
            raise DatabaseSelectionError(
                "The selected database is not ready for a connection attempt; install/configure it outside the agent first."
            )
        return UserDatabaseSelection(
            database=selected,
            user_confirmed=True,
            recommended_database=assessment.recommendation.recommended_database,
            compatible_with_device=True,
            ready_for_connection_attempt=True,
            next_step=(
                "Collect an explicit, read-only connection profile and user-approved "
                "table/column allowlist; do not accept raw credentials or raw SQL."
            ),
        )


__all__ = [
    "DatabaseSelectionAssessment",
    "DatabaseSelectionError",
    "DatabaseSelectionService",
    "UserDatabaseSelection",
]
