"""Conservative database recommendation logic.

This selector recommends only. It never opens connections, selects credentials,
changes configuration, migrates data, or executes a workload.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .contracts import (
    DatabaseRecommendation,
    WorkloadAnalysis,
    WorkloadRequest,
    WorkloadType,
)
from .database_detector import DatabaseDetectionReport


class DatabaseSelector:
    """Rank database options using workload fit and detected local readiness."""

    def recommend(
        self,
        request: WorkloadRequest,
        analysis: WorkloadAnalysis,
        detection: DatabaseDetectionReport,
    ) -> DatabaseRecommendation:
        reasons: List[str] = []
        blockers: List[str] = []
        options: List[Dict[str, object]] = []

        drivers_by_db: Dict[str, bool] = {}
        for driver in detection.drivers:
            if driver.available:
                drivers_by_db[driver.database] = True
            else:
                drivers_by_db.setdefault(driver.database, False)

        running_services = {
            service.database
            for service in detection.services
            if service.running
        }

        kind = analysis.workload_type
        if kind == WorkloadType.UNKNOWN:
            return DatabaseRecommendation(
                workload_type=kind,
                recommended_database=None,
                eligible=False,
                requires_user_approval=True,
                reasons=("Workload is unknown; database selection would be speculative.",),
                blockers=("Provide an explicit workload type or clarify the task.",),
                detected_options=(),
            )

        if kind == WorkloadType.INTERNAL_STATE:
            return DatabaseRecommendation(
                workload_type=kind,
                recommended_database="sqlite",
                eligible=drivers_by_db.get("sqlite", False),
                requires_user_approval=True,
                reasons=(
                    "SQLite is suitable for local Agent metadata and single-process state.",
                    "SQLite driver is available in the current Python environment.",
                ),
                blockers=(),
                detected_options=(
                    {
                        "database": "sqlite",
                        "driver_available": drivers_by_db.get("sqlite", False),
                        "service_required": False,
                        "service_detected": True,
                    },
                ),
            )

        if kind == WorkloadType.TRANSACTIONAL_READ:
            preference = ("postgresql", "mysql", "sqlserver", "oracle", "mongodb")
            reasons.append("Transactional business data should remain in its explicitly authorized business database.")
        elif kind == WorkloadType.ANALYTICS:
            preference = ("duckdb", "postgresql", "mysql", "sqlserver", "oracle", "mongodb")
            reasons.append("Analytical workloads may use a dedicated analytical engine or an authorized read-only source.")
        else:
            preference = ("postgresql", "mysql", "sqlserver", "oracle", "duckdb", "mongodb")
            reasons.append("Mixed workloads require an explicit integration design and per-source authorization.")

        for database in preference:
            driver_available = drivers_by_db.get(database, False)
            service_detected = database in running_services
            options.append(
                {
                    "database": database,
                    "driver_available": driver_available,
                    "service_detected": service_detected,
                    "ready_for_connection_attempt": bool(driver_available and service_detected),
                }
            )

        ready = [
            item for item in options
            if item["driver_available"] and item["service_detected"]
        ]

        if not ready:
            if not any(item["driver_available"] for item in options):
                blockers.append("No compatible Python driver is installed for the detected database choices.")
            if not any(item["service_detected"] for item in options):
                blockers.append("No matching running database service was detected.")
            blockers.append("Connection credentials, permissions, host, port, and schema have not been verified.")
            return DatabaseRecommendation(
                workload_type=kind,
                recommended_database=None,
                eligible=False,
                requires_user_approval=True,
                reasons=tuple(reasons),
                blockers=tuple(blockers),
                detected_options=tuple(options),
            )

        chosen = ready[0]["database"]
        reasons.append(
            f"{chosen} has both a detected Python driver and a detected running service."
        )
        blockers.append(
            "Detection does not prove that credentials, network access, schema, or read-only permissions work."
        )
        if kind in (WorkloadType.TRANSACTIONAL_READ, WorkloadType.MIXED):
            blockers.append(
                "The user must identify and authorize the exact business database; detected services are not presumed Agent-owned."
            )

        return DatabaseRecommendation(
            workload_type=kind,
            recommended_database=str(chosen),
            eligible=True,
            requires_user_approval=True,
            reasons=tuple(reasons),
            blockers=tuple(blockers),
            detected_options=tuple(options),
        )
