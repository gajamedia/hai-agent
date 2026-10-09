from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SELF_MODEL_MODE_NEUTRAL,
    SelfModelDecisionInfluence,
)

from agent.learning.self_uncertainty import (
    SelfUncertaintyResult,
)


# ============================================================
# Persistence Snapshot
# ============================================================


@dataclass(frozen=True)
class SelfModelContinuitySnapshot:
    """
    Persisted self-awareness state.

    The snapshot contains the aggregate self-uncertainty result
    and the decision influence derived from it.
    """

    agent_name: str
    self_uncertainty: SelfUncertaintyResult
    decision_influence: (
        SelfModelDecisionInfluence | None
    )
    version: int
    updated_at: str

    def snapshot(self) -> dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "self_uncertainty": (
                self.self_uncertainty.snapshot()
            ),
            "decision_influence": (
                self.decision_influence.snapshot()
                if self.decision_influence is not None
                else None
            ),
            "version": self.version,
            "updated_at": self.updated_at,
        }


# ============================================================
# Persistent Store
# ============================================================


class PersistentSelfModelStore:
    """
    SQLite-backed persistence for self-model continuity.

    This store is intentionally independent from the existing
    SelfModelStore and PersistentMemory implementations.

    It persists the derived self-awareness state produced by:

        Capability Confidence
            ↓
        Self Uncertainty
            ↓
        Decision Influence
    """

    TABLE_NAME = "self_model_continuity"

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:

        if database_path is None:
            raise ValueError(
                "database_path cannot be None."
            )

        self.database_path = Path(
            database_path
        )

        parent = (
            self.database_path.parent
        )

        if str(parent):
            parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize_database()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def save(
        self,
        *,
        agent_name: str,
        self_uncertainty: SelfUncertaintyResult,
        decision_influence: (
            SelfModelDecisionInfluence | None
        ) = None,
    ) -> SelfModelContinuitySnapshot:

        self._validate_agent_name(
            agent_name
        )

        if self_uncertainty is None:
            raise ValueError(
                "self_uncertainty cannot be None."
            )

        if not isinstance(
            self_uncertainty,
            SelfUncertaintyResult,
        ):
            raise ValueError(
                "self_uncertainty must be "
                "SelfUncertaintyResult."
            )

        if (
            decision_influence is not None
            and not isinstance(
                decision_influence,
                SelfModelDecisionInfluence,
            )
        ):
            raise ValueError(
                "decision_influence must be "
                "SelfModelDecisionInfluence or None."
            )

        updated_at = (
            self._utc_now()
        )

        payload = {
            "agent_name": agent_name,
            "self_uncertainty": (
                self_uncertainty.snapshot()
            ),
            "decision_influence": (
                decision_influence.snapshot()
                if decision_influence is not None
                else None
            ),
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                SELECT version
                FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                """,
                (agent_name,),
            )

            row = cursor.fetchone()

            if row is None:
                version = 1

                connection.execute(
                    f"""
                    INSERT INTO {self.TABLE_NAME}
                    (
                        agent_name,
                        payload,
                        version,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        agent_name,
                        serialized,
                        version,
                        updated_at,
                    ),
                )

            else:
                version = (
                    int(row[0])
                    + 1
                )

                connection.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET
                        payload = ?,
                        version = ?,
                        updated_at = ?
                    WHERE agent_name = ?
                    """,
                    (
                        serialized,
                        version,
                        updated_at,
                        agent_name,
                    ),
                )

        return SelfModelContinuitySnapshot(
            agent_name=agent_name,
            self_uncertainty=(
                self_uncertainty
            ),
            decision_influence=(
                decision_influence
            ),
            version=version,
            updated_at=updated_at,
        )

    def load(
        self,
        agent_name: str,
    ) -> SelfModelContinuitySnapshot | None:

        self._validate_agent_name(
            agent_name
        )

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                SELECT
                    payload,
                    version,
                    updated_at
                FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                """,
                (agent_name,),
            )

            row = cursor.fetchone()

        if row is None:
            return None

        payload_text = row[0]
        version = int(row[1])
        updated_at = str(row[2])

        try:
            payload = json.loads(
                payload_text
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Persisted self-model state "
                "contains invalid JSON."
            ) from exc

        return self._deserialize(
            payload=payload,
            version=version,
            updated_at=updated_at,
        )

    def exists(
        self,
        agent_name: str,
    ) -> bool:

        self._validate_agent_name(
            agent_name
        )

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                SELECT 1
                FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                LIMIT 1
                """,
                (agent_name,),
            )

            return cursor.fetchone() is not None

    def delete(
        self,
        agent_name: str,
    ) -> bool:

        self._validate_agent_name(
            agent_name
        )

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                """,
                (agent_name,),
            )

            return cursor.rowcount > 0

    def list_agent_names(
        self,
    ) -> tuple[str, ...]:

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                SELECT agent_name
                FROM {self.TABLE_NAME}
                ORDER BY agent_name
                """
            )

            rows = cursor.fetchall()

        return tuple(
            str(row[0])
            for row in rows
        )

    def count(self) -> int:

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {self.TABLE_NAME}
                """
            )

            row = cursor.fetchone()

        return int(
            row[0]
        )

    def clear(self) -> int:

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                """
            )

            return cursor.rowcount

    # --------------------------------------------------------
    # Convenience API
    # --------------------------------------------------------

    def save_results(
        self,
        *,
        agent_name: str,
        self_uncertainty: SelfUncertaintyResult,
        decision_influence: (
            SelfModelDecisionInfluence | None
        ) = None,
    ) -> SelfModelContinuitySnapshot:

        return self.save(
            agent_name=agent_name,
            self_uncertainty=self_uncertainty,
            decision_influence=(
                decision_influence
            ),
        )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    def _initialize_database(
        self,
    ) -> None:

        with self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                {self.TABLE_NAME}
                (
                    agent_name TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(
        self,
    ) -> sqlite3.Connection:

        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    @classmethod
    def _deserialize(
        cls,
        *,
        payload: dict[str, Any],
        version: int,
        updated_at: str,
    ) -> SelfModelContinuitySnapshot:

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Persisted self-model payload "
                "must be a dictionary."
            )

        agent_name = payload.get(
            "agent_name"
        )

        if not isinstance(
            agent_name,
            str,
        ) or not agent_name.strip():
            raise ValueError(
                "Persisted self-model payload "
                "contains an invalid agent_name."
            )

        uncertainty_data = payload.get(
            "self_uncertainty"
        )

        if not isinstance(
            uncertainty_data,
            dict,
        ):
            raise ValueError(
                "Persisted self-model payload "
                "contains invalid self_uncertainty data."
            )

        self_uncertainty = (
            cls._deserialize_uncertainty(
                uncertainty_data
            )
        )

        influence_data = payload.get(
            "decision_influence"
        )

        decision_influence = None

        if influence_data is not None:
            if not isinstance(
                influence_data,
                dict,
            ):
                raise ValueError(
                    "Persisted decision influence "
                    "must be a dictionary or None."
                )

            decision_influence = (
                cls._deserialize_influence(
                    influence_data
                )
            )

        return SelfModelContinuitySnapshot(
            agent_name=agent_name,
            self_uncertainty=(
                self_uncertainty
            ),
            decision_influence=(
                decision_influence
            ),
            version=version,
            updated_at=updated_at,
        )

    @staticmethod
    def _deserialize_uncertainty(
        data: dict[str, Any],
    ) -> SelfUncertaintyResult:

        required_fields = (
            "uncertainty",
            "confidence",
            "evidence_strength",
            "average_capability_uncertainty",
            "average_capability_confidence",
            "confidence_dispersion",
            "capability_count",
            "observed_capability_count",
            "high_uncertainty_count",
            "low_confidence_count",
            "strongest_capability",
            "weakest_capability",
        )

        for field in required_fields:
            if field not in data:
                raise ValueError(
                    "Persisted self-uncertainty "
                    f"is missing field: {field}"
                )

        return SelfUncertaintyResult(
            uncertainty=float(
                data["uncertainty"]
            ),
            confidence=float(
                data["confidence"]
            ),
            evidence_strength=float(
                data["evidence_strength"]
            ),
            average_capability_uncertainty=float(
                data[
                    "average_capability_uncertainty"
                ]
            ),
            average_capability_confidence=float(
                data[
                    "average_capability_confidence"
                ]
            ),
            confidence_dispersion=float(
                data["confidence_dispersion"]
            ),
            capability_count=int(
                data["capability_count"]
            ),
            observed_capability_count=int(
                data[
                    "observed_capability_count"
                ]
            ),
            high_uncertainty_count=int(
                data["high_uncertainty_count"]
            ),
            low_confidence_count=int(
                data["low_confidence_count"]
            ),
            strongest_capability=(
                data["strongest_capability"]
            ),
            weakest_capability=(
                data["weakest_capability"]
            ),
        )

    @staticmethod
    def _deserialize_influence(
        data: dict[str, Any],
    ) -> SelfModelDecisionInfluence:

        required_fields = (
            "preferred_mode",
            "explore_bonus",
            "exploit_bonus",
            "confidence",
            "uncertainty",
            "strength",
            "applied",
            "reason",
        )

        for field in required_fields:
            if field not in data:
                raise ValueError(
                    "Persisted decision influence "
                    f"is missing field: {field}"
                )

        preferred_mode = str(
            data["preferred_mode"]
        )

        valid_modes = {
            SELF_MODEL_MODE_EXPLORE,
            SELF_MODEL_MODE_EXPLOIT,
            SELF_MODEL_MODE_NEUTRAL,
        }

        if preferred_mode not in valid_modes:
            raise ValueError(
                "Persisted decision influence "
                "contains an invalid preferred_mode."
            )

        return SelfModelDecisionInfluence(
            preferred_mode=preferred_mode,
            explore_bonus=float(
                data["explore_bonus"]
            ),
            exploit_bonus=float(
                data["exploit_bonus"]
            ),
            confidence=float(
                data["confidence"]
            ),
            uncertainty=float(
                data["uncertainty"]
            ),
            strength=float(
                data["strength"]
            ),
            applied=bool(
                data["applied"]
            ),
            reason=str(
                data["reason"]
            ),
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_agent_name(
        agent_name: str,
    ) -> None:

        if not isinstance(
            agent_name,
            str,
        ):
            raise ValueError(
                "agent_name must be a string."
            )

        if not agent_name.strip():
            raise ValueError(
                "agent_name cannot be empty."
            )

    @staticmethod
    def _utc_now() -> str:

        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )


# ============================================================
# Convenience Builder
# ============================================================


def build_persistent_self_model_store(
    database_path: str | Path,
) -> PersistentSelfModelStore:

    return PersistentSelfModelStore(
        database_path
    )
