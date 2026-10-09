from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.learning.calibrated_self_awareness import (
    CalibratedSelfAwareInfluenceResult,
)
from agent.learning.self_model_decision_influence import (
    SelfModelDecisionInfluence,
)


@dataclass(frozen=True)
class CalibratedSelfAwarenessPersistenceSnapshot:
    """
    Persisted calibrated self-awareness state.
    """

    agent_name: str
    influence: SelfModelDecisionInfluence
    calibration_result: CalibratedSelfAwareInfluenceResult
    version: int
    updated_at: str

    @property
    def snapshot(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "influence": {
                "preferred_mode": (
                    self.influence.preferred_mode
                ),
                "explore_bonus": (
                    self.influence.explore_bonus
                ),
                "exploit_bonus": (
                    self.influence.exploit_bonus
                ),
                "confidence": (
                    self.influence.confidence
                ),
                "uncertainty": (
                    self.influence.uncertainty
                ),
                "strength": (
                    self.influence.strength
                ),
                "applied": (
                    self.influence.applied
                ),
                "reason": (
                    self.influence.reason
                ),
            },
            "calibration_result": (
                self.calibration_result.snapshot
            ),
            "version": self.version,
            "updated_at": self.updated_at,
        }


class PersistentCalibratedSelfAwarenessStore:
    """
    SQLite-backed persistence for calibrated self-awareness.

    This storage is deliberately independent from the existing
    self-model continuity store.

    One row represents one Agent's latest calibrated
    self-awareness state.
    """

    TABLE_NAME = (
        "calibrated_self_awareness"
    )

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    # =====================================================
    # SQLite
    # =====================================================

    def _connect(
        self,
    ) -> sqlite3.Connection:

        connection = sqlite3.connect(
            self.path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def _initialize(
        self,
    ) -> None:

        with self._connect() as connection:

            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                {self.TABLE_NAME} (
                    agent_name TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    # =====================================================
    # Serialization
    # =====================================================

    @staticmethod
    def _serialize_influence(
        influence: SelfModelDecisionInfluence,
    ) -> dict[str, Any]:

        return {
            "preferred_mode": (
                influence.preferred_mode
            ),
            "explore_bonus": (
                influence.explore_bonus
            ),
            "exploit_bonus": (
                influence.exploit_bonus
            ),
            "confidence": (
                influence.confidence
            ),
            "uncertainty": (
                influence.uncertainty
            ),
            "strength": (
                influence.strength
            ),
            "applied": (
                influence.applied
            ),
            "reason": (
                influence.reason
            ),
        }

    @staticmethod
    def _serialize_calibration(
        result: CalibratedSelfAwareInfluenceResult,
    ) -> dict[str, Any]:

        return result.snapshot

    # =====================================================
    # Deserialization
    # =====================================================

    @staticmethod
    def _deserialize_influence(
        payload: dict[str, Any],
    ) -> SelfModelDecisionInfluence:

        return SelfModelDecisionInfluence(
            preferred_mode=payload[
                "preferred_mode"
            ],
            explore_bonus=float(
                payload[
                    "explore_bonus"
                ]
            ),
            exploit_bonus=float(
                payload[
                    "exploit_bonus"
                ]
            ),
            confidence=float(
                payload[
                    "confidence"
                ]
            ),
            uncertainty=float(
                payload[
                    "uncertainty"
                ]
            ),
            strength=float(
                payload[
                    "strength"
                ]
            ),
            applied=bool(
                payload[
                    "applied"
                ]
            ),
            reason=str(
                payload.get(
                    "reason",
                    "",
                )
            ),
        )

    @staticmethod
    def _deserialize_calibration(
        payload: dict[str, Any],
        influence: SelfModelDecisionInfluence,
    ) -> CalibratedSelfAwareInfluenceResult:
        return CalibratedSelfAwareInfluenceResult(
            influence=influence,
            raw_confidence=float(
                payload[
                    "raw_confidence"
                ]
            ),
            calibrated_confidence=float(
                payload[
                    "calibrated_confidence"
                ]
            ),
            raw_uncertainty=float(
                payload[
                    "raw_uncertainty"
                ]
            ),
            calibrated_uncertainty=float(
                payload[
                    "calibrated_uncertainty"
                ]
            ),
            confidence_delta=float(
                payload[
                    "confidence_delta"
                ]
            ),
            raw_explore_bonus=float(
                payload[
                    "raw_explore_bonus"
                ]
            ),
            calibrated_explore_bonus=float(
                payload[
                    "calibrated_explore_bonus"
                ]
            ),
            raw_exploit_bonus=float(
                payload[
                    "raw_exploit_bonus"
                ]
            ),
            calibrated_exploit_bonus=float(
                payload[
                    "calibrated_exploit_bonus"
                ]
            ),
            adjusted=bool(
                payload[
                    "adjusted"
                ]
            ),
        )

    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        *,
        agent_name: str,
        influence: SelfModelDecisionInfluence,
        calibration_result: (
            CalibratedSelfAwareInfluenceResult
        ),
    ) -> CalibratedSelfAwarenessPersistenceSnapshot:

        if not agent_name or not agent_name.strip():
            raise ValueError(
                "agent_name cannot be empty."
            )

        if not isinstance(
            influence,
            SelfModelDecisionInfluence,
        ):
            raise TypeError(
                "influence must be a "
                "SelfModelDecisionInfluence."
            )

        if not isinstance(
            calibration_result,
            CalibratedSelfAwareInfluenceResult,
        ):
            raise TypeError(
                "calibration_result must be a "
                "CalibratedSelfAwareInfluenceResult."
            )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with self._connect() as connection:

            existing = connection.execute(
                f"""
                SELECT version
                FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                """,
                (
                    agent_name,
                ),
            ).fetchone()

            version = (
                int(
                    existing["version"]
                )
                + 1
                if existing is not None
                else 1
            )

            payload = {
                "influence": (
                    self._serialize_influence(
                        influence
                    )
                ),
                "calibration_result": (
                    self._serialize_calibration(
                        calibration_result
                    )
                ),
            }

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
                ON CONFLICT(agent_name)
                DO UPDATE SET
                    payload = excluded.payload,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (
                    agent_name,
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                    version,
                    now,
                ),
            )

        return CalibratedSelfAwarenessPersistenceSnapshot(
            agent_name=agent_name,
            influence=influence,
            calibration_result=calibration_result,
            version=version,
            updated_at=now,
        )

    # =====================================================
    # Load
    # =====================================================

    def load(
        self,
        agent_name: str,
    ) -> (
        CalibratedSelfAwarenessPersistenceSnapshot
        | None
    ):

        with self._connect() as connection:

            row = connection.execute(
                f"""
                SELECT
                    agent_name,
                    payload,
                    version,
                    updated_at
                FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                """,
                (
                    agent_name,
                ),
            ).fetchone()

        if row is None:
            return None

        payload = json.loads(
            row["payload"]
        )

        influence = (
            self._deserialize_influence(
                payload["influence"]
            )
        )

        calibration_result = (
            self._deserialize_calibration(
                payload[
                    "calibration_result"
                ],
                influence,
            )
        )

        return (
            CalibratedSelfAwarenessPersistenceSnapshot(
                agent_name=row[
                    "agent_name"
                ],
                influence=influence,
                calibration_result=(
                    calibration_result
                ),
                version=int(
                    row["version"]
                ),
                updated_at=row[
                    "updated_at"
                ],
            )
        )

    # =====================================================
    # Existence
    # =====================================================

    def exists(
        self,
        agent_name: str,
    ) -> bool:

        with self._connect() as connection:

            row = connection.execute(
                f"""
                SELECT 1
                FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                LIMIT 1
                """,
                (
                    agent_name,
                ),
            ).fetchone()

        return row is not None

    # =====================================================
    # Delete
    # =====================================================

    def delete(
        self,
        agent_name: str,
    ) -> bool:

        with self._connect() as connection:

            cursor = connection.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                """,
                (
                    agent_name,
                ),
            )

        return (
            cursor.rowcount > 0
        )

    # =====================================================
    # Count
    # =====================================================

    def count(
        self,
    ) -> int:

        with self._connect() as connection:

            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM {self.TABLE_NAME}
                """
            ).fetchone()

        return int(
            row["count"]
        )

    # =====================================================
    # List
    # =====================================================

    def list_agent_names(
        self,
    ) -> list[str]:

        with self._connect() as connection:

            rows = connection.execute(
                f"""
                SELECT agent_name
                FROM {self.TABLE_NAME}
                ORDER BY agent_name
                """
            ).fetchall()

        return [
            row["agent_name"]
            for row in rows
        ]

    # =====================================================
    # Clear
    # =====================================================

    def clear(
        self,
    ) -> None:

        with self._connect() as connection:

            connection.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                """
            )
