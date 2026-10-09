"""
Persistent repository for meta-learning profiles.

Stores MetaLearningProfile state in SQLite.

This repository is deliberately independent from the existing
MetaLearningStore so the in-memory implementation remains stable
while persistence is introduced incrementally.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from agent.learning.meta_learning import (
    MetaLearningProfile,
)


class MetaLearningRepository:
    """
    Persist MetaLearningProfile objects in SQLite.

    One row represents one agent profile.

    The repository owns only the persistence concern. It does not
    calculate learning behavior and does not modify the profile
    after loading.
    """

    TABLE_NAME = "meta_learning_profile"

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

        if (
            str(parent) != "."
            and not parent.exists()
        ):
            parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    # =====================================================
    # Database
    # =====================================================

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def _initialize(self) -> None:

        with self._connect() as connection:

            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                {self.TABLE_NAME} (
                    agent_name TEXT PRIMARY KEY,

                    exploration_count INTEGER NOT NULL,
                    exploration_success_count INTEGER NOT NULL,
                    exploration_failure_count INTEGER NOT NULL,

                    exploitation_count INTEGER NOT NULL,
                    exploitation_success_count INTEGER NOT NULL,
                    exploitation_failure_count INTEGER NOT NULL,

                    evolution_count INTEGER NOT NULL,
                    evolution_success_count INTEGER NOT NULL,
                    evolution_failure_count INTEGER NOT NULL,

                    recovery_count INTEGER NOT NULL,
                    recovery_success_count INTEGER NOT NULL,
                    recovery_failure_count INTEGER NOT NULL,

                    total_progress REAL NOT NULL,
                    progress_observation_count INTEGER NOT NULL,

                    learning_events TEXT NOT NULL
                )
                """
            )

    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        agent_name: str,
        profile: MetaLearningProfile,
    ) -> None:

        if not agent_name:
            raise ValueError(
                "agent_name cannot be empty."
            )

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        learning_events = json.dumps(
            list(profile.learning_events),
            ensure_ascii=False,
        )

        values = (
            agent_name,

            profile.exploration_count,
            profile.exploration_success_count,
            profile.exploration_failure_count,

            profile.exploitation_count,
            profile.exploitation_success_count,
            profile.exploitation_failure_count,

            profile.evolution_count,
            profile.evolution_success_count,
            profile.evolution_failure_count,

            profile.recovery_count,
            profile.recovery_success_count,
            profile.recovery_failure_count,

            profile.total_progress,
            profile.progress_observation_count,

            learning_events,
        )

        with self._connect() as connection:

            connection.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (
                    agent_name,

                    exploration_count,
                    exploration_success_count,
                    exploration_failure_count,

                    exploitation_count,
                    exploitation_success_count,
                    exploitation_failure_count,

                    evolution_count,
                    evolution_success_count,
                    evolution_failure_count,

                    recovery_count,
                    recovery_success_count,
                    recovery_failure_count,

                    total_progress,
                    progress_observation_count,

                    learning_events
                )
                VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?
                )
                ON CONFLICT(agent_name)
                DO UPDATE SET

                    exploration_count =
                        excluded.exploration_count,

                    exploration_success_count =
                        excluded.exploration_success_count,

                    exploration_failure_count =
                        excluded.exploration_failure_count,

                    exploitation_count =
                        excluded.exploitation_count,

                    exploitation_success_count =
                        excluded.exploitation_success_count,

                    exploitation_failure_count =
                        excluded.exploitation_failure_count,

                    evolution_count =
                        excluded.evolution_count,

                    evolution_success_count =
                        excluded.evolution_success_count,

                    evolution_failure_count =
                        excluded.evolution_failure_count,

                    recovery_count =
                        excluded.recovery_count,

                    recovery_success_count =
                        excluded.recovery_success_count,

                    recovery_failure_count =
                        excluded.recovery_failure_count,

                    total_progress =
                        excluded.total_progress,

                    progress_observation_count =
                        excluded.progress_observation_count,

                    learning_events =
                        excluded.learning_events
                """,
                values,
            )

    # =====================================================
    # Load
    # =====================================================

    def get(
        self,
        agent_name: str,
    ) -> Optional[MetaLearningProfile]:

        if not agent_name:
            raise ValueError(
                "agent_name cannot be empty."
            )

        with self._connect() as connection:

            row = connection.execute(
                f"""
                SELECT *
                FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                """,
                (agent_name,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_profile(
            row
        )

    # =====================================================
    # Existence
    # =====================================================

    def exists(
        self,
        agent_name: str,
    ) -> bool:

        if not agent_name:
            raise ValueError(
                "agent_name cannot be empty."
            )

        with self._connect() as connection:

            row = connection.execute(
                f"""
                SELECT 1
                FROM {self.TABLE_NAME}
                WHERE agent_name = ?
                LIMIT 1
                """,
                (agent_name,),
            ).fetchone()

        return row is not None

    # =====================================================
    # Delete
    # =====================================================

    def delete(
        self,
        agent_name: str,
    ) -> bool:

        if not agent_name:
            raise ValueError(
                "agent_name cannot be empty."
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

    # =====================================================
    # Count
    # =====================================================

    def count(self) -> int:

        with self._connect() as connection:

            row = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {self.TABLE_NAME}
                """
            ).fetchone()

        return int(row[0])

    # =====================================================
    # Clear
    # =====================================================

    def clear(self) -> None:

        with self._connect() as connection:

            connection.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                """
            )

    # =====================================================
    # Serialization
    # =====================================================

    @staticmethod
    def _row_to_profile(
        row: sqlite3.Row,
    ) -> MetaLearningProfile:

        try:
            learning_events = json.loads(
                row["learning_events"]
            )

        except (
            TypeError,
            json.JSONDecodeError,
        ) as exc:

            raise ValueError(
                "Invalid persisted learning_events."
            ) from exc

        if not isinstance(
            learning_events,
            list,
        ):
            raise ValueError(
                "Persisted learning_events must be a list."
            )

        profile = MetaLearningProfile()

        profile.exploration_count = (
            int(row["exploration_count"])
        )

        profile.exploration_success_count = (
            int(
                row[
                    "exploration_success_count"
                ]
            )
        )

        profile.exploration_failure_count = (
            int(
                row[
                    "exploration_failure_count"
                ]
            )
        )

        profile.exploitation_count = (
            int(row["exploitation_count"])
        )

        profile.exploitation_success_count = (
            int(
                row[
                    "exploitation_success_count"
                ]
            )
        )

        profile.exploitation_failure_count = (
            int(
                row[
                    "exploitation_failure_count"
                ]
            )
        )

        profile.evolution_count = (
            int(row["evolution_count"])
        )

        profile.evolution_success_count = (
            int(
                row[
                    "evolution_success_count"
                ]
            )
        )

        profile.evolution_failure_count = (
            int(
                row[
                    "evolution_failure_count"
                ]
            )
        )

        profile.recovery_count = (
            int(row["recovery_count"])
        )

        profile.recovery_success_count = (
            int(
                row[
                    "recovery_success_count"
                ]
            )
        )

        profile.recovery_failure_count = (
            int(
                row[
                    "recovery_failure_count"
                ]
            )
        )

        profile.total_progress = float(
            row["total_progress"]
        )

        profile.progress_observation_count = (
            int(
                row[
                    "progress_observation_count"
                ]
            )
        )

        profile.learning_events = list(
            learning_events
        )

        return profile
