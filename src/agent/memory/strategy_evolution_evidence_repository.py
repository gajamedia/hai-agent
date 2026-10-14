
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.learning.strategy_evolution_evidence import (
    StrategyEvolutionEvidence,
)


class StrategyEvolutionEvidenceRepository:
    """SQLite persistence for strategy evolution evidence."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_evolution_evidence (
                    strategy_name TEXT PRIMARY KEY,
                    source_strategy_name TEXT NOT NULL,
                    source_experience_id TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 1,
                    trial_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    total_progress REAL NOT NULL DEFAULT 0.0,
                    source_experience_ids TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'experimental'
                )
                """
            )

    def save(self, evidence: StrategyEvolutionEvidence) -> None:
        if evidence is None:
            raise ValueError("evidence cannot be None.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO strategy_evolution_evidence (
                    strategy_name,
                    source_strategy_name,
                    source_experience_id,
                    generation,
                    trial_count,
                    success_count,
                    failure_count,
                    total_progress,
                    source_experience_ids,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_name) DO UPDATE SET
                    source_strategy_name = excluded.source_strategy_name,
                    source_experience_id = excluded.source_experience_id,
                    generation = excluded.generation,
                    trial_count = excluded.trial_count,
                    success_count = excluded.success_count,
                    failure_count = excluded.failure_count,
                    total_progress = excluded.total_progress,
                    source_experience_ids = excluded.source_experience_ids,
                    status = excluded.status
                """,
                (
                    evidence.strategy_name,
                    evidence.source_strategy_name,
                    evidence.source_experience_id,
                    evidence.generation,
                    evidence.trial_count,
                    evidence.success_count,
                    evidence.failure_count,
                    evidence.total_progress,
                    json.dumps(
                        evidence.source_experience_ids,
                        ensure_ascii=False,
                    ),
                    evidence.status,
                ),
            )

    def load_all(self) -> list[StrategyEvolutionEvidence]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM strategy_evolution_evidence
                ORDER BY strategy_name
                """
            ).fetchall()

        return [
            StrategyEvolutionEvidence(
                strategy_name=row["strategy_name"],
                source_strategy_name=row["source_strategy_name"],
                source_experience_id=row["source_experience_id"],
                generation=row["generation"],
                trial_count=row["trial_count"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                total_progress=row["total_progress"],
                source_experience_ids=json.loads(
                    row["source_experience_ids"]
                ),
                status=row["status"],
            )
            for row in rows
        ]

    def delete(self, strategy_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM strategy_evolution_evidence
                WHERE strategy_name = ?
                """,
                (strategy_name,),
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM strategy_evolution_evidence"
            )
