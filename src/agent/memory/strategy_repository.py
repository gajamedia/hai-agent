from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from agent.learning.strategy import Strategy


class StrategyRepository:
    """
    SQLite repository for persistent Strategy objects.

    This repository owns only the `strategies` table and does
    not modify the schema of other memory repositories.
    """

    def __init__(
        self,
        database_path: str | Path,
    ):
        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    # =================================================
    # Database
    # =================================================

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize(self) -> None:

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    steps TEXT NOT NULL,

                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    total_actions INTEGER NOT NULL DEFAULT 0,

                    reflection_penalty REAL NOT NULL DEFAULT 0.0,

                    source_experiences TEXT NOT NULL DEFAULT '[]',

                    total_progress REAL NOT NULL DEFAULT 0.0,
                    progress_count INTEGER NOT NULL DEFAULT 0,

                    positive_progress_count INTEGER NOT NULL DEFAULT 0,
                    negative_progress_count INTEGER NOT NULL DEFAULT 0,

                    status TEXT NOT NULL DEFAULT 'active',

                    PRIMARY KEY (name, goal, steps)
                )
                """
            )

            connection.commit()

    # =================================================
    # Save
    # =================================================

    def save(
        self,
        strategy: Strategy,
    ) -> None:

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        steps_json = json.dumps(
            list(strategy.steps),
            ensure_ascii=False,
        )

        experiences_json = json.dumps(
            list(
                strategy.source_experiences
            ),
            ensure_ascii=False,
        )

        with self._connect() as connection:

            connection.execute(
                """
                INSERT INTO strategies (
                    name,
                    goal,
                    steps,
                    success_count,
                    failure_count,
                    total_actions,
                    reflection_penalty,
                    source_experiences,
                    total_progress,
                    progress_count,
                    positive_progress_count,
                    negative_progress_count,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name, goal, steps)
                DO UPDATE SET
                    success_count = excluded.success_count,
                    failure_count = excluded.failure_count,
                    total_actions = excluded.total_actions,
                    reflection_penalty = excluded.reflection_penalty,
                    source_experiences = excluded.source_experiences,
                    total_progress = excluded.total_progress,
                    progress_count = excluded.progress_count,
                    positive_progress_count =
                        excluded.positive_progress_count,
                    negative_progress_count =
                        excluded.negative_progress_count,
                    status = excluded.status
                """,
                (
                    strategy.name,
                    strategy.goal,
                    steps_json,
                    strategy.success_count,
                    strategy.failure_count,
                    strategy.total_actions,
                    strategy.reflection_penalty,
                    experiences_json,
                    strategy.total_progress,
                    strategy.progress_count,
                    strategy.positive_progress_count,
                    strategy.negative_progress_count,
                    strategy.status,
                ),
            )

            connection.commit()

    # =================================================
    # Delete
    # =================================================

    def delete(
        self,
        strategy: Strategy,
    ) -> None:

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        steps_json = json.dumps(
            list(strategy.steps),
            ensure_ascii=False,
        )

        with self._connect() as connection:

            connection.execute(
                """
                DELETE FROM strategies
                WHERE name = ?
                  AND goal = ?
                  AND steps = ?
                """,
                (
                    strategy.name,
                    strategy.goal,
                    steps_json,
                ),
            )

            connection.commit()

    # =================================================
    # Load all
    # =================================================

    def load_all(self) -> list[Strategy]:

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT
                    name,
                    goal,
                    steps,
                    success_count,
                    failure_count,
                    total_actions,
                    reflection_penalty,
                    source_experiences,
                    total_progress,
                    progress_count,
                    positive_progress_count,
                    negative_progress_count,
                    status
                FROM strategies
                ORDER BY name
                """
            ).fetchall()

        strategies: list[Strategy] = []

        for row in rows:

            strategies.append(
                Strategy(
                    name=row["name"],
                    goal=row["goal"],
                    steps=json.loads(
                        row["steps"]
                    ),
                    success_count=row[
                        "success_count"
                    ],
                    failure_count=row[
                        "failure_count"
                    ],
                    total_actions=row[
                        "total_actions"
                    ],
                    reflection_penalty=row[
                        "reflection_penalty"
                    ],
                    source_experiences=json.loads(
                        row[
                            "source_experiences"
                        ]
                    ),
                    total_progress=row[
                        "total_progress"
                    ],
                    progress_count=row[
                        "progress_count"
                    ],
                    positive_progress_count=row[
                        "positive_progress_count"
                    ],
                    negative_progress_count=row[
                        "negative_progress_count"
                    ],
                    status=row[
                        "status"
                    ],
                )
            )

        return strategies

    # =================================================
    # Find
    # =================================================

    def find(
        self,
        name: str,
        goal: str,
        steps: list[str],
    ) -> Optional[Strategy]:

        steps_json = json.dumps(
            list(steps),
            ensure_ascii=False,
        )

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT
                    name,
                    goal,
                    steps,
                    success_count,
                    failure_count,
                    total_actions,
                    reflection_penalty,
                    source_experiences,
                    total_progress,
                    progress_count,
                    positive_progress_count,
                    negative_progress_count,
                    status
                FROM strategies
                WHERE name = ?
                  AND goal = ?
                  AND steps = ?
                """,
                (
                    name,
                    goal,
                    steps_json,
                ),
            ).fetchone()

        if row is None:
            return None

        return Strategy(
            name=row["name"],
            goal=row["goal"],
            steps=json.loads(
                row["steps"]
            ),
            success_count=row[
                "success_count"
            ],
            failure_count=row[
                "failure_count"
            ],
            total_actions=row[
                "total_actions"
            ],
            reflection_penalty=row[
                "reflection_penalty"
            ],
            source_experiences=json.loads(
                row["source_experiences"]
            ),
            total_progress=row[
                "total_progress"
            ],
            progress_count=row[
                "progress_count"
            ],
            positive_progress_count=row[
                "positive_progress_count"
            ],
            negative_progress_count=row[
                "negative_progress_count"
            ],
            status=row[
                "status"
            ],
        )

    # =================================================
    # Clear
    # =================================================

    def clear(self) -> None:

        with self._connect() as connection:

            connection.execute(
                "DELETE FROM strategies"
            )

            connection.commit()
