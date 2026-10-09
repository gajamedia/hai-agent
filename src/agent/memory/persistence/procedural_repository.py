import json
from typing import Any, List, Optional

from agent.memory.persistence.database import (
    SQLiteDatabase,
)
from agent.memory.procedural import (
    ProceduralMemory,
)


class ProceduralMemoryRepository:

    def __init__(
        self,
        database: SQLiteDatabase,
    ):
        self.database = database
        self._initialize()

    def _initialize(self) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS procedural_memories (
                    memory_id TEXT PRIMARY KEY,

                    content TEXT NOT NULL,

                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,

                    source TEXT NOT NULL,

                    tags TEXT NOT NULL,

                    skill TEXT NOT NULL,
                    steps TEXT NOT NULL,

                    success_count INTEGER NOT NULL,
                    failure_count INTEGER NOT NULL,

                    source_episodes TEXT NOT NULL
                )
                """
            )

            connection.commit()

    def save(
        self,
        memory: ProceduralMemory,
    ) -> None:

        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO procedural_memories (
                    memory_id,
                    content,
                    importance,
                    confidence,
                    source,
                    tags,
                    skill,
                    steps,
                    success_count,
                    failure_count,
                    source_episodes
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )

                ON CONFLICT(memory_id)
                DO UPDATE SET
                    content = excluded.content,
                    importance = excluded.importance,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    tags = excluded.tags,
                    skill = excluded.skill,
                    steps = excluded.steps,
                    success_count =
                        excluded.success_count,
                    failure_count =
                        excluded.failure_count,
                    source_episodes =
                        excluded.source_episodes
                """,
                (
                    memory.memory_id,
                    memory.content,
                    memory.importance,
                    memory.confidence,
                    memory.source,
                    json.dumps(
                        memory.tags,
                        ensure_ascii=False,
                    ),
                    memory.skill,
                    json.dumps(
                        memory.steps,
                        ensure_ascii=False,
                    ),
                    memory.success_count,
                    memory.failure_count,
                    json.dumps(
                        memory.source_episodes,
                        ensure_ascii=False,
                    ),
                ),
            )

            connection.commit()

    def save_many(
        self,
        memories: List[ProceduralMemory],
    ) -> None:

        for memory in memories:
            self.save(memory)

    def get(
        self,
        memory_id: str,
    ) -> Optional[ProceduralMemory]:

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM procedural_memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return self._from_row(row)

    def all(self) -> List[ProceduralMemory]:

        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM procedural_memories
                ORDER BY rowid
                """
            ).fetchall()

        return [
            self._from_row(row)
            for row in rows
        ]

    def count(self) -> int:

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM procedural_memories
                """
            ).fetchone()

        return int(row[0])

    def delete(
        self,
        memory_id: str,
    ) -> bool:

        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM procedural_memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            )

            connection.commit()

            return cursor.rowcount > 0

    def clear(self) -> None:

        with self.database.connect() as connection:
            connection.execute(
                """
                DELETE FROM procedural_memories
                """
            )

            connection.commit()

    def _from_row(
        self,
        row: Any,
    ) -> ProceduralMemory:

        return ProceduralMemory(
            memory_id=row["memory_id"],
            content=row["content"],
            importance=row["importance"],
            confidence=row["confidence"],
            source=row["source"],
            tags=json.loads(
                row["tags"]
            ),
            skill=row["skill"],
            steps=json.loads(
                row["steps"]
            ),
            success_count=row[
                "success_count"
            ],
            failure_count=row[
                "failure_count"
            ],
            source_episodes=json.loads(
                row["source_episodes"]
            ),
        )
