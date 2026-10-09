import json
from typing import Any, List, Optional

from agent.memory.episodic import EpisodicMemory
from agent.memory.persistence.database import SQLiteDatabase


class EpisodicMemoryRepository:

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
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    memory_id TEXT PRIMARY KEY,

                    content TEXT NOT NULL,

                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,

                    source TEXT NOT NULL,

                    tags TEXT NOT NULL,

                    goal TEXT,

                    action TEXT,

                    observation_before TEXT,

                    observation_after TEXT,

                    result TEXT,

                    cycle INTEGER
                )
                """
            )

            connection.commit()

    def save(
        self,
        memory: EpisodicMemory,
    ) -> None:

        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO episodic_memories (
                    memory_id,
                    content,
                    importance,
                    confidence,
                    source,
                    tags,
                    goal,
                    action,
                    observation_before,
                    observation_after,
                    result,
                    cycle
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )

                ON CONFLICT(memory_id)
                DO UPDATE SET
                    content = excluded.content,
                    importance = excluded.importance,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    tags = excluded.tags,
                    goal = excluded.goal,
                    action = excluded.action,
                    observation_before =
                        excluded.observation_before,
                    observation_after =
                        excluded.observation_after,
                    result = excluded.result,
                    cycle = excluded.cycle
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
                    memory.goal,
                    memory.action,
                    json.dumps(
                        memory.observation_before,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        memory.observation_after,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        memory.result,
                        ensure_ascii=False,
                    ),
                    memory.cycle,
                ),
            )

            connection.commit()

    def save_many(
        self,
        memories: List[EpisodicMemory],
    ) -> None:

        for memory in memories:
            self.save(memory)

    def get(
        self,
        memory_id: str,
    ) -> Optional[EpisodicMemory]:

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM episodic_memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return self._from_row(row)

    def all(self) -> List[EpisodicMemory]:

        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM episodic_memories
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
                FROM episodic_memories
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
                DELETE FROM episodic_memories
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
                DELETE FROM episodic_memories
                """
            )

            connection.commit()

    def _from_row(
        self,
        row: Any,
    ) -> EpisodicMemory:

        return EpisodicMemory(
            memory_id=row["memory_id"],
            content=row["content"],
            importance=row["importance"],
            confidence=row["confidence"],
            source=row["source"],
            tags=json.loads(row["tags"]),
            goal=row["goal"],
            action=row["action"],
            observation_before=json.loads(
                row["observation_before"]
            )
            if row["observation_before"] is not None
            else None,
            observation_after=json.loads(
                row["observation_after"]
            )
            if row["observation_after"] is not None
            else None,
            result=json.loads(
                row["result"]
            )
            if row["result"] is not None
            else None,
            cycle=row["cycle"],
        )
