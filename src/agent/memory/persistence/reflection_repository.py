import json
from typing import Any, List, Optional

from agent.memory.persistence.database import SQLiteDatabase
from agent.memory.reflection import ReflectionMemory


class ReflectionMemoryRepository:

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
                CREATE TABLE IF NOT EXISTS reflection_memories (
                    memory_id TEXT PRIMARY KEY,

                    content TEXT NOT NULL,

                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,

                    source TEXT NOT NULL,

                    tags TEXT NOT NULL,

                    goal TEXT,
                    action TEXT,
                    problem TEXT,
                    cause TEXT,
                    lesson TEXT,

                    condition_json TEXT,

                    source_episodes TEXT NOT NULL,

                    applied INTEGER NOT NULL
                )
                """
            )

            connection.commit()

    def save(
        self,
        memory: ReflectionMemory,
    ) -> None:

        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO reflection_memories (
                    memory_id,
                    content,
                    importance,
                    confidence,
                    source,
                    tags,
                    goal,
                    action,
                    problem,
                    cause,
                    lesson,
                    condition_json,
                    source_episodes,
                    applied
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
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
                    problem = excluded.problem,
                    cause = excluded.cause,
                    lesson = excluded.lesson,
                    condition_json =
                        excluded.condition_json,
                    source_episodes =
                        excluded.source_episodes,
                    applied = excluded.applied
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
                    memory.problem,
                    memory.cause,
                    memory.lesson,
                    json.dumps(
                        memory.condition,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        memory.source_episodes,
                        ensure_ascii=False,
                    ),
                    1 if memory.applied else 0,
                ),
            )

            connection.commit()

    def save_many(
        self,
        memories: List[ReflectionMemory],
    ) -> None:

        for memory in memories:
            self.save(memory)

    def get(
        self,
        memory_id: str,
    ) -> Optional[ReflectionMemory]:

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM reflection_memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return self._from_row(row)

    def all(self) -> List[ReflectionMemory]:

        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM reflection_memories
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
                FROM reflection_memories
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
                DELETE FROM reflection_memories
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
                DELETE FROM reflection_memories
                """
            )

            connection.commit()

    def _from_row(
        self,
        row: Any,
    ) -> ReflectionMemory:

        condition_json = (
            row["condition_json"]
        )

        return ReflectionMemory(
            memory_id=row["memory_id"],
            content=row["content"],
            importance=row["importance"],
            confidence=row["confidence"],
            source=row["source"],
            tags=json.loads(
                row["tags"]
            ),
            goal=row["goal"],
            action=row["action"],
            problem=row["problem"],
            cause=row["cause"],
            lesson=row["lesson"],
            condition=(
                json.loads(condition_json)
                if condition_json is not None
                else None
            ),
            source_episodes=json.loads(
                row["source_episodes"]
            ),
            applied=bool(row["applied"]),
        )
