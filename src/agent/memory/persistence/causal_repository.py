from typing import Any, List, Optional

from agent.memory.causal import CausalMemory
from agent.memory.persistence.database import SQLiteDatabase


class CausalMemoryRepository:

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
                CREATE TABLE IF NOT EXISTS causal_memories (
                    memory_id TEXT PRIMARY KEY,

                    content TEXT NOT NULL,

                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,

                    source TEXT NOT NULL,

                    tags TEXT NOT NULL,

                    cause TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    effect TEXT NOT NULL,

                    condition_json TEXT,

                    evidence_count INTEGER NOT NULL,
                    contradiction_count INTEGER NOT NULL,

                    source_episodes TEXT NOT NULL
                )
                """
            )

            connection.commit()

    def save(
        self,
        memory: CausalMemory,
    ) -> None:
        import json

        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO causal_memories (
                    memory_id,
                    content,
                    importance,
                    confidence,
                    source,
                    tags,
                    cause,
                    relation,
                    effect,
                    condition_json,
                    evidence_count,
                    contradiction_count,
                    source_episodes
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                ON CONFLICT(memory_id)
                DO UPDATE SET
                    content = excluded.content,
                    importance = excluded.importance,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    tags = excluded.tags,
                    cause = excluded.cause,
                    relation = excluded.relation,
                    effect = excluded.effect,
                    condition_json = excluded.condition_json,
                    evidence_count = excluded.evidence_count,
                    contradiction_count =
                        excluded.contradiction_count,
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
                    memory.cause,
                    memory.relation,
                    memory.effect,
                    json.dumps(
                        memory.condition,
                        ensure_ascii=False,
                    ),
                    memory.evidence_count,
                    memory.contradiction_count,
                    json.dumps(
                        memory.source_episodes,
                        ensure_ascii=False,
                    ),
                ),
            )

            connection.commit()

    def save_many(
        self,
        memories: List[CausalMemory],
    ) -> None:
        for memory in memories:
            self.save(memory)

    def get(
        self,
        memory_id: str,
    ) -> Optional[CausalMemory]:
        import json

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM causal_memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return CausalMemory(
            memory_id=row["memory_id"],
            content=row["content"],
            importance=row["importance"],
            confidence=row["confidence"],
            source=row["source"],
            tags=json.loads(row["tags"]),
            cause=row["cause"],
            relation=row["relation"],
            effect=row["effect"],
            condition=(
                json.loads(row["condition_json"])
                if row["condition_json"] is not None
                else None
            ),
            evidence_count=row["evidence_count"],
            contradiction_count=row["contradiction_count"],
            source_episodes=json.loads(
                row["source_episodes"]
            ),
        )

    def all(self) -> List[CausalMemory]:
        import json

        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM causal_memories
                ORDER BY rowid
                """
            ).fetchall()

        return [
            CausalMemory(
                memory_id=row["memory_id"],
                content=row["content"],
                importance=row["importance"],
                confidence=row["confidence"],
                source=row["source"],
                tags=json.loads(row["tags"]),
                cause=row["cause"],
                relation=row["relation"],
                effect=row["effect"],
                condition=(
                    json.loads(row["condition_json"])
                    if row["condition_json"] is not None
                    else None
                ),
                evidence_count=row["evidence_count"],
                contradiction_count=row[
                    "contradiction_count"
                ],
                source_episodes=json.loads(
                    row["source_episodes"]
                ),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM causal_memories
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
                DELETE FROM causal_memories
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
                DELETE FROM causal_memories
                """
            )

            connection.commit()
