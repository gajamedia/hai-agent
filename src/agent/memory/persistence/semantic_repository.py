import json
from typing import Any, Dict, List, Optional

from agent.memory.semantic import SemanticMemory
from agent.memory.persistence.database import SQLiteDatabase


class SemanticMemoryRepository:
    def __init__(self, database: SQLiteDatabase):
        self.database = database

    def save(self, memory: SemanticMemory) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO semantic_memories (
                    memory_id,
                    content,
                    importance,
                    confidence,
                    source,
                    tags,
                    subject,
                    predicate,
                    object_json,
                    evidence_count,
                    contradiction_count,
                    source_episodes,
                    context_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id)
                DO UPDATE SET
                    content = excluded.content,
                    importance = excluded.importance,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    tags = excluded.tags,
                    subject = excluded.subject,
                    predicate = excluded.predicate,
                    object_json = excluded.object_json,
                    evidence_count = excluded.evidence_count,
                    contradiction_count = excluded.contradiction_count,
                    source_episodes = excluded.source_episodes,
                    context_json = excluded.context_json
                """,
                (
                    memory.memory_id,
                    memory.content,
                    memory.importance,
                    memory.confidence,
                    memory.source,
                    json.dumps(memory.tags),
                    memory.subject,
                    memory.predicate,
                    json.dumps(
                        memory.object,
                        ensure_ascii=False,
                    ),
                    memory.evidence_count,
                    memory.contradiction_count,
                    json.dumps(memory.source_episodes),
                    json.dumps(
                        memory.context,
                        ensure_ascii=False,
                    ),
                ),
            )

            connection.commit()

    def save_many(
        self,
        memories: List[SemanticMemory],
    ) -> None:
        for memory in memories:
            self.save(memory)

    def get(
        self,
        memory_id: str,
    ) -> Optional[SemanticMemory]:

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM semantic_memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return self._from_row(row)

    def all(self) -> List[SemanticMemory]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM semantic_memories
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
                FROM semantic_memories
                """
            ).fetchone()

        return int(row[0])

    def delete(self, memory_id: str) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM semantic_memories
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
                DELETE FROM semantic_memories
                """
            )

            connection.commit()

    def _from_row(
        self,
        row: Any,
    ) -> SemanticMemory:

        return SemanticMemory(
            memory_id=row["memory_id"],
            content=row["content"],
            importance=row["importance"],
            confidence=row["confidence"],
            source=row["source"],
            tags=json.loads(row["tags"]),
            subject=row["subject"],
            predicate=row["predicate"],
            object=json.loads(row["object_json"]),
            evidence_count=row["evidence_count"],
            contradiction_count=row["contradiction_count"],
            source_episodes=json.loads(
                row["source_episodes"]
            ),
            context=json.loads(
                row["context_json"]
            ),
        )
