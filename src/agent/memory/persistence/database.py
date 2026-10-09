import sqlite3
from pathlib import Path
from typing import Optional


DEFAULT_DATABASE_PATH = Path("data/memory.db")


class SQLiteDatabase:
    def __init__(self, database_path: Optional[str | Path] = None):
        self.database_path = Path(
            database_path
            if database_path is not None
            else DEFAULT_DATABASE_PATH
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    memory_id TEXT PRIMARY KEY,

                    content TEXT NOT NULL,

                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,

                    source TEXT NOT NULL,

                    tags TEXT NOT NULL,

                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,

                    object_json TEXT NOT NULL,

                    evidence_count INTEGER NOT NULL,
                    contradiction_count INTEGER NOT NULL,

                    source_episodes TEXT NOT NULL,

                    context_json TEXT NOT NULL
                )
                """
            )

            connection.commit()
