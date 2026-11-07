from __future__ import annotations

"""Creating a new database (and its tables) without letting an LLM write DDL.

The LLM only chooses a *structured* design: a database name and tables whose columns use a small closed
set of types. Our code validates it (the name must start with the configured prefix, nothing may already
exist under that name), shows it, and only then builds the CREATE statements. There is no way to drop,
alter or overwrite anything here: a name that exists is refused.
"""

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.llm.contracts import LLMMessage
from agent.workflow.table_report_planner import PlanValidationError, _extract_json

MAX_TABLES = 20
MAX_COLUMNS = 50
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
# Closed set of column types -> MySQL types. Nothing else can be requested.
COLUMN_TYPES = {
    "integer": "INT", "decimal": "DECIMAL(14,2)", "text": "VARCHAR(255)", "long_text": "TEXT",
    "date": "DATE", "datetime": "DATETIME", "boolean": "TINYINT(1)",
}


class CreationRefused(ValueError):
    """The database cannot be created (it exists, or the design is not allowed)."""


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type: str
    nullable: bool = True


@dataclass(frozen=True)
class TableDef:
    name: str
    columns: tuple[ColumnDef, ...]


@dataclass(frozen=True)
class DatabaseDesign:
    database: str
    tables: tuple[TableDef, ...] = ()
    title: str = ""


@dataclass(frozen=True)
class DesignPlanResult:
    question: str
    design: DatabaseDesign | None
    reason: str | None
    provider: str | None
    attempts: int = 1


class DatabaseDesigner:
    def __init__(self, llm: Any, *, prefix: str, max_attempts: int = 2) -> None:
        self.llm, self.prefix, self.max_attempts = llm, prefix, max_attempts

    def _system_prompt(self) -> str:
        return (
            "You design ONE new database. You never write SQL.\n"
            f"The database name MUST start with '{self.prefix}' and use only lowercase letters, digits "
            "and underscores. Reply with ONLY one JSON object:\n"
            '{"supported": true, "database": "' + self.prefix + 'name", "title": "...", "tables": ['
            '{"name": "tb_x", "columns": [{"name": "col", "type": "text", "nullable": true}]}]}\n'
            f"Column types, only these: {sorted(COLUMN_TYPES)}. Do NOT add an id column: a numeric primary "
            f"key named id is added automatically. At most {MAX_TABLES} tables and {MAX_COLUMNS} columns "
            "per table; table and column names are lowercase letters, digits and underscores. A database "
            "with no tables is allowed if the user asks for only the database.\n"
            'If the request cannot be done: {"supported": false, "reason": "..."}.'
        )

    def plan(self, question: str) -> DesignPlanResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        question = question.strip()
        messages = [LLMMessage("system", self._system_prompt()), LLMMessage("user", f"Request:\n{question}")]
        errors: list[str] = []
        provider = None
        for attempt in range(1, self.max_attempts + 1):
            response = self.llm.complete(messages, temperature=0)
            provider = response.provider
            try:
                data = _extract_json(response.content)
                if data.get("supported") is False:
                    return DesignPlanResult(question, None, str(data.get("reason") or "Cannot be done.").strip(),
                                            provider, attempt)
                return DesignPlanResult(question, self.validate(data), None, provider, attempt)
            except PlanValidationError as exc:
                errors.append(str(exc))
                messages += [LLMMessage("assistant", response.content),
                             LLMMessage("user", f"Invalid: {exc} Reply again with ONLY the corrected JSON object.")]
        return DesignPlanResult(question, None, f"Could not produce a valid design: {errors[-1]}",
                                provider, self.max_attempts)

    def validate(self, data: Mapping[str, Any]) -> DatabaseDesign:
        database = data.get("database")
        if (not isinstance(database, str) or not database.startswith(self.prefix)
                or not re.fullmatch(r"[a-z][a-z0-9_]{1,56}", database) or database == self.prefix):
            raise PlanValidationError(
                f"The database name must start with '{self.prefix}' and use lowercase letters, digits and _.")
        raw_tables = data.get("tables", [])
        if not isinstance(raw_tables, list) or len(raw_tables) > MAX_TABLES:
            raise PlanValidationError(f"tables must be a list of at most {MAX_TABLES}.")
        tables, seen = [], set()
        for raw in raw_tables:
            if not isinstance(raw, Mapping) or set(raw) - {"name", "columns"}:
                raise PlanValidationError('Each table must be {"name": ..., "columns": [...]}.')
            name = raw.get("name")
            if not isinstance(name, str) or not _NAME.match(name) or name in seen:
                raise PlanValidationError(f"Invalid or repeated table name {name!r}.")
            seen.add(name)
            columns_raw = raw.get("columns")
            if not isinstance(columns_raw, list) or not 1 <= len(columns_raw) <= MAX_COLUMNS:
                raise PlanValidationError(f"Table {name!r} needs 1 to {MAX_COLUMNS} columns.")
            columns, names = [], {"id"}
            for col in columns_raw:
                if not isinstance(col, Mapping) or set(col) - {"name", "type", "nullable"}:
                    raise PlanValidationError('Each column must be {"name", "type", "nullable"}.')
                cname, ctype = col.get("name"), col.get("type")
                if not isinstance(cname, str) or not _NAME.match(cname):
                    raise PlanValidationError(f"Invalid column name {cname!r} in {name!r}.")
                if cname in names:
                    raise PlanValidationError(
                        f"Column {cname!r} repeats in {name!r} (an id column is added automatically).")
                if ctype not in COLUMN_TYPES:
                    raise PlanValidationError(f"Column type {ctype!r} is not allowed; use one of {sorted(COLUMN_TYPES)}.")
                nullable = col.get("nullable", True)
                if not isinstance(nullable, bool):
                    raise PlanValidationError("nullable must be true or false.")
                names.add(cname)
                columns.append(ColumnDef(cname, ctype, nullable))
            tables.append(TableDef(name, tuple(columns)))
        title = data.get("title")
        title = title.strip()[:200] if isinstance(title, str) and title.strip() else f"Create {database}"
        return DatabaseDesign(database, tuple(tables), title)


def build_statements(design: DatabaseDesign) -> list[str]:
    """CREATE statements. Every name was validated against a strict pattern and is quoted."""
    statements = [f"CREATE DATABASE `{design.database}` CHARACTER SET utf8mb4"]
    for table in design.tables:
        columns = ["`id` INT NOT NULL AUTO_INCREMENT"]
        columns += [f"`{c.name}` {COLUMN_TYPES[c.type]} {'NULL' if c.nullable else 'NOT NULL'}"
                    for c in table.columns]
        columns.append("PRIMARY KEY (`id`)")
        statements.append(f"CREATE TABLE `{design.database}`.`{table.name}` ({', '.join(columns)}) "
                          "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4")
    return statements


class DatabaseCreationService:
    def __init__(self, *, designer: DatabaseDesigner, connect: Callable[[], Any], profile: str,
                 audit_path: Path,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.designer, self._connect, self.profile = designer, connect, profile
        self.audit_path, self._clock = Path(audit_path), clock
        self._lock = threading.Lock()

    @staticmethod
    def summarize(design: DatabaseDesign) -> dict[str, Any]:
        return {"database": design.database, "title": design.title,
                "tables": [{"name": t.name, "columns": [{"name": "id", "type": "integer (otomatis)", "nullable": False}]
                            + [{"name": c.name, "type": c.type, "nullable": c.nullable} for c in t.columns]}
                           for t in design.tables]}

    @staticmethod
    def confirm_phrase(design: DatabaseDesign) -> str:
        return f"BUAT DATABASE {design.database}"

    def _exists(self, connection, database: str) -> bool:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s",
                           (database,))
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def preview(self, design: DatabaseDesign) -> list[str]:
        """The statements that would run. Refuses when the name is taken. Creates nothing."""
        connection = self._connect()
        try:
            if self._exists(connection, design.database):
                raise CreationRefused(f"A database named '{design.database}' already exists; "
                                      "nothing is overwritten. Choose another name.")
        finally:
            connection.close()
        return build_statements(design)

    def execute(self, design: DatabaseDesign, *, actor: str, plan_id: str, question: str) -> dict[str, Any]:
        statements = build_statements(design)
        record: dict[str, Any] = {"plan_id": plan_id, "actor": actor, "database": design.database,
                                  "tables": [t.name for t in design.tables], "statements": statements,
                                  "profile": self.profile, "question": question[:500]}
        connection = self._connect()
        done = 0
        try:
            if self._exists(connection, design.database):  # re-checked: the name may have been taken meanwhile
                raise CreationRefused(f"A database named '{design.database}' already exists.")
            cursor = connection.cursor()
            try:
                for sql in statements:  # DDL commits by itself; there is no transaction to roll back
                    cursor.execute(sql)
                    done += 1
            finally:
                cursor.close()
        except Exception as exc:
            record.update(status="failed", statements_done=done, reason=f"{type(exc).__name__}: {exc}"[:300])
            self._audit(record)
            raise
        finally:
            try:
                connection.close()
            except Exception:
                pass
        record.update(status="created", statements_done=done)
        self._audit(record)
        return {"database": design.database, "tables": [t.name for t in design.tables], "profile": self.profile}

    def audit(self, event: str, **fields: Any) -> None:
        self._audit({"event": event, **fields})

    def _audit(self, record: dict[str, Any]) -> None:
        record = {"at": self._clock().isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
        with self._lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


__all__ = ["COLUMN_TYPES", "ColumnDef", "CreationRefused", "DatabaseCreationService", "DatabaseDesign",
           "DatabaseDesigner", "DesignPlanResult", "TableDef", "build_statements"]
