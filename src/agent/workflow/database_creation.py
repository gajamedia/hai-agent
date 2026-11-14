from __future__ import annotations

"""Creating a new database (and its tables) without letting an LLM write DDL.

The LLM only chooses a *structured* design: a database name and tables whose columns use a small closed
set of types. Our code validates it (the name must start with the configured prefix, nothing may already
exist under that name), shows it, and only then builds the CREATE statements. There is no way to drop,
alter or overwrite anything here: a name that exists is refused.
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent.llm.contracts import LLMMessage
from agent.workflow.table_report_planner import PlanValidationError, _extract_json

logger = logging.getLogger(__name__)
DEFAULT_MAX_TABLES = 20
TABLES_CEILING = 200
MAX_COLUMNS = 500


def max_tables() -> int:
    """How many tables one design may have, and how many tables one connection may approve: HAI_MAX_TABLES
    (default 20, at most 200). One setting for both, so what can be created can also be approved. Read on every
    call; an unreadable or out-of-range value falls back to the default, with a warning in the log."""
    raw = os.getenv("HAI_MAX_TABLES", "").strip()
    if not raw:
        return DEFAULT_MAX_TABLES
    if raw.isdigit() and 1 <= int(raw) <= TABLES_CEILING:
        return int(raw)
    logger.warning("HAI_MAX_TABLES=%r is not a whole number from 1 to %d; using %d.",
                   raw, TABLES_CEILING, DEFAULT_MAX_TABLES)
    return DEFAULT_MAX_TABLES

# MySQL counts every column in the row size limit (65,535 bytes). utf8mb4 VARCHAR(255) = 1,022 bytes,
# TEXT only a pointer. Checked here so a too-wide design is refused with a clear message, not a DDL error.
_ROW_BYTES = {"integer": 4, "decimal": 8, "text": 1022, "long_text": 12, "date": 3, "datetime": 8, "boolean": 1}
_ROW_LIMIT = 65000
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
DEFAULT_TABLE_PREFIX = "tb_"
_PREFIX_OK = re.compile(r"^[a-z][a-z0-9_]{0,9}$")


def table_prefix() -> str:
    """What every new table name starts with: HAI_TABLE_PREFIX (default ``tb_``). Set it to nothing (an empty
    value) to leave names as they are. Read on every call; a value that is not 1 to 10 lowercase letters, digits
    or underscores (starting with a letter) falls back to the default, with a warning in the log."""
    raw = os.environ.get("HAI_TABLE_PREFIX")
    if raw is None:
        return DEFAULT_TABLE_PREFIX
    raw = raw.strip()
    if raw == "" or _PREFIX_OK.match(raw):
        return raw
    logger.warning("HAI_TABLE_PREFIX=%r is not 1 to 10 lowercase letters, digits or underscores; using %r.",
                   raw, DEFAULT_TABLE_PREFIX)
    return DEFAULT_TABLE_PREFIX
# Closed set of column types -> MySQL types. Nothing else can be requested.
COLUMN_TYPES = {
    "integer": "INT", "decimal": "DECIMAL(14,2)", "text": "VARCHAR(255)", "long_text": "TEXT",
    "date": "DATE", "datetime": "DATETIME", "boolean": "TINYINT(1)",
}


_SQLSERVER_TYPES = {
    "integer": "INT", "decimal": "DECIMAL(14,2)", "text": "NVARCHAR(255)", "long_text": "NVARCHAR(MAX)",
    "date": "DATE", "datetime": "DATETIME2", "boolean": "BIT",
}


@dataclass(frozen=True)
class Dialect:
    """What differs between the servers: type names, the placeholder, and whether the row size is capped."""
    name: str
    types: Mapping[str, str]
    placeholder: str
    row_limit: int | None


MYSQL = Dialect("mysql", COLUMN_TYPES, "%s", _ROW_LIMIT)
# SQL Server moves long variable-length values off the row, so the 65,000-byte MySQL cap does not apply.
SQLSERVER = Dialect("sqlserver", _SQLSERVER_TYPES, "?", None)
# SQLite keeps each database in one file, so there is no server to ask and no row size cap. The type names
# are chosen so that reading the table back maps to the same kinds (integer, number, date, ...).
_SQLITE_TYPES = {
    "integer": "INTEGER", "decimal": "NUMERIC", "text": "TEXT", "long_text": "TEXT",
    "date": "DATE", "datetime": "DATETIME", "boolean": "BOOLEAN",
}
SQLITE = Dialect("sqlite", _SQLITE_TYPES, "?", None)
DIALECTS = {"mysql": MYSQL, "sqlserver": SQLSERVER, "sqlite": SQLITE}
_LOGIN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


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


IMAGE_NOTE = (
    "\nAn image is attached: a database diagram (ERD), a sketch of tables, or a form. Read the tables and "
    "their columns from it and design exactly those (do not invent extra tables). Relations between tables "
    "are not enforced here: keep each as an ordinary integer column named like the key it points to (for "
    "example id_pelanggan). Ignore a primary key called id (it is added automatically). If the image does "
    "not show a database design, reply with supported false and say what you see."
)


class DatabaseDesigner:
    def __init__(self, llm: Any, *, prefix: str, max_attempts: int = 2, dialect: Dialect = MYSQL) -> None:
        self.llm, self.prefix, self.max_attempts, self.dialect = llm, prefix, max_attempts, dialect

    def _system_prompt(self, with_image: bool = False) -> str:
        return (
            "You design tables in ONE database. You never write SQL. The database may not exist yet "
            "(it is then created) or may already exist (the tables are then ADDED to it; existing tables "
            "are never changed, and a table name that already exists is refused).\n"
            f"The database name MUST start with '{self.prefix}' and use only lowercase letters, digits "
            "and underscores. Reply with ONLY one JSON object:\n"
            '{"supported": true, "database": "' + self.prefix + 'name", "title": "...", "tables": ['
            '{"name": "' + table_prefix() + 'x", "columns": [{"name": "col", "type": "text", "nullable": true}]}]}\n'
            f"Column types, only these: {sorted(COLUMN_TYPES)}. Do NOT add an id column: a numeric primary "
            f"key named id is added automatically. At most {max_tables()} tables and {MAX_COLUMNS} columns "
            "per table (a table needs at least one column; if the user names a kind of system such as HRIS, "
            "choose the usual columns yourself). Use text only for short values; use long_text for notes"
            + (", because a row can hold about 65 short text columns" if self.dialect.row_limit else "")
            + (f". Table names start with {table_prefix()}" if table_prefix() else "")
            + ". Table and column names are lowercase "
            "letters, digits and underscores. A database "
            "with no tables is allowed if the user asks for only the database. If the user only wants "
            "more tables in a database that exists, list just the new tables.\n"
            'If the request cannot be done: {"supported": false, "reason": "..."}.'
            + (IMAGE_NOTE if with_image else "")
        )

    def plan(self, question: str, images: Sequence[Any] = ()) -> DesignPlanResult:
        """Design from text, and optionally from images (an ERD, a sketched table) attached to the request."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        question = question.strip()
        messages = [LLMMessage("system", self._system_prompt(bool(images))), LLMMessage("user", f"Request:\n{question}")]
        extra = {"images": tuple(images)} if images else {}
        errors: list[str] = []
        provider = None
        for attempt in range(1, self.max_attempts + 1):
            response = self.llm.complete(messages, temperature=0, **extra)
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
        limit = max_tables()
        if not isinstance(raw_tables, list) or len(raw_tables) > limit:
            raise PlanValidationError(f"tables must be a list of at most {limit}.")
        tables, seen = [], set()
        for raw in raw_tables:
            if not isinstance(raw, Mapping) or set(raw) - {"name", "columns"}:
                raise PlanValidationError('Each table must be {"name": ..., "columns": [...]}.')
            name = raw.get("name")
            if not isinstance(name, str) or not _NAME.match(name):
                raise PlanValidationError(f"Invalid or repeated table name {name!r}.")
            # Every table name starts with the configured prefix (tb_ by default), whatever the model (or the
            # file) called it, on every server. An empty prefix leaves names as they are.
            prefix = table_prefix()
            if prefix and not name.startswith(prefix):
                name = prefix + name
            if not _NAME.match(name) or name in seen:
                raise PlanValidationError(f"Invalid or repeated table name {name!r}"
                                          + (f" (names start with {prefix})." if prefix else "."))
            seen.add(name)
            columns_raw = raw.get("columns")
            if not isinstance(columns_raw, list) or not columns_raw:
                raise PlanValidationError(
                    f"Table {name!r} has no columns. List every column yourself (for a request such as "
                    "'the usual fields of an HRIS', choose the usual fields).")
            if len(columns_raw) > MAX_COLUMNS:
                raise PlanValidationError(f"Table {name!r} has {len(columns_raw)} columns; the limit is {MAX_COLUMNS}.")
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
            row_bytes = 4 + sum(_ROW_BYTES[c.type] for c in columns)
            if self.dialect.row_limit and row_bytes > self.dialect.row_limit:
                raise PlanValidationError(
                    f"Table {name!r} is too wide for MySQL (a row may hold about 65,000 bytes and each short text "
                    "column takes about 1,000). Use long_text for descriptions and notes, or split the table in two.")
            tables.append(TableDef(name, tuple(columns)))
        title = data.get("title")
        title = title.strip()[:200] if isinstance(title, str) and title.strip() else f"Create {database}"
        return DatabaseDesign(database, tuple(tables), title)


def _build_sqlserver(design: DatabaseDesign, database_exists: bool, grant_reader: str,
                     grant_writer: str = "") -> list[str]:
    statements = [] if database_exists else [f"CREATE DATABASE [{design.database}]"]
    if grant_writer and not database_exists:
        if not _LOGIN.match(grant_writer):
            raise CreationRefused("the writer account name is not a plain name")
        # The writer may change data in the new database (the application still limits tables and columns).
        for line in (f"CREATE USER [{grant_writer}] FOR LOGIN [{grant_writer}]",
                     f"ALTER ROLE [db_datareader] ADD MEMBER [{grant_writer}]",
                     f"ALTER ROLE [db_datawriter] ADD MEMBER [{grant_writer}]"):
            statements.append(f"EXEC [{design.database}].sys.sp_executesql N'{line}'")
    if grant_reader and not database_exists:
        if not _LOGIN.match(grant_reader):
            raise CreationRefused("the reader account name is not a plain name")
        # Run inside the new database, as its owner (the creator): the reader may read, nothing more.
        statements.append(f"EXEC [{design.database}].sys.sp_executesql N'CREATE USER [{grant_reader}] "
                          f"FOR LOGIN [{grant_reader}]'")
        statements.append(f"EXEC [{design.database}].sys.sp_executesql N'ALTER ROLE [db_datareader] "
                          f"ADD MEMBER [{grant_reader}]'")
    for table in design.tables:
        columns = ["[id] INT IDENTITY(1,1) NOT NULL"]
        columns += [f"[{c.name}] {_SQLSERVER_TYPES[c.type]} {'NULL' if c.nullable else 'NOT NULL'}"
                    for c in table.columns]
        columns.append("PRIMARY KEY ([id])")
        statements.append(f"CREATE TABLE [{design.database}].[dbo].[{table.name}] ({', '.join(columns)})")
    return statements


def _build_sqlite(design: DatabaseDesign) -> list[str]:
    """Only CREATE TABLE: the database is the file, which exists as soon as it is opened."""
    statements = []
    for table in design.tables:
        columns = ['"id" INTEGER PRIMARY KEY AUTOINCREMENT']
        columns += [f'"{c.name}" {_SQLITE_TYPES[c.type]} {"NULL" if c.nullable else "NOT NULL"}'
                    for c in table.columns]
        statements.append(f'CREATE TABLE "{table.name}" ({", ".join(columns)})')
    return statements


def build_statements(design: DatabaseDesign, *, database_exists: bool = False, dialect: Dialect = MYSQL,
                     grant_reader: str = "", grant_writer: str = "") -> list[str]:
    """CREATE statements. Every name was validated against a strict pattern and is quoted.

    When the database already exists only its new tables are created: nothing existing is touched."""
    if dialect.name == "sqlserver":
        return _build_sqlserver(design, database_exists, grant_reader, grant_writer)
    if dialect.name == "sqlite":
        return _build_sqlite(design)
    statements = [] if database_exists else [f"CREATE DATABASE `{design.database}` CHARACTER SET utf8mb4"]
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
                 audit_path: Path, grant_reader: str = "", grant_writer: str = "",
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.designer, self._connect, self.profile = designer, connect, profile
        self.dialect, self.grant_reader, self.grant_writer = designer.dialect, grant_reader, grant_writer
        self.audit_path, self._clock = Path(audit_path), clock
        self._lock = threading.Lock()

    @staticmethod
    def summarize(design: DatabaseDesign) -> dict[str, Any]:
        return {"database": design.database, "title": design.title,
                "tables": [{"name": t.name, "columns": [{"name": "id", "type": "integer (otomatis)", "nullable": False}]
                            + [{"name": c.name, "type": c.type, "nullable": c.nullable} for c in t.columns]}
                           for t in design.tables]}

    @staticmethod
    def confirm_phrase(design: DatabaseDesign, mode: str = "create_database") -> str:
        return (f"TAMBAH TABEL {design.database}" if mode == "add_tables"
                else f"BUAT DATABASE {design.database}")

    @staticmethod
    def _ask(connection, sql: str, params: tuple = ()) -> list[tuple]:
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params) if params else cursor.execute(sql)
            return list(cursor.fetchall())
        finally:
            cursor.close()

    def _inspect(self, connection, design: DatabaseDesign) -> tuple[str, list[str]]:
        """("create_database" | "add_tables", the statements), or CreationRefused."""
        sqlserver = self.dialect.name == "sqlserver"
        exists = bool(self._ask(
            connection, "SELECT name FROM sys.databases WHERE name = ?" if sqlserver
            else "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s",
            (design.database,)))
        if not exists:
            return "create_database", build_statements(design, dialect=self.dialect,
                                                       grant_reader=self.grant_reader,
                                                       grant_writer=self.grant_writer)
        if not design.tables:
            raise CreationRefused(f"A database named '{design.database}' already exists and no tables were "
                                  "asked for; nothing is overwritten. Ask for tables to add, or choose another name.")
        try:
            have = {str(r[0]).lower() for r in (self._ask(
                connection, f"SELECT TABLE_NAME FROM [{design.database}].INFORMATION_SCHEMA.TABLES "
                            "WHERE TABLE_TYPE = 'BASE TABLE'") if sqlserver else self._ask(
                connection, "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s",
                (design.database,)))}
        except CreationRefused:
            raise
        except Exception as exc:
            if not sqlserver:
                raise
            raise CreationRefused(f"The creator account cannot open the existing database '{design.database}' "
                                  "(it was not created by this application); choose another name.") from exc
        clash = sorted(t.name for t in design.tables if t.name.lower() in have)
        if clash:
            raise CreationRefused(
                f"The database '{design.database}' already has the table(s) {', '.join(clash)}; existing tables "
                "are never changed. Use other table names.")
        return "add_tables", build_statements(design, database_exists=True, dialect=self.dialect)

    def preview(self, design: DatabaseDesign) -> tuple[str, list[str]]:
        """The mode and the statements that would run. Refuses what would overwrite. Creates nothing."""
        connection = self._connect()
        try:
            return self._inspect(connection, design)
        finally:
            connection.close()

    def execute(self, design: DatabaseDesign, *, actor: str, plan_id: str, question: str) -> dict[str, Any]:
        record: dict[str, Any] = {"plan_id": plan_id, "actor": actor, "database": design.database,
                                  "tables": [t.name for t in design.tables], "profile": self.profile,
                                  "question": question[:500]}
        connection = self._connect()
        done = 0
        try:
            mode, statements = self._inspect(connection, design)  # re-checked: things may have changed
            record.update(mode=mode, statements=statements)
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
        return {"database": design.database, "tables": [t.name for t in design.tables],
                "profile": self.profile, "mode": mode}

    def verify(self, design: DatabaseDesign) -> dict[str, Any]:
        """Ask the server again, after the statements ran: does the database exist, with every table?

        ``confirmed`` is True only when both are seen. A server that cannot be asked (or hides the
        objects from the creator account) gives ``confirmed: False`` with the reason, never a guess.
        """
        sqlserver = self.dialect.name == "sqlserver"
        try:
            connection = self._connect()
            try:
                exists = bool(self._ask(
                    connection, "SELECT name FROM sys.databases WHERE name = ?" if sqlserver
                    else "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s",
                    (design.database,)))
                have: set[str] = set()
                if exists and design.tables:
                    have = {str(r[0]).lower() for r in (self._ask(
                        connection, f"SELECT TABLE_NAME FROM [{design.database}].INFORMATION_SCHEMA.TABLES "
                                    "WHERE TABLE_TYPE = 'BASE TABLE'") if sqlserver else self._ask(
                        connection, "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s",
                        (design.database,)))}
            finally:
                try:
                    connection.close()
                except Exception:
                    pass
        except Exception as exc:
            return {"confirmed": False, "database_found": None, "missing_tables": None,
                    "reason": f"{type(exc).__name__}"}
        missing = sorted(t.name for t in design.tables if t.name.lower() not in have)
        return {"confirmed": exists and not missing, "database_found": exists, "missing_tables": missing}

    def audit(self, event: str, **fields: Any) -> None:
        self._audit({"event": event, **fields})

    def _audit(self, record: dict[str, Any]) -> None:
        record = {"at": self._clock().isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
        with self._lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


class SQLiteCreationService(DatabaseCreationService):
    """Creating a database on SQLite: a new file ``<name>.sqlite3`` inside the data folder.

    Same promises as on a server: the name must start with the prefix, nothing that exists is overwritten
    (an existing file only gets new tables, never changed ones) and nothing is ever dropped. The file is
    always inside ``data_root``: the name is checked against a strict pattern, so it cannot be a path.
    All the statements run in one transaction, and a file this run created is removed if it fails."""

    SUFFIX = ".sqlite3"

    def __init__(self, *, designer: DatabaseDesigner, data_root: Path, profile: str, audit_path: Path,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        super().__init__(designer=designer, connect=lambda: None, profile=profile,
                         audit_path=audit_path, clock=clock)
        self.data_root = Path(data_root)

    def _path(self, design: DatabaseDesign) -> Path:
        name = design.database
        prefix = self.designer.prefix
        if (not re.fullmatch(r"[a-z][a-z0-9_]{1,56}", name) or not name.startswith(prefix)
                or name == prefix):
            raise CreationRefused(f"The database name must start with '{prefix}' and have a name after it.")
        return self.data_root / f"{name}{self.SUFFIX}"

    @staticmethod
    def _tables(path: Path) -> set[str]:
        import sqlite3
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10)
        try:
            return {str(r[0]).lower() for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")}
        finally:
            connection.close()

    def _plan(self, design: DatabaseDesign) -> tuple[str, list[str]]:
        path = self._path(design)
        if not path.exists():
            return "create_database", build_statements(design, dialect=self.dialect)
        if not design.tables:
            raise CreationRefused(f"A database named '{design.database}' already exists and no tables were "
                                  "asked for; nothing is overwritten. Ask for tables to add, or choose another name.")
        try:
            have = self._tables(path)
        except Exception as exc:
            raise CreationRefused(f"The file of '{design.database}' cannot be opened as a SQLite database; "
                                  "choose another name.") from exc
        clash = sorted(t.name for t in design.tables if t.name.lower() in have)
        if clash:
            raise CreationRefused(
                f"The database '{design.database}' already has the table(s) {', '.join(clash)}; existing tables "
                "are never changed. Use other table names.")
        return "add_tables", build_statements(design, dialect=self.dialect)

    def preview(self, design: DatabaseDesign) -> tuple[str, list[str]]:
        return self._plan(design)

    def execute(self, design: DatabaseDesign, *, actor: str, plan_id: str, question: str) -> dict[str, Any]:
        import sqlite3
        record: dict[str, Any] = {"plan_id": plan_id, "actor": actor, "database": design.database,
                                  "tables": [t.name for t in design.tables], "profile": self.profile,
                                  "question": question[:500]}
        path = self._path(design)
        made_file = False
        done = 0
        try:
            mode, statements = self._plan(design)          # re-checked: things may have changed
            record.update(mode=mode, statements=statements)
            made_file = not path.exists()
            connection = sqlite3.connect(str(path), timeout=10, isolation_level=None)
            try:
                connection.execute("BEGIN IMMEDIATE")
                for sql in statements:                      # SQLite DDL is transactional: all or nothing
                    connection.execute(sql)
                    done += 1
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                connection.close()
        except Exception as exc:
            if made_file:
                try:
                    path.unlink()
                except OSError:
                    pass
            record.update(status="failed", statements_done=done, reason=f"{type(exc).__name__}: {exc}"[:300])
            self._audit(record)
            raise
        record.update(status="created", statements_done=done)
        self._audit(record)
        return {"database": design.database, "tables": [t.name for t in design.tables],
                "profile": self.profile, "mode": mode}

    def verify(self, design: DatabaseDesign) -> dict[str, Any]:
        try:
            path = self._path(design)
            exists = path.is_file()
            have = self._tables(path) if exists and design.tables else set()
        except Exception as exc:
            return {"confirmed": False, "database_found": None, "missing_tables": None,
                    "reason": f"{type(exc).__name__}"}
        missing = sorted(t.name for t in design.tables if t.name.lower() not in have)
        return {"confirmed": exists and not missing, "database_found": exists, "missing_tables": missing}


__all__ = ["COLUMN_TYPES", "DIALECTS", "Dialect", "MYSQL", "SQLITE", "SQLSERVER", "SQLiteCreationService", "ColumnDef", "CreationRefused", "DatabaseCreationService", "DatabaseDesign",
           "DatabaseDesigner", "DesignPlanResult", "TableDef", "build_statements"]
