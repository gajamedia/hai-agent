# Adding a database engine

Supported today: SQLite, PostgreSQL, MySQL. This is the procedure for adding another
(SQL Server, Oracle, ...), whether a person or a coding agent does the work. A new
engine is accepted **only when the shared conformance suite passes against a
disposable server**.

## What an agent may and may not do

| Allowed | Not allowed without a human |
|---|---|
| Draft the engine class, profile fields, tests and docs in a branch | Install packages on a shared or production host |
| Start a throwaway local container and run the suite against it | Connect to any database that holds real data |
| Open a pull request with the conformance results | Merge, deploy, or enable the engine in a running server |

The runtime agent never loads code it wrote at runtime. Engines ship through review.

## Steps

1. **Profile type** in `api/database_engines.py`: add a frozen dataclass and extend
   `parse_profiles`. A password is never a field; only `password_env` (a variable name).
   Require encrypted transport unless the product cannot offer it.
2. **Engine class** with these members (see `PostgresEngine`, `MySQLEngine`):
   - `kind`, `paramstyle` (`qmark` or `format`), `quote` (identifier quote character),
     `map_type(native_type) -> dataset type`.
   - `open(manager, name, source, auth_ref) -> (schema, warnings)`; it must:
     - register a profile whose **every** connection is read-only and has a statement timeout;
     - **refuse** accounts that can write (superuser, all privileges, write grants, unverifiable
       role grants) with an `EngineError`;
     - discover tables/columns/primary keys using only what a SELECT-only account can see.
   - Driver errors must never reach the user: only `EngineError` messages are shown.
3. **Wire it** in `DatabaseRuntimeRegistry.__init__` (one `by_type` branch) and add the driver
   as an optional dependency in `pyproject.toml`; import the driver lazily inside `open`.
4. **Tests**: copy `tests/test_mysql_engine.py`. Provide a fixture that creates the standard
   dataset and accounts, then subclass `ConformanceSuite`. Add engine-specific tests (TLS,
   grants parsing, type mapping) beside it. Gate live tests on environment variables so the
   normal run stays offline.
5. **Run** the live suite against a disposable container, then the whole test suite.
6. **Prove the suite bites**: keep or add a test that a deliberately faulty engine fails the
   suite (see `test_the_conformance_suite_rejects_an_engine_that_skips_the_role_check`).

## The contract (tests/engine_conformance.py)

1. a SELECT-only account connects and its schema is discovered (types, primary key);
2. the session is read-only and cannot change the database;
3. a question becomes a report whose numbers equal an independent SQL query;
4. privileged and write-capable accounts are refused;
5. connection errors never leak host, user, database or password;
6. an approved connection survives a restart and answers again;
7. reserved-word and mixed-case names can be reported.

## Known gaps to design around

- The adapter builds single-table `SELECT ... WHERE col = ? LIMIT ?`. Engines whose limit syntax
  differs (SQL Server `TOP`/`OFFSET FETCH`, Oracle `FETCH FIRST`) need a small dialect hook in
  `BusinessDatabaseAdapter` first. MongoDB is not SQL and needs its own adapter.
- Licence terms of vendor images (for example SQL Server) must be accepted by a person, not a script.
