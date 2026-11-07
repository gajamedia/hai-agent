from __future__ import annotations

import pytest

from agent.llm.business_database_adapter import BusinessDatabaseAdapter, ReadOnlyTableSpec

SPEC = ReadOnlyTableSpec(table="sales", columns=("id", "region", "amount"),
                         filterable_columns=("region", "amount"), sortable_columns=("amount",))


class _Capture:
    def __init__(self):
        self.queries = []

    def __call__(self):
        outer = self

        class Cursor:
            description = [("id",)]

            def execute(self, query, params):
                outer.queries.append((query, params))

            def fetchall(self):
                return []

            def close(self):
                pass

        class Conn:
            def cursor(self):
                return Cursor()

            def close(self):
                pass
        return Conn()


def _build(dialect, paramstyle, quote=None, **arguments):
    capture = _Capture()
    BusinessDatabaseAdapter(
        connection_factory=capture, table_specs={"sales": SPEC}, paramstyle=paramstyle,
        identifier_quote=quote, dialect=dialect,
    ).read("business_database.select", {"table": "sales", **arguments})
    return capture.queries[0]


ARGS = dict(columns=["id", "region"], filters={"region": "west", "amount": 5}, order_by=["-amount"],
            limit=25)


def test_limit_dialect_is_unchanged():
    assert _build("limit", "qmark", **ARGS) == (
        "SELECT id, region FROM sales WHERE region = ? AND amount = ? ORDER BY amount DESC LIMIT ?",
        ("west", 5, 25))


def test_sql_server_top_puts_the_limit_first_and_binds_it_first():
    sql, params = _build("top", "qmark", "[", **ARGS)
    assert sql == ("SELECT TOP (?) [id], [region] FROM [sales] WHERE [region] = ? AND [amount] = ? "
                   "ORDER BY [amount] DESC")
    assert params == (25, "west", 5)  # TOP's parameter appears first in the text


def test_oracle_fetch_first_with_numeric_placeholders_in_text_order():
    sql, params = _build("fetch_first", "numeric", '"', **ARGS)
    assert sql == ('SELECT "id", "region" FROM "sales" WHERE "region" = :1 AND "amount" = :2 '
                   'ORDER BY "amount" DESC FETCH FIRST :3 ROWS ONLY')
    assert params == ("west", 5, 25)


def test_top_with_numeric_numbers_the_limit_as_one():
    sql, params = _build("top", "numeric", None, filters={"region": "x"})
    assert sql == "SELECT TOP (:1) id, region, amount FROM sales WHERE region = :2"
    assert params == (100, "x")


def test_postgres_style_format_placeholders_still_work():
    sql, params = _build("limit", "format", '"', filters={"region": "x"})
    assert sql.endswith('WHERE "region" = %s LIMIT %s') and params == ("x", 100)


def test_null_filter_binds_nothing_and_keeps_positions_correct():
    sql, params = _build("top", "numeric", None, filters={"region": None, "amount": 7})
    assert sql == "SELECT TOP (:1) id, region, amount FROM sales WHERE region IS NULL AND amount = :2"
    assert params == (100, 7)


@pytest.mark.parametrize("dialect,paramstyle", [("top", "qmark"), ("fetch_first", "numeric"),
                                                ("limit", "format"), ("limit", "qmark")])
def test_values_are_never_interpolated(dialect, paramstyle):
    evil = "x'; DROP TABLE sales; --"
    sql, params = _build(dialect, paramstyle, filters={"region": evil})
    assert "DROP" not in sql and evil in params


def test_limit_is_still_range_checked_for_every_dialect():
    for dialect in ("limit", "top", "fetch_first"):
        with pytest.raises(Exception):
            _build(dialect, "qmark", limit=10_000)
        with pytest.raises(Exception):
            _build(dialect, "qmark", limit="5; DROP")


@pytest.mark.parametrize("kwargs", [{"dialect": "rownum"}, {"paramstyle": "named"},
                                    {"identifier_quote": "("}])
def test_unknown_options_are_rejected(kwargs):
    base = dict(connection_factory=lambda: None, table_specs={"sales": SPEC})
    with pytest.raises(ValueError):
        BusinessDatabaseAdapter(**base, **kwargs)


def test_bracket_quote_cannot_be_broken_out_of():
    # identifiers are validated first, so "]" can never appear inside one
    with pytest.raises(ValueError):
        ReadOnlyTableSpec(table="sales", columns=("id]; DROP TABLE x; --",))
