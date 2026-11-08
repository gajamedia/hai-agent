import sqlite3

from agent.database import ConnectionManager
from agent.llm.authenticated_capability_execution_gateway import AuthenticatedCapabilityReadRequest
from agent.llm.business_database_adapter import ReadOnlyTableSpec
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from api.sqlite_database_gateway_factory import create_sqlite_readonly_gateway_factory


class App:
    state = type("State", (), {"authorization_policy": AuthorizationPolicy()})()


def test_sqlite_factory_composes_an_authorized_read_only_capability(tmp_path):
    database_path = tmp_path / "sales.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount INTEGER)")
        connection.execute("INSERT INTO sales (region, amount) VALUES ('west', 125)")

    factory = create_sqlite_readonly_gateway_factory(
        database_path=database_path,
        profile_name="sales_reporting",
        authorization_ref="user-approval:sales-001",
        table_specs={
            "sales": ReadOnlyTableSpec(
                table="sales", columns=("id", "region", "amount"),
                filterable_columns=("region",), sortable_columns=("amount",),
            ),
        },
        trace_path=tmp_path / "trace.jsonl",
        fsync_trace=False,
    )
    bundle = factory(App())
    try:
        result = bundle.gateway.read(
            AuthenticatedPrincipal(
                actor_id="report-user", permissions=("business:database:read",)
            ),
            AuthenticatedCapabilityReadRequest(
                target="business_database",
                capability_id="business_database.select",
                arguments={
                    "table": "sales", "columns": ["region", "amount"],
                    "filters": {"region": "west"}, "order_by": ["-amount"], "limit": 10,
                },
            ),
            execution_id="sqlite-factory-test-001",
        )
        assert result.data == [{"region": "west", "amount": 125}]
        assert bundle.gateway is not None
    finally:
        for resource in bundle.owned_resources:
            resource.shutdown()


def test_sqlite_factory_refuses_a_missing_file(tmp_path):
    factory = create_sqlite_readonly_gateway_factory(
        database_path=tmp_path / "missing.sqlite3",
        profile_name="missing",
        authorization_ref="user-approval:missing",
        table_specs={"sales": ReadOnlyTableSpec(table="sales", columns=("id",))},
        trace_path=tmp_path / "trace.jsonl",
    )
    try:
        factory(App())
    except ValueError as error:
        assert "existing regular file" in str(error)
    else:
        raise AssertionError("missing SQLite file must not be configured")
