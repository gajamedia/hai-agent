"""Explicit API composition for a user-approved read-only SQLite capability.

Importing this module does nothing.  A caller must deliberately pass the
returned factory into ``create_app(capability_execution_gateway_factory=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from agent.database.connection_manager import ConnectionManager
from agent.database.sqlite_readonly import register_sqlite_read_only_profile
from agent.llm.business_database_adapter import ReadOnlyTableSpec

from api.database_gateway_factory import create_api_business_database_gateway_factory


def create_sqlite_readonly_gateway_factory(
    *,
    database_path: str | Path,
    profile_name: str,
    authorization_ref: str,
    table_specs: Mapping[str, ReadOnlyTableSpec],
    trace_path: str | Path,
    manager_factory: Callable[[], ConnectionManager] = ConnectionManager,
    permission: str = "business:database:read",
    risk_score: float = 0.1,
    fsync_trace: bool = True,
):
    """Build a create_app-compatible, read-only SQLite gateway factory.

    ``database_path`` must already exist.  The supplied ``authorization_ref``
    is the user approval/audit reference, never a password or connection URL.
    Table specs are expected to originate from the approved schema allowlist.
    """
    approved_path = Path(database_path).expanduser()

    def configure(manager: ConnectionManager, _app) -> None:
        register_sqlite_read_only_profile(
            manager,
            profile_name=profile_name,
            database_path=approved_path,
            authorization_ref=authorization_ref,
        )

    return create_api_business_database_gateway_factory(
        profile_name=profile_name,
        profile_configurator=configure,
        table_specs=table_specs,
        trace_path=trace_path,
        paramstyle="qmark",
        manager_factory=manager_factory,
        permission=permission,
        risk_score=risk_score,
        fsync_trace=fsync_trace,
    )


__all__ = ["create_sqlite_readonly_gateway_factory"]
