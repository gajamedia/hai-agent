"""Application-lifecycle factory for the read-only business database gateway.

This module is deliberately opt-in. It does not read environment variables,
select a database, load credentials, install drivers, or create a DB profile.
The caller must supply a manager factory and a trusted profile configurator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agent.database.connection_manager import ConnectionManager
from agent.database.capability_gateway import (
    BusinessDatabaseGatewayAssembly,
    build_business_database_gateway,
)
from agent.llm.business_database_adapter import ReadOnlyTableSpec
from api.resource_lifecycle import CapabilityExecutionResourceBundle
from api.security.authorization import AuthorizationPolicy


class APIDatabaseGatewayConfigurationError(ValueError):
    """Raised when explicit API database gateway configuration is incomplete."""


ManagerFactory = Callable[[], ConnectionManager]
ProfileConfigurator = Callable[[ConnectionManager, Any], None]


def create_api_business_database_gateway_factory(
    *,
    profile_name: str,
    profile_configurator: ProfileConfigurator,
    table_specs: Mapping[str, ReadOnlyTableSpec],
    trace_path: str | Path,
    paramstyle: str,
    manager_factory: ManagerFactory = ConnectionManager,
    permission: str = "business:database:read",
    risk_score: float = 0.1,
    fsync_trace: bool = True,
) -> Callable[[Any], CapabilityExecutionResourceBundle]:
    """Return a ``create_app``-compatible, opt-in gateway factory.

    ``profile_configurator`` must explicitly register the selected read-only
    profile and provide a real backend-specific read-only guard. Production
    must additionally use a DB principal with SELECT-only grants. The
    ConnectionManager is created here and transferred to the API lifespan.

    On any composition error, the newly created manager is shut down before
    the exception is re-raised. No database connection is opened at startup.
    """
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise APIDatabaseGatewayConfigurationError(
            "profile_name must be a non-empty string."
        )
    if not callable(profile_configurator):
        raise APIDatabaseGatewayConfigurationError(
            "profile_configurator must be callable and explicitly configure a read-only profile."
        )
    if not callable(manager_factory):
        raise APIDatabaseGatewayConfigurationError("manager_factory must be callable.")
    if not isinstance(table_specs, Mapping) or not table_specs:
        raise APIDatabaseGatewayConfigurationError(
            "table_specs must be a non-empty mapping of allowlisted table specifications."
        )
    if not isinstance(paramstyle, str) or paramstyle not in {"qmark", "format"}:
        raise APIDatabaseGatewayConfigurationError(
            "paramstyle must be 'qmark' or 'format'."
        )

    normalized_profile = profile_name.strip()
    normalized_trace_path = Path(trace_path).expanduser()

    def factory(app: Any) -> CapabilityExecutionResourceBundle:
        manager = manager_factory()
        if not isinstance(manager, ConnectionManager):
            closer = getattr(manager, "shutdown", None)
            if callable(closer):
                closer()
            raise TypeError("manager_factory must return ConnectionManager.")

        try:
            policy = getattr(getattr(app, "state", None), "authorization_policy", None)
            if not isinstance(policy, AuthorizationPolicy):
                raise APIDatabaseGatewayConfigurationError(
                    "The API application must initialize a valid AuthorizationPolicy before the database gateway factory runs."
                )

            # This callback is the explicit trust boundary for database profile
            # selection, credentials, authorization reference and read-only guard.
            profile_configurator(manager, app)

            assembly: BusinessDatabaseGatewayAssembly = build_business_database_gateway(
                connection_manager=manager,
                profile_name=normalized_profile,
                authorization_policy=policy,
                table_specs=table_specs,
                trace_path=normalized_trace_path,
                paramstyle=paramstyle,
                permission=permission,
                risk_score=risk_score,
                fsync_trace=fsync_trace,
            )

            # These are the same app-state observability slots used by the
            # existing API lifecycle and trace routes. They are not new owners.
            app.state.capability_execution_registry = assembly.registry
            app.state.execution_trace_store = assembly.trace_store
            # Exposed only to the authenticated schema-discovery control plane;
            # it remains application-owned and is closed by the lifespan.
            app.state.business_database_connection_manager = manager

            # Ownership transfer is explicit. The API lifespan closes the
            # manager during shutdown; its gateway/adapter are lightweight
            # dependants and must not close the manager independently.
            return CapabilityExecutionResourceBundle(
                gateway=assembly.gateway,
                owned_resources=(manager,),
            )
        except BaseException:
            manager.shutdown()
            raise

    return factory


__all__ = [
    "APIDatabaseGatewayConfigurationError",
    "create_api_business_database_gateway_factory",
]
