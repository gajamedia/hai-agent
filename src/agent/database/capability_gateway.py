"""Composition factory for authenticated, traced business-database reads.

The caller must explicitly configure and authorize a read-only ConnectionManager
profile before calling this factory. This module never creates credentials,
opens a database connection during construction, installs drivers, or chooses
a business database automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from agent.core.action import Action
from agent.llm.authenticated_capability_execution_gateway import (
    AuthenticatedCapabilityExecutionGateway,
)
from agent.llm.business_database_adapter import (
    BusinessDatabaseAdapter,
    ReadOnlyTableSpec,
)
from agent.llm.capability_registry import TrustedCapabilityRegistry
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.execution_trace_store import ExecutionTraceStore
from agent.llm.traced_authenticated_capability_execution_gateway import (
    TracedAuthenticatedCapabilityExecutionGateway,
)
from api.security.authorization import AuthorizationPolicy

from .connection_manager import ConnectionManager


DEFAULT_CAPABILITY_ID = "business_database.select"
DEFAULT_ACTION_NAME = "read_business_data"
DEFAULT_PERMISSION = "business:database:read"
DEFAULT_RISK_SCORE = 0.1


class BusinessDatabaseGatewayConfigurationError(ValueError):
    """Raised when the explicit database gateway configuration is invalid."""


@dataclass(frozen=True)
class BusinessDatabaseGatewayAssembly:
    """References to the composed gateway and its application-owned parts.

    ConnectionManager ownership remains with the caller. The assembly does not
    shut it down because it may be shared with other application components.
    """

    gateway: TracedAuthenticatedCapabilityExecutionGateway
    authenticated_gateway: AuthenticatedCapabilityExecutionGateway
    adapter: BusinessDatabaseAdapter
    registry: TrustedCapabilityRegistry
    trace_store: ExecutionTraceStore
    connection_manager: ConnectionManager
    profile_name: str


def build_business_database_gateway(
    *,
    connection_manager: ConnectionManager,
    profile_name: str,
    authorization_policy: AuthorizationPolicy,
    table_specs: Mapping[str, ReadOnlyTableSpec],
    trace_path: str | Path,
    paramstyle: str,
    permission: str = DEFAULT_PERMISSION,
    risk_score: float = DEFAULT_RISK_SCORE,
    fsync_trace: bool = True,
    identifier_quote: str | None = None,
) -> BusinessDatabaseGatewayAssembly:
    """Compose an explicitly configured read-only database capability.

    The named ConnectionManager profile must already exist and be marked
    read-only. Its mandatory read_only_guard must enforce backend-specific
    restrictions for every connection. The factory cannot prove the behavior
    of an arbitrary guard; production setup must additionally use a database
    principal with SELECT-only grants.

    No connection is opened by this function. A connection is opened only
    when an authenticated capability read reaches the adapter.
    """
    if not isinstance(connection_manager, ConnectionManager):
        raise TypeError("connection_manager must be ConnectionManager.")
    if not isinstance(authorization_policy, AuthorizationPolicy):
        raise TypeError("authorization_policy must be AuthorizationPolicy.")
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise BusinessDatabaseGatewayConfigurationError(
            "profile_name must be a non-empty string."
        )
    if not isinstance(permission, str) or not permission.strip():
        raise BusinessDatabaseGatewayConfigurationError(
            "permission must be a non-empty string."
        )
    if isinstance(risk_score, bool) or not isinstance(risk_score, (int, float)):
        raise BusinessDatabaseGatewayConfigurationError(
            "risk_score must be a number between 0 and 1."
        )
    if not 0.0 <= float(risk_score) <= 1.0:
        raise BusinessDatabaseGatewayConfigurationError(
            "risk_score must be between 0 and 1."
        )

    normalized_profile = profile_name.strip()
    profile = connection_manager.get_profile(normalized_profile)
    if profile.read_only is not True or profile.enabled is not True:
        raise BusinessDatabaseGatewayConfigurationError(
            "The selected ConnectionManager profile must be enabled and read-only."
        )
    if not profile.authorization_ref.strip():
        raise BusinessDatabaseGatewayConfigurationError(
            "The selected profile must have an explicit authorization reference."
        )

    adapter = BusinessDatabaseAdapter(
        connection_factory=connection_manager.connection_factory(normalized_profile),
        table_specs=table_specs,
        paramstyle=paramstyle,
        identifier_quote=identifier_quote,
    )

    registry = TrustedCapabilityRegistry()
    registry.register_trusted(
        capability_id=DEFAULT_CAPABILITY_ID,
        action=Action(
            name=DEFAULT_ACTION_NAME,
            description="Read application-allowlisted business database data.",
            # The adapter is the only data path; this registry handler is
            # metadata required by the existing trusted capability contract.
            handler=lambda **_: {"success": True},
        ),
        risk_level="low",
        permissions=(permission.strip(),),
    )

    authenticated_gateway = AuthenticatedCapabilityExecutionGateway(
        [adapter],
        capability_registry=registry,
        authorization_policy=authorization_policy,
        execution_security_policy=ExecutionSecurityPolicy(
            action_risks={DEFAULT_CAPABILITY_ID: float(risk_score)}
        ),
    )

    trace_store = ExecutionTraceStore(trace_path, fsync=fsync_trace)
    traced_gateway = TracedAuthenticatedCapabilityExecutionGateway(
        authenticated_gateway,
        trace_store,
    )

    return BusinessDatabaseGatewayAssembly(
        gateway=traced_gateway,
        authenticated_gateway=authenticated_gateway,
        adapter=adapter,
        registry=registry,
        trace_store=trace_store,
        connection_manager=connection_manager,
        profile_name=normalized_profile,
    )


__all__ = [
    "BusinessDatabaseGatewayAssembly",
    "BusinessDatabaseGatewayConfigurationError",
    "DEFAULT_CAPABILITY_ID",
    "DEFAULT_PERMISSION",
    "build_business_database_gateway",
]
