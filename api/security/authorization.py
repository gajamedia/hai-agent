from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from api.security.authentication import AuthenticatedPrincipal

_PERMISSION_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*:[a-z][a-z0-9_-]*:[a-z][a-z0-9_-]*$")
_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?::[a-z][a-z0-9_-]*)*$")


class AuthorizationError(RuntimeError):
    """Base class for authorization failures."""


class AuthorizationPolicyConfigurationError(AuthorizationError, ValueError):
    """Raised when an authorization policy is malformed."""


class AuthorizationDenied(AuthorizationError):
    """Raised when a verified principal lacks a required permission."""

    def __init__(self, permission: str, actor_id: str | None = None) -> None:
        self.permission = permission
        self.actor_id = actor_id
        super().__init__(f"Permission denied: {permission}")


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    allowed: bool
    permission: str
    actor_id: str | None
    reason: str


def _validate_permission(permission: str, field_name: str = "permission") -> str:
    if not isinstance(permission, str):
        raise AuthorizationPolicyConfigurationError(f"{field_name} must be a string.")
    normalized = permission.strip()
    if not _PERMISSION_PATTERN.fullmatch(normalized):
        raise AuthorizationPolicyConfigurationError(
            f"{field_name} must be a namespaced permission such as 'capability:proposal:read'."
        )
    return normalized


def _validate_role(role: str, field_name: str = "role") -> str:
    if not isinstance(role, str):
        raise AuthorizationPolicyConfigurationError(f"{field_name} must be a string.")
    normalized = role.strip()
    if not _ROLE_PATTERN.fullmatch(normalized):
        raise AuthorizationPolicyConfigurationError(
            f"{field_name} must be a lowercase role identifier, optionally namespaced."
        )
    return normalized


def _normalize_permission_collection(values: Iterable[str], field_name: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise AuthorizationPolicyConfigurationError(
            f"{field_name} must be a collection, not a string."
        )
    try:
        return frozenset(_validate_permission(value, field_name) for value in values)
    except TypeError as exc:
        raise AuthorizationPolicyConfigurationError(
            f"{field_name} must be an iterable of permission strings."
        ) from exc


def _normalize_role_map(role_permissions: Mapping[str, Iterable[str]] | None) -> Mapping[str, frozenset[str]]:
    if role_permissions is None:
        return MappingProxyType({})
    if not isinstance(role_permissions, Mapping):
        raise AuthorizationPolicyConfigurationError(
            "role_permissions must be a mapping of role to permission collection."
        )
    normalized: dict[str, frozenset[str]] = {}
    for role, permissions in role_permissions.items():
        normalized_role = _validate_role(role)
        if normalized_role in normalized:
            raise AuthorizationPolicyConfigurationError(
                f"Duplicate normalized role: {normalized_role}"
            )
        normalized[normalized_role] = _normalize_permission_collection(
            permissions, f"permissions for role {normalized_role}"
        )
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class AuthorizationPolicy:
    """Exact permission policy with default-deny and explicit-deny semantics."""

    role_permissions: Mapping[str, Iterable[str]] | None = None
    denied_permissions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_permissions", _normalize_role_map(self.role_permissions))
        object.__setattr__(
            self, "denied_permissions",
            _normalize_permission_collection(self.denied_permissions, "denied_permissions")
        )

    def evaluate(self, principal: AuthenticatedPrincipal | None, permission: str) -> AuthorizationDecision:
        required = _validate_permission(permission)
        actor_id = principal.actor_id if isinstance(principal, AuthenticatedPrincipal) else None
        if not isinstance(principal, AuthenticatedPrincipal):
            return AuthorizationDecision(False, required, None, "missing_or_untrusted_principal")
        if required in self.denied_permissions:
            return AuthorizationDecision(False, required, actor_id, "explicit_policy_deny")
        if required in principal.permissions:
            return AuthorizationDecision(True, required, actor_id, "explicit_principal_permission")
        for role in principal.roles:
            if required in (self.role_permissions or {}).get(role, frozenset()):
                return AuthorizationDecision(True, required, actor_id, f"role_grant:{role}")
        return AuthorizationDecision(False, required, actor_id, "no_matching_permission")

    def is_allowed(self, principal: AuthenticatedPrincipal | None, permission: str) -> bool:
        return self.evaluate(principal, permission).allowed

    def require(self, principal: AuthenticatedPrincipal | None, permission: str) -> AuthorizationDecision:
        decision = self.evaluate(principal, permission)
        if not decision.allowed:
            raise AuthorizationDenied(decision.permission, decision.actor_id)
        return decision


API_ROUTE_PERMISSIONS = MappingProxyType({
    "execution_state_read": "execution:state:read",
    "execution_state_write": "execution:state:write",
    "execution_action_execute": "execution:action:execute",
    "execution_plan_execute": "execution:plan:execute",
    "execution_state_reset": "execution:state:reset",
    "execution_reconciliation_write": "execution:reconciliation:write",
    "canonical_capability_invoke": "capability:canonical:invoke",
    "cognitive_cycle_run": "cognitive:cycle:run",
    "strategy_registry_read": "strategy:registry:read",
    "telemetry_capability_read": "telemetry:capability:read",
    "autonomous_batch_create": "autonomous-testing:batch:create",
    "autonomous_batch_read": "autonomous-testing:batch:read",
    "autonomous_batch_run": "autonomous-testing:batch:run",
    "autonomous_batch_reconcile": "autonomous-testing:batch:reconcile",
    "database_onboarding_read": "database:onboarding:read",
    "database_onboarding_write": "database:onboarding:write",
    "table_report_run": "report:table:run",
    "business_database_read": "business:database:read",
    "database_connection_manage": "database:connection:manage",
    "database_engine_request": "database:engine:request",
    "database_engine_review": "database:engine:review",
    "data_change_plan": "data:change:plan",
    "data_change_execute": "data:change:execute",
    "data_change_approve": "data:change:approve",
    "database_create_manage": "database:create:manage",
    "task_history_read": "task:history:read",
})

CAPABILITY_PERMISSIONS = MappingProxyType({
    "proposal_read": "capability:proposal:read",
    "proposal_submit": "capability:proposal:submit",
    "proposal_approve": "capability:proposal:approve",
    "proposal_reject": "capability:proposal:reject",
    "proposal_register": "capability:proposal:register",
    "lifecycle_read": "capability:lifecycle:read",
    "lifecycle_activate": "capability:lifecycle:activate",
    "lifecycle_disable": "capability:lifecycle:disable",
    "lifecycle_revoke": "capability:lifecycle:revoke",
    "environment_workspace_read": "environment:workspace:read",
})

__all__ = [
    "AuthorizationError", "AuthorizationPolicyConfigurationError",
    "AuthorizationDenied", "AuthorizationDecision", "AuthorizationPolicy",
    "API_ROUTE_PERMISSIONS", "CAPABILITY_PERMISSIONS",
]
