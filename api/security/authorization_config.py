from __future__ import annotations

import os
import re
from typing import Mapping

from api.security.authorization import (
    API_ROUTE_PERMISSIONS,
    CAPABILITY_PERMISSIONS,
    AuthorizationPolicy,
    AuthorizationPolicyConfigurationError,
)


class AuthorizationConfigurationError(ValueError):
    """Raised when environment-backed authorization configuration is invalid."""


_CLIENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _role_map_for_keycloak(api_client_id: str) -> dict[str, tuple[str, ...]]:
    if not _CLIENT_ID_PATTERN.fullmatch(api_client_id):
        raise AuthorizationConfigurationError(
            "HAI_KEYCLOAK_API_CLIENT_ID must be a lowercase client identifier."
        )
    prefix = f"client:{api_client_id}:"
    p = CAPABILITY_PERMISSIONS
    api = API_ROUTE_PERMISSIONS
    return {
        f"{prefix}api-reader": (
            api["execution_state_read"],
            api["strategy_registry_read"],
            api["telemetry_capability_read"],
            api["autonomous_batch_read"],
        ),
        f"{prefix}database-onboarding-reader": (api["database_onboarding_read"],),
        f"{prefix}database-onboarding-writer": (api["database_onboarding_write"],),
        f"{prefix}execution-operator": (
            api["execution_action_execute"],
            api["execution_plan_execute"],
        ),
        f"{prefix}execution-state-operator": (
            api["execution_state_write"],
        ),
        f"{prefix}execution-reset-operator": (
            api["execution_state_reset"],
        ),
        f"{prefix}execution-reconciler": (
            api["execution_reconciliation_write"],
        ),
        f"{prefix}cognitive-operator": (api["cognitive_cycle_run"],),
        f"{prefix}autonomous-testing-operator": (
            api["autonomous_batch_create"],
            api["autonomous_batch_run"],
        ),
        f"{prefix}autonomous-testing-reconciler": (
            api["autonomous_batch_reconcile"],
        ),
        f"{prefix}proposal-reader": (p["proposal_read"],),
        f"{prefix}proposal-submitter": (p["proposal_submit"],),
        f"{prefix}proposal-approver": (p["proposal_approve"],),
        f"{prefix}proposal-rejector": (p["proposal_reject"],),
        f"{prefix}proposal-register": (p["proposal_register"],),
        f"{prefix}lifecycle-reader": (p["lifecycle_read"],),
        f"{prefix}lifecycle-activator": (p["lifecycle_activate"],),
        f"{prefix}lifecycle-disabler": (p["lifecycle_disable"],),
        f"{prefix}lifecycle-revoker": (p["lifecycle_revoke"],),
        f"{prefix}environment-workspace-reader": (p["environment_workspace_read"],),
        f"{prefix}canonical-capability-invoker": (api["canonical_capability_invoke"],),
        # Data reports. Running a report needs the report permission AND the data read
        # permission; managing connections and reviewing engine requests are separate.
        f"{prefix}data-reports-user": (
            api["table_report_run"],
            api["business_database_read"],
            api["database_engine_request"],
            api["task_history_read"],
        ),
        f"{prefix}database-connection-manager": (api["database_connection_manage"],),
        # Changing data (insert/update/delete) is separate from reading: the writer may plan and
        # execute; a different person with the approver role approves what needs a second pair of eyes.
        f"{prefix}data-writer": (api["data_change_plan"], api["data_change_execute"], api["task_history_read"]),
        # Recording new data only (a cashier or clerk): no update, no delete, no approval of others' changes.
        f"{prefix}data-entry": (api["data_entry_record"], api["task_history_read"]),
        f"{prefix}data-change-approver": (api["data_change_approve"],),
        f"{prefix}data-schema-creator": (api["database_create_manage"], api["task_history_read"]),
        f"{prefix}database-engine-reviewer": (
            api["database_engine_review"],
            api["database_engine_request"],  # reviewers also list requests
        ),
    }


def create_environment_authorization_policy(
    app=None, *, environ: Mapping[str, str] | None = None
) -> AuthorizationPolicy | None:
    """Build a trusted, exact role-to-permission policy from environment.

    Disabled authentication returns None. Keycloak maps only roles belonging
    to the configured API client. Generic OIDC has no implicit role grants;
    verified explicit permission claims remain usable.
    """
    env = os.environ if environ is None else environ
    enabled = env.get("HAI_AUTH_ENABLED", "false").strip().lower()
    if enabled not in {"true", "false"}:
        raise AuthorizationConfigurationError(
            "HAI_AUTH_ENABLED must be exactly 'true' or 'false'."
        )
    if enabled == "false":
        return None

    provider = env.get("HAI_AUTH_PROVIDER", "oidc").strip().lower()
    if provider == "keycloak":
        client_id = env.get("HAI_KEYCLOAK_API_CLIENT_ID", "").strip()
        if not client_id:
            raise AuthorizationConfigurationError(
                "HAI_KEYCLOAK_API_CLIENT_ID is required for Keycloak PBAC."
            )
        role_map = _role_map_for_keycloak(client_id)
    elif provider == "oidc":
        role_map = {}
    else:
        raise AuthorizationConfigurationError(
            "HAI_AUTH_PROVIDER must be either 'oidc' or 'keycloak'."
        )

    try:
        return AuthorizationPolicy(role_permissions=role_map)
    except AuthorizationPolicyConfigurationError as exc:
        raise AuthorizationConfigurationError(
            "Environment authorization policy is invalid."
        ) from exc


__all__ = ["AuthorizationConfigurationError", "create_environment_authorization_policy"]
