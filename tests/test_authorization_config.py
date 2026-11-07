from __future__ import annotations

import pytest

from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import CAPABILITY_PERMISSIONS
from api.security.authorization_config import (
    AuthorizationConfigurationError,
    create_environment_authorization_policy,
)


def keycloak_env(**overrides):
    env = {
        "HAI_AUTH_ENABLED": "true",
        "HAI_AUTH_PROVIDER": "keycloak",
        "HAI_KEYCLOAK_API_CLIENT_ID": "hai-agent-api",
    }
    env.update(overrides)
    return env


def test_disabled_authentication_keeps_policy_unavailable():
    assert create_environment_authorization_policy(environ={}) is None


def test_keycloak_policy_maps_only_exact_api_client_roles():
    policy = create_environment_authorization_policy(environ=keycloak_env())
    principal = AuthenticatedPrincipal(
        actor_id="hai-dev",
        roles=(
            "realm:hai-user",
            "client:hai-agent-api:proposal-reader",
            "client:hai-agent-api:lifecycle-reader",
        ),
    )
    p = CAPABILITY_PERMISSIONS
    assert policy.is_allowed(principal, p["proposal_read"])
    assert policy.is_allowed(principal, p["lifecycle_read"])
    for key in (
        "proposal_submit", "proposal_approve", "proposal_reject",
        "proposal_register", "lifecycle_activate", "lifecycle_disable",
        "lifecycle_revoke",
    ):
        assert not policy.is_allowed(principal, p[key])


def test_other_client_role_does_not_grant_api_permission():
    policy = create_environment_authorization_policy(environ=keycloak_env())
    principal = AuthenticatedPrincipal(
        actor_id="user-1", roles=("client:other-client:proposal-reader",)
    )
    assert not policy.is_allowed(principal, CAPABILITY_PERMISSIONS["proposal_read"])


def test_realm_admin_role_has_no_implicit_permissions():
    policy = create_environment_authorization_policy(environ=keycloak_env())
    principal = AuthenticatedPrincipal(
        actor_id="user-1", roles=("realm:admin", "realm:hai-user")
    )
    assert not policy.is_allowed(principal, CAPABILITY_PERMISSIONS["proposal_read"])
    assert not policy.is_allowed(principal, CAPABILITY_PERMISSIONS["lifecycle_revoke"])


def test_oidc_policy_does_not_invent_role_grants():
    policy = create_environment_authorization_policy(
        environ={"HAI_AUTH_ENABLED": "true", "HAI_AUTH_PROVIDER": "oidc"}
    )
    principal = AuthenticatedPrincipal(actor_id="user-1", roles=("admin", "proposal-reader"))
    assert not policy.is_allowed(principal, CAPABILITY_PERMISSIONS["proposal_read"])
    explicit = AuthenticatedPrincipal(
        actor_id="user-2", permissions=(CAPABILITY_PERMISSIONS["proposal_read"],)
    )
    assert policy.is_allowed(explicit, CAPABILITY_PERMISSIONS["proposal_read"])


@pytest.mark.parametrize("environ", [
    {"HAI_AUTH_ENABLED": "yes"},
    {"HAI_AUTH_ENABLED": "true", "HAI_AUTH_PROVIDER": "unknown"},
    {"HAI_AUTH_ENABLED": "true", "HAI_AUTH_PROVIDER": "keycloak"},
    {
        "HAI_AUTH_ENABLED": "true",
        "HAI_AUTH_PROVIDER": "keycloak",
        "HAI_KEYCLOAK_API_CLIENT_ID": "Bad Client",
    },
])
def test_invalid_environment_configuration_is_rejected(environ):
    with pytest.raises(AuthorizationConfigurationError):
        create_environment_authorization_policy(environ=environ)
