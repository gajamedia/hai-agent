from __future__ import annotations

import asyncio

import pytest
from jwt import InvalidTokenError

from api.security.authentication import AuthenticatedPrincipal
from api.security.authentication_config import (
    AuthenticationConfigurationError,
    create_environment_authentication_provider,
)
from api.security.bearer_token_provider import BearerTokenAuthenticationProvider
from api.security.keycloak_authentication import (
    KeycloakClaimsVerifier,
    create_keycloak_authentication_provider,
)


def keycloak_env(**overrides):
    env = {
        "HAI_AUTH_ENABLED": "true",
        "HAI_AUTH_PROVIDER": "keycloak",
        "HAI_KEYCLOAK_BASE_URL": "https://sso.example.test",
        "HAI_KEYCLOAK_REALM": "hai-agent",
        "HAI_KEYCLOAK_API_CLIENT_ID": "hai-agent-api",
    }
    env.update(overrides)
    return env


class StaticVerifier:
    def __init__(self, claims):
        self.claims = claims
        self.closed = False

    def verify(self, token):
        if token != "signed-access-token":
            raise InvalidTokenError("bad token")
        return self.claims

    async def aclose(self):
        self.closed = True


def test_keycloak_maps_realm_and_api_client_roles_with_namespaces():
    verifier = KeycloakClaimsVerifier(
        StaticVerifier({
            "sub": "user-123",
            "iss": "https://sso.example.test/realms/hai-agent",
            "typ": "Bearer",
            "realm_access": {"roles": ["operator", "reviewer"]},
            "resource_access": {
                "hai-agent-api": {"roles": ["proposal-reader"]},
                "other-client": {"roles": ["irrelevant-admin"]},
            },
            "permissions": ["capability:proposal:read"],
        }),
        api_client_id="hai-agent-api",
    )
    claims = asyncio.run(verifier.verify("signed-access-token"))
    assert claims["hai_roles"] == (
        "realm:operator",
        "realm:reviewer",
        "client:hai-agent-api:proposal-reader",
    )
    assert claims["hai_permissions"] == ("capability:proposal:read",)


def test_keycloak_does_not_promote_roles_to_permissions():
    verifier = KeycloakClaimsVerifier(
        StaticVerifier({
            "sub": "user-123",
            "iss": "https://sso.example.test/realms/hai-agent",
            "typ": "Bearer",
            "realm_access": {"roles": ["admin"]},
            "resource_access": {},
        }),
        api_client_id="hai-agent-api",
    )
    claims = asyncio.run(verifier.verify("signed-access-token"))
    assert claims["hai_roles"] == ("realm:admin",)
    assert claims["hai_permissions"] == ()


def test_keycloak_rejects_id_token_or_missing_access_token_marker():
    verifier = KeycloakClaimsVerifier(
        StaticVerifier({"sub": "user-123", "typ": "ID"}),
        api_client_id="hai-agent-api",
    )
    with pytest.raises(InvalidTokenError, match="access token"):
        asyncio.run(verifier.verify("signed-access-token"))


@pytest.mark.parametrize(
    "claims",
    [
        {"sub": "u", "typ": "Bearer", "realm_access": []},
        {"sub": "u", "typ": "Bearer", "realm_access": {"roles": "admin"}},
        {"sub": "u", "typ": "Bearer", "resource_access": []},
        {
            "sub": "u",
            "typ": "Bearer",
            "resource_access": {"hai-agent-api": {"roles": [1]}},
        },
        {"sub": "u", "typ": "Bearer", "permissions": "admin"},
    ],
)
def test_keycloak_rejects_malformed_role_or_permission_claims(claims):
    verifier = KeycloakClaimsVerifier(
        StaticVerifier(claims),
        api_client_id="hai-agent-api",
    )
    with pytest.raises(InvalidTokenError):
        asyncio.run(verifier.verify("signed-access-token"))


def test_keycloak_provider_derives_issuer_and_jwks_from_same_realm():
    provider = create_keycloak_authentication_provider(environ=keycloak_env())
    assert isinstance(provider, BearerTokenAuthenticationProvider)
    verifier = provider._verifier
    assert verifier._api_client_id == "hai-agent-api"
    cryptographic = verifier._verifier
    assert cryptographic._issuer == "https://sso.example.test/realms/hai-agent"
    assert cryptographic._jwks_url == (
        "https://sso.example.test/realms/hai-agent/protocol/openid-connect/certs"
    )
    assert cryptographic._audience == ("hai-agent-api",)
    asyncio.run(provider.aclose())
    assert cryptographic._closed is True


def test_keycloak_provider_disabled_by_default():
    assert create_keycloak_authentication_provider(environ={}) is None


def test_keycloak_provider_requires_realm_settings_when_enabled():
    with pytest.raises(AuthenticationConfigurationError, match="HAI_KEYCLOAK"):
        create_keycloak_authentication_provider(
            environ={"HAI_AUTH_ENABLED": "true", "HAI_AUTH_PROVIDER": "keycloak"}
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://sso.example.test",
        "https://user:pass@sso.example.test",
        "https://sso.example.test/path?x=1",
        "file:///tmp/keycloak",
    ],
)
def test_keycloak_requires_https_outside_explicit_local_development(base_url):
    with pytest.raises(AuthenticationConfigurationError):
        create_keycloak_authentication_provider(
            environ=keycloak_env(HAI_KEYCLOAK_BASE_URL=base_url)
        )


def test_keycloak_allows_http_only_for_explicit_loopback_development():
    provider = create_keycloak_authentication_provider(
        environ=keycloak_env(
            HAI_KEYCLOAK_BASE_URL="http://127.0.0.1:8080",
            HAI_KEYCLOAK_ALLOW_INSECURE_HTTP="true",
        )
    )
    assert isinstance(provider, BearerTokenAuthenticationProvider)
    asyncio.run(provider.aclose())


def test_environment_factory_dispatches_to_keycloak():
    provider = create_environment_authentication_provider(environ=keycloak_env())
    assert isinstance(provider, BearerTokenAuthenticationProvider)
    assert isinstance(provider._verifier, KeycloakClaimsVerifier)
    asyncio.run(provider.aclose())


def test_environment_factory_rejects_unknown_provider():
    with pytest.raises(AuthenticationConfigurationError, match="HAI_AUTH_PROVIDER"):
        create_environment_authentication_provider(
            environ=keycloak_env(HAI_AUTH_PROVIDER="unknown")
        )


def test_keycloak_roles_are_the_only_roles_exposed_to_pbac():
    source = StaticVerifier({
        "sub": "user-123",
        "iss": "https://sso.example.test/realms/hai-agent",
        "typ": "Bearer",
        "realm_access": {"roles": ["reviewer"]},
        "resource_access": {"hai-agent-api": {"roles": ["proposal-reader"]}},
        "permissions": [],
    })
    normalized = KeycloakClaimsVerifier(
        source, api_client_id="hai-agent-api"
    )
    normalized = asyncio.run(normalized.verify("signed-access-token"))
    assert "roles" not in normalized
    assert normalized["hai_roles"] == (
        "realm:reviewer",
        "client:hai-agent-api:proposal-reader",
    )



def test_keycloak_bearer_provider_builds_authenticated_principal():
    from starlette.requests import Request

    verifier = KeycloakClaimsVerifier(
        StaticVerifier({
            "sub": "user-456",
            "iss": "https://sso.example.test/realms/hai-agent",
            "typ": "Bearer",
            "realm_access": {"roles": ["reviewer"]},
            "resource_access": {
                "hai-agent-api": {"roles": ["proposal-reader"]}
            },
            "permissions": ["capability:proposal:read"],
        }),
        api_client_id="hai-agent-api",
    )
    provider = BearerTokenAuthenticationProvider(
        verifier,
        roles_claim="hai_roles",
        permissions_claim="hai_permissions",
    )
    request = Request({
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": "/api/v1/capability-proposals",
        "raw_path": b"/api/v1/capability-proposals",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer signed-access-token")],
        "server": ("testserver", 443),
        "client": ("127.0.0.1", 12345),
    })
    principal = asyncio.run(provider.authenticate(request))
    assert isinstance(principal, AuthenticatedPrincipal)
    assert principal.actor_id == "user-456"
    assert principal.subject == "user-456"
    assert principal.issuer == "https://sso.example.test/realms/hai-agent"
    assert principal.roles == (
        "realm:reviewer",
        "client:hai-agent-api:proposal-reader",
    )
    assert principal.permissions == ("capability:proposal:read",)
