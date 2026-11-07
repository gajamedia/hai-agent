from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from api.security.authentication import AuthenticationProviderUnavailable
from api.security.authentication_config import (
    AuthenticationConfigurationError,
    OIDCAuthenticationSettings,
    create_environment_authentication_provider,
)
from api.security.bearer_token_provider import BearerTokenAuthenticationProvider


def enabled_env(**overrides):
    values = {
        "HAI_AUTH_ENABLED": "true",
        "HAI_AUTH_JWKS_URL": "https://identity.example/.well-known/jwks.json",
        "HAI_AUTH_ISSUER": "https://identity.example/",
        "HAI_AUTH_AUDIENCE": "hai-api",
    }
    values.update(overrides)
    return values


def test_authentication_is_disabled_by_default():
    assert OIDCAuthenticationSettings.from_environment({}) is None
    assert create_environment_authentication_provider(environ={}) is None


def test_enabled_authentication_requires_all_core_settings():
    with pytest.raises(AuthenticationConfigurationError, match="JWKS_URL"):
        OIDCAuthenticationSettings.from_environment(
            {"HAI_AUTH_ENABLED": "true"}
        )


def test_enabled_authentication_parses_csv_and_optional_claims():
    settings = OIDCAuthenticationSettings.from_environment(
        enabled_env(
            HAI_AUTH_AUDIENCE="hai-api,hai-admin",
            HAI_AUTH_ALGORITHMS="RS256,ES256",
            HAI_AUTH_ACTOR_ID_CLAIM="uid",
            HAI_AUTH_ROLES_CLAIM="groups",
            HAI_AUTH_PERMISSIONS_CLAIM="permissions",
            HAI_AUTH_LEEWAY_SECONDS="15",
            HAI_AUTH_JWKS_CACHE_TTL_SECONDS="120",
        )
    )
    assert settings is not None
    assert settings.audience == ("hai-api", "hai-admin")
    assert settings.algorithms == ("RS256", "ES256")
    assert settings.actor_id_claim == "uid"
    assert settings.roles_claim == "groups"
    assert settings.leeway_seconds == 15
    assert settings.jwks_cache_ttl_seconds == 120


@pytest.mark.parametrize("value", ["1", "yes", "on", "", "TRUEE"])
def test_authentication_enabled_flag_is_strict(value):
    with pytest.raises(AuthenticationConfigurationError, match="HAI_AUTH_ENABLED"):
        OIDCAuthenticationSettings.from_environment(
            {"HAI_AUTH_ENABLED": value}
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://identity.example/jwks",
        "file:///tmp/jwks.json",
        "https://user:password@identity.example/jwks",
        "/relative/jwks",
    ],
)
def test_jwks_url_must_be_secure_absolute_url(url):
    with pytest.raises(AuthenticationConfigurationError, match="HTTPS"):
        OIDCAuthenticationSettings.from_environment(
            enabled_env(HAI_AUTH_JWKS_URL=url)
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HAI_AUTH_LEEWAY_SECONDS", "-1"),
        ("HAI_AUTH_JWKS_CACHE_TTL_SECONDS", "0"),
        ("HAI_AUTH_MAX_JWKS_BYTES", "0"),
        ("HAI_AUTH_MAX_TOKEN_LENGTH", "-10"),
        ("HAI_AUTH_HTTP_TIMEOUT_SECONDS", "0"),
        ("HAI_AUTH_HTTP_TIMEOUT_SECONDS", "not-a-number"),
    ],
)
def test_numeric_settings_are_validated(name, value):
    with pytest.raises(AuthenticationConfigurationError):
        OIDCAuthenticationSettings.from_environment(enabled_env(**{name: value}))


def test_unsupported_symmetric_algorithm_is_rejected_during_provider_creation():
    with pytest.raises(AuthenticationConfigurationError):
        create_environment_authentication_provider(
            environ=enabled_env(HAI_AUTH_ALGORITHMS="HS256")
        )


def test_provider_factory_returns_bearer_adapter_and_closes_http_client():
    provider = create_environment_authentication_provider(environ=enabled_env())
    assert isinstance(provider, BearerTokenAuthenticationProvider)
    verifier = provider._verifier
    assert verifier._closed is False
    asyncio.run(provider.aclose())
    assert verifier._closed is True


def test_create_app_uses_environment_factory_when_no_custom_factory(monkeypatch):
    from api.main import create_app

    monkeypatch.setenv("HAI_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "HAI_AUTH_JWKS_URL",
        "https://identity.example/.well-known/jwks.json",
    )
    monkeypatch.setenv("HAI_AUTH_ISSUER", "https://identity.example/")
    monkeypatch.setenv("HAI_AUTH_AUDIENCE", "hai-api")

    application = create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=lambda: object(),
    )
    with TestClient(application):
        assert isinstance(
            application.state.authentication_provider,
            BearerTokenAuthenticationProvider,
        )
    assert application.state.authentication_provider is None


def test_create_app_fails_startup_for_incomplete_enabled_configuration(monkeypatch):
    from api.main import create_app

    monkeypatch.setenv("HAI_AUTH_ENABLED", "true")
    monkeypatch.delenv("HAI_AUTH_JWKS_URL", raising=False)
    monkeypatch.delenv("HAI_AUTH_ISSUER", raising=False)
    monkeypatch.delenv("HAI_AUTH_AUDIENCE", raising=False)

    application = create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=lambda: object(),
    )
    with pytest.raises(AuthenticationConfigurationError):
        with TestClient(application):
            pass
