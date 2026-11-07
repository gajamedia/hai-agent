from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

from api.security.authentication import AuthenticationProviderUnavailable
from api.security.bearer_token_provider import BearerTokenAuthenticationProvider
from api.security.jwks_jwt_verifier import AsyncJWKSJWTVerifier


class AuthenticationConfigurationError(ValueError):
    """Raised when enabled authentication has missing or invalid configuration."""


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "")
    if not isinstance(value, str) or not value.strip():
        raise AuthenticationConfigurationError(
            f"Required authentication setting {name} is missing."
        )
    return value.strip()


def _integer(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError) as exc:
        raise AuthenticationConfigurationError(
            f"Authentication setting {name} must be an integer."
        ) from exc
    if value < minimum:
        raise AuthenticationConfigurationError(
            f"Authentication setting {name} must be >= {minimum}."
        )
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except (TypeError, ValueError) as exc:
        raise AuthenticationConfigurationError(
            f"Authentication setting {name} must be a number."
        ) from exc
    if value <= 0:
        raise AuthenticationConfigurationError(
            f"Authentication setting {name} must be greater than zero."
        )
    return value


def _claim_name(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default)
    if not value.strip():
        raise AuthenticationConfigurationError(
            f"Authentication setting {name} cannot be empty."
        )
    return value.strip()


@dataclass(frozen=True, slots=True)
class OIDCAuthenticationSettings:
    jwks_url: str
    issuer: str
    audience: tuple[str, ...]
    algorithms: tuple[str, ...]
    leeway_seconds: int
    jwks_cache_ttl_seconds: int
    http_timeout_seconds: float
    max_jwks_bytes: int
    max_token_length: int
    subject_claim: str
    actor_id_claim: str | None
    roles_claim: str
    permissions_claim: str
    issuer_claim: str

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> OIDCAuthenticationSettings | None:
        env = os.environ if environ is None else environ
        enabled_raw = env.get("HAI_AUTH_ENABLED", "false").strip().lower()
        if enabled_raw not in {"true", "false"}:
            raise AuthenticationConfigurationError(
                "HAI_AUTH_ENABLED must be exactly 'true' or 'false'."
            )
        if enabled_raw == "false":
            return None

        jwks_url = _required(env, "HAI_AUTH_JWKS_URL")
        issuer = _required(env, "HAI_AUTH_ISSUER")
        audience_raw = _required(env, "HAI_AUTH_AUDIENCE")
        audience = tuple(item.strip() for item in audience_raw.split(",") if item.strip())
        if not audience:
            raise AuthenticationConfigurationError(
                "HAI_AUTH_AUDIENCE must contain at least one value."
            )

        algorithms_raw = env.get("HAI_AUTH_ALGORITHMS", "RS256")
        algorithms = tuple(
            item.strip() for item in algorithms_raw.split(",") if item.strip()
        )
        if not algorithms:
            raise AuthenticationConfigurationError(
                "HAI_AUTH_ALGORITHMS must contain at least one algorithm."
            )

        parsed_url = urlparse(jwks_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise AuthenticationConfigurationError(
                "HAI_AUTH_JWKS_URL must be an absolute HTTPS URL without credentials."
            )

        actor_claim = env.get("HAI_AUTH_ACTOR_ID_CLAIM")
        if actor_claim is not None:
            actor_claim = actor_claim.strip()
            if not actor_claim:
                raise AuthenticationConfigurationError(
                    "HAI_AUTH_ACTOR_ID_CLAIM cannot be empty when supplied."
                )

        return cls(
            jwks_url=jwks_url,
            issuer=issuer,
            audience=audience,
            algorithms=algorithms,
            leeway_seconds=_integer(
                env, "HAI_AUTH_LEEWAY_SECONDS", 0, minimum=0
            ),
            jwks_cache_ttl_seconds=_integer(
                env, "HAI_AUTH_JWKS_CACHE_TTL_SECONDS", 300, minimum=1
            ),
            http_timeout_seconds=_positive_float(
                env, "HAI_AUTH_HTTP_TIMEOUT_SECONDS", 5.0
            ),
            max_jwks_bytes=_integer(
                env, "HAI_AUTH_MAX_JWKS_BYTES", 1_048_576, minimum=1
            ),
            max_token_length=_integer(
                env, "HAI_AUTH_MAX_TOKEN_LENGTH", 32_768, minimum=1
            ),
            subject_claim=_claim_name(
                env, "HAI_AUTH_SUBJECT_CLAIM", "sub"
            ),
            actor_id_claim=actor_claim,
            roles_claim=_claim_name(env, "HAI_AUTH_ROLES_CLAIM", "roles"),
            permissions_claim=_claim_name(
                env, "HAI_AUTH_PERMISSIONS_CLAIM", "permissions"
            ),
            issuer_claim=_claim_name(env, "HAI_AUTH_ISSUER_CLAIM", "iss"),
        )


def create_environment_authentication_provider(
    app=None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BearerTokenAuthenticationProvider | None:
    """Build the trusted Bearer provider from explicit environment settings.

    Returns None only when HAI_AUTH_ENABLED is false. When enabled, incomplete
    or invalid configuration raises AuthenticationConfigurationError so the
    application fails during startup rather than silently running unprotected.
    """
    env = os.environ if environ is None else environ
    provider_name = env.get("HAI_AUTH_PROVIDER", "oidc").strip().lower()
    if provider_name not in {"oidc", "keycloak"}:
        raise AuthenticationConfigurationError(
            "HAI_AUTH_PROVIDER must be either 'oidc' or 'keycloak'."
        )
    if provider_name == "keycloak":
        from api.security.keycloak_authentication import (
            create_keycloak_authentication_provider,
        )
        return create_keycloak_authentication_provider(app, environ=env)

    settings = OIDCAuthenticationSettings.from_environment(env)
    if settings is None:
        return None

    try:
        verifier = AsyncJWKSJWTVerifier(
            jwks_url=settings.jwks_url,
            issuer=settings.issuer,
            audience=settings.audience,
            algorithms=settings.algorithms,
            leeway_seconds=settings.leeway_seconds,
            cache_ttl_seconds=settings.jwks_cache_ttl_seconds,
            timeout_seconds=settings.http_timeout_seconds,
            max_jwks_bytes=settings.max_jwks_bytes,
            max_token_length=settings.max_token_length,
        )
        return BearerTokenAuthenticationProvider(
            verifier,
            subject_claim=settings.subject_claim,
            actor_id_claim=settings.actor_id_claim,
            roles_claim=settings.roles_claim,
            permissions_claim=settings.permissions_claim,
            issuer_claim=settings.issuer_claim,
        )
    except (ValueError, TypeError) as exc:
        raise AuthenticationConfigurationError(
            "OIDC/JWT authentication configuration is invalid."
        ) from exc


__all__ = [
    "AuthenticationConfigurationError",
    "OIDCAuthenticationSettings",
    "create_environment_authentication_provider",
]
