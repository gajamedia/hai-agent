from __future__ import annotations

from collections.abc import Mapping
from inspect import isawaitable
import re
from typing import Any
from urllib.parse import urlparse

from jwt import InvalidTokenError

from api.security.authentication_config import (
    AuthenticationConfigurationError,
    _integer,
    _positive_float,
    _required,
)
from api.security.bearer_token_provider import BearerTokenAuthenticationProvider
from api.security.jwks_jwt_verifier import AsyncJWKSJWTVerifier


class KeycloakClaimsVerifier:
    """Validate Keycloak access-token claims and expose namespaced roles.

    The wrapped verifier performs signature, issuer, audience, exp and nbf
    verification. This adapter additionally requires Keycloak's access-token
    marker and maps only realm roles plus roles belonging to the configured
    API client. Roles from unrelated clients are ignored.
    """

    def __init__(
        self,
        verifier: Any,
        *,
        api_client_id: str,
        require_access_token_type: bool = True,
    ) -> None:
        if verifier is None or not callable(getattr(verifier, "verify", None)):
            raise ValueError("verifier must implement verify(token).")
        if not isinstance(api_client_id, str) or not api_client_id.strip():
            raise ValueError("api_client_id must be a non-empty string.")
        self._verifier = verifier
        self._api_client_id = api_client_id.strip()
        self._require_access_token_type = bool(require_access_token_type)

    @staticmethod
    def _roles(value: Any, *, claim_name: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise InvalidTokenError(
                f"Keycloak {claim_name} must be an array of non-empty strings."
            )
        return tuple(item.strip() for item in value)

    async def verify(self, token: str) -> Mapping[str, Any]:
        claims = self._verifier.verify(token)
        if isawaitable(claims):
            claims = await claims
        if not isinstance(claims, Mapping):
            raise InvalidTokenError("Verified Keycloak claims must be an object.")

        if self._require_access_token_type and claims.get("typ") != "Bearer":
            raise InvalidTokenError("A Keycloak access token is required.")

        realm_access = claims.get("realm_access", {})
        if realm_access is None:
            realm_access = {}
        if not isinstance(realm_access, Mapping):
            raise InvalidTokenError("Keycloak realm_access claim must be an object.")
        realm_roles = self._roles(
            realm_access.get("roles"),
            claim_name="realm_access.roles",
        )

        resource_access = claims.get("resource_access", {})
        if resource_access is None:
            resource_access = {}
        if not isinstance(resource_access, Mapping):
            raise InvalidTokenError("Keycloak resource_access claim must be an object.")

        client_access = resource_access.get(self._api_client_id, {})
        if client_access is None:
            client_access = {}
        if not isinstance(client_access, Mapping):
            raise InvalidTokenError(
                "Keycloak API client resource_access claim must be an object."
            )
        client_roles = self._roles(
            client_access.get("roles"),
            claim_name=f"resource_access.{self._api_client_id}.roles",
        )

        # Namespace roles to prevent a realm role and a client role with the
        # same spelling from colliding in the application's PBAC policy.
        normalized_roles = tuple(
            [f"realm:{role}" for role in realm_roles]
            + [
                f"client:{self._api_client_id}:{role}"
                for role in client_roles
            ]
        )

        normalized = dict(claims)
        normalized["hai_roles"] = normalized_roles

        # Permissions remain explicit application claims only. Keycloak roles
        # are never silently promoted into permissions.
        permissions = claims.get("permissions", ())
        if not isinstance(permissions, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip()
            for item in permissions
        ):
            raise InvalidTokenError(
                "Keycloak permissions claim must be an array of strings."
            )
        normalized["hai_permissions"] = tuple(
            item.strip() for item in permissions
        )
        return normalized

    async def aclose(self) -> None:
        close = getattr(self._verifier, "aclose", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result


def create_keycloak_authentication_provider(
    app=None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BearerTokenAuthenticationProvider | None:
    """Create a Keycloak-specific provider from environment configuration.

    HAI_AUTH_ENABLED=false disables authentication. When enabled, Keycloak
    issuer and JWKS URLs are derived from the same base URL and realm, avoiding
    accidental issuer/JWKS mismatches.
    """
    import os

    env = os.environ if environ is None else environ
    enabled_raw = env.get("HAI_AUTH_ENABLED", "false").strip().lower()
    if enabled_raw not in {"true", "false"}:
        raise AuthenticationConfigurationError(
            "HAI_AUTH_ENABLED must be exactly 'true' or 'false'."
        )
    if enabled_raw == "false":
        return None

    base_url = _required(env, "HAI_KEYCLOAK_BASE_URL").rstrip("/")
    realm = _required(env, "HAI_KEYCLOAK_REALM")
    client_id = _required(env, "HAI_KEYCLOAK_API_CLIENT_ID")

    parsed = urlparse(base_url)
    insecure_raw = env.get(
        "HAI_KEYCLOAK_ALLOW_INSECURE_HTTP", "false"
    ).strip().lower()
    if insecure_raw not in {"true", "false"}:
        raise AuthenticationConfigurationError(
            "HAI_KEYCLOAK_ALLOW_INSECURE_HTTP must be 'true' or 'false'."
        )
    allow_insecure = insecure_raw == "true"
    is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not re.fullmatch(r"[A-Za-z0-9._~-]+", realm) or realm in {".", ".."}:
        raise AuthenticationConfigurationError(
            "HAI_KEYCLOAK_REALM must be a safe single path segment."
        )
    if (
        not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.scheme not in {"https", "http"}
        or (parsed.scheme != "https" and not (allow_insecure and is_loopback))
    ):
        raise AuthenticationConfigurationError(
            "HAI_KEYCLOAK_BASE_URL must be HTTPS; HTTP is allowed only for explicit loopback development."
        )
    issuer = f"{base_url}/realms/{realm}"
    jwks_url = f"{issuer}/protocol/openid-connect/certs"
    algorithms_raw = env.get("HAI_AUTH_ALGORITHMS", "RS256")
    algorithms = tuple(
        value.strip() for value in algorithms_raw.split(",") if value.strip()
    )
    if not algorithms:
        raise AuthenticationConfigurationError(
            "HAI_AUTH_ALGORITHMS must contain at least one algorithm."
        )

    try:
        cryptographic_verifier = AsyncJWKSJWTVerifier(
            jwks_url=jwks_url,
            issuer=issuer,
            audience=(client_id,),
            algorithms=algorithms,
            leeway_seconds=_integer(
                env, "HAI_AUTH_LEEWAY_SECONDS", 0, minimum=0
            ),
            cache_ttl_seconds=_integer(
                env, "HAI_AUTH_JWKS_CACHE_TTL_SECONDS", 300, minimum=1
            ),
            timeout_seconds=_positive_float(
                env, "HAI_AUTH_HTTP_TIMEOUT_SECONDS", 5.0
            ),
            max_jwks_bytes=_integer(
                env, "HAI_AUTH_MAX_JWKS_BYTES", 1_048_576, minimum=1
            ),
            max_token_length=_integer(
                env, "HAI_AUTH_MAX_TOKEN_LENGTH", 32_768, minimum=1
            ),
            allow_insecure_http=parsed.scheme == "http",
        )
        keycloak_verifier = KeycloakClaimsVerifier(
            cryptographic_verifier,
            api_client_id=client_id,
        )
        return BearerTokenAuthenticationProvider(
            keycloak_verifier,
            roles_claim="hai_roles",
            permissions_claim="hai_permissions",
        )
    except (ValueError, TypeError) as exc:
        raise AuthenticationConfigurationError(
            "Keycloak authentication configuration is invalid."
        ) from exc


__all__ = [
    "KeycloakClaimsVerifier",
    "create_keycloak_authentication_provider",
]
