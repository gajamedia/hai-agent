from __future__ import annotations

from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Protocol, runtime_checkable

from fastapi import Request


class AuthenticationError(RuntimeError):
    """Base error for trusted authentication failures."""


class AuthenticationProviderUnavailable(AuthenticationError):
    """Raised when no application-owned authentication provider is configured."""


class AuthenticationRejected(AuthenticationError):
    """Raised when a provider cannot authenticate the request."""


class InvalidAuthenticatedPrincipal(AuthenticationError, ValueError):
    """Raised when a provider returns an invalid principal."""


def _normalize_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise InvalidAuthenticatedPrincipal(
            f"{field_name} must be a sequence of strings, not a string."
        )
    try:
        values = tuple(value)
    except TypeError as exc:
        raise InvalidAuthenticatedPrincipal(
            f"{field_name} must be a sequence of strings."
        ) from exc

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise InvalidAuthenticatedPrincipal(
                f"{field_name} entries must be non-empty strings."
            )
        candidate = item.strip()
        if candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    """Verified identity returned by an application-owned auth provider.

    `actor_id` is the stable application identity used for audit attribution.
    Roles and permissions are claims supplied by the trusted provider, not by
    request payloads or arbitrary headers. This type does not store credentials
    or access tokens.
    """

    actor_id: str
    roles: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    issuer: str | None = None
    subject: str | None = None
    is_authenticated: bool = True
    # When the user last typed their credentials (the token's auth_time, seconds since the epoch).
    # Sensitive actions require this to be recent ("step-up"); None means the provider did not say.
    auth_time: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str) or not self.actor_id.strip():
            raise InvalidAuthenticatedPrincipal(
                "actor_id must be a non-empty stable application identity."
            )
        object.__setattr__(self, "actor_id", self.actor_id.strip())
        object.__setattr__(
            self,
            "roles",
            _normalize_string_tuple(self.roles, "roles"),
        )
        object.__setattr__(
            self,
            "permissions",
            _normalize_string_tuple(self.permissions, "permissions"),
        )
        for field_name in ("issuer", "subject"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise InvalidAuthenticatedPrincipal(
                        f"{field_name} must be None or a non-empty string."
                    )
                object.__setattr__(self, field_name, value.strip())
        if self.auth_time is not None and (
            isinstance(self.auth_time, bool) or not isinstance(self.auth_time, int)
            or self.auth_time <= 0
        ):
            raise InvalidAuthenticatedPrincipal("auth_time must be None or positive epoch seconds.")
        if self.is_authenticated is not True:
            raise InvalidAuthenticatedPrincipal(
                "AuthenticatedPrincipal must represent a verified identity."
            )


@runtime_checkable
class AuthenticationProvider(Protocol):
    """Application-owned adapter to a real identity provider.

    Implementations must validate credentials/signatures/issuer/audience and
    return a principal only after successful verification. They must never
    trust actor IDs, roles, or permissions supplied directly by request JSON
    or unverified headers.
    """

    async def authenticate(self, request: Request) -> AuthenticatedPrincipal | None:
        ...


def get_authentication_provider(request: Request) -> AuthenticationProvider:
    provider = getattr(
        request.app.state,
        "authentication_provider",
        None,
    )
    if provider is None:
        raise AuthenticationProviderUnavailable(
            "No trusted authentication provider is configured."
        )
    authenticate = getattr(provider, "authenticate", None)
    if not callable(authenticate):
        raise AuthenticationProviderUnavailable(
            "Configured authentication provider does not implement authenticate(request)."
        )
    return provider


async def authenticate_request(request: Request) -> AuthenticatedPrincipal:
    """Authenticate a request through the configured trusted provider.

    This function intentionally has no development fallback, anonymous actor,
    header parsing, or request-body identity handling.
    """
    provider = get_authentication_provider(request)
    try:
        result = provider.authenticate(request)
        if isawaitable(result):
            result = await result
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthenticationRejected(
            "The authentication provider failed to verify the request."
        ) from exc

    if result is None:
        raise AuthenticationRejected("The request is not authenticated.")
    if not isinstance(result, AuthenticatedPrincipal):
        raise InvalidAuthenticatedPrincipal(
            "Authentication provider must return AuthenticatedPrincipal or None."
        )
    return result


def attach_authenticated_principal(
    request: Request,
    principal: AuthenticatedPrincipal,
) -> None:
    """Publish a verified principal to request state for trusted API middleware.

    Call this only after a configured AuthenticationProvider has successfully
    verified the request. Never call it using request-supplied identity data.
    """
    if not isinstance(principal, AuthenticatedPrincipal):
        raise InvalidAuthenticatedPrincipal(
            "Only an AuthenticatedPrincipal can be attached to request state."
        )
    request.state.authenticated_principal = principal
    # Compatibility bridge for the existing Stage 2.7B/2.7C route contract.
    request.state.capability_actor = principal.actor_id


def get_attached_principal(request: Request) -> AuthenticatedPrincipal:
    """Read an already verified principal attached by trusted middleware."""
    principal = getattr(request.state, "authenticated_principal", None)
    if not isinstance(principal, AuthenticatedPrincipal):
        raise AuthenticationRejected(
            "No verified authenticated principal is attached to this request."
        )
    return principal


__all__ = [
    "AuthenticationError",
    "AuthenticationProviderUnavailable",
    "AuthenticationRejected",
    "InvalidAuthenticatedPrincipal",
    "AuthenticatedPrincipal",
    "AuthenticationProvider",
    "get_authentication_provider",
    "authenticate_request",
    "attach_authenticated_principal",
    "get_attached_principal",
]
