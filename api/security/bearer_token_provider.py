from __future__ import annotations

from inspect import isawaitable
from typing import Any, Mapping, Protocol, runtime_checkable

from fastapi import Request

from api.security.authentication import (
    AuthenticatedPrincipal,
    AuthenticationProviderUnavailable,
    AuthenticationRejected,
    InvalidAuthenticatedPrincipal,
)


@runtime_checkable
class VerifiedTokenVerifier(Protocol):
    """Application-owned cryptographic token verifier."""

    def verify(self, token: str) -> Mapping[str, Any]:
        ...


class BearerTokenAuthenticationProvider:
    """Provider-neutral Bearer adapter around a trusted token verifier."""

    def __init__(
        self,
        verifier: VerifiedTokenVerifier,
        *,
        subject_claim: str = "sub",
        actor_id_claim: str | None = None,
        roles_claim: str = "roles",
        permissions_claim: str = "permissions",
        issuer_claim: str = "iss",
    ) -> None:
        if verifier is None or not callable(getattr(verifier, "verify", None)):
            raise ValueError("verifier must implement verify(token).")
        for name, value in (
            ("subject_claim", subject_claim),
            ("roles_claim", roles_claim),
            ("permissions_claim", permissions_claim),
            ("issuer_claim", issuer_claim),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if actor_id_claim is not None and (
            not isinstance(actor_id_claim, str) or not actor_id_claim.strip()
        ):
            raise ValueError("actor_id_claim must be None or a non-empty string.")

        self._verifier = verifier
        self._subject_claim = subject_claim
        self._actor_id_claim = actor_id_claim
        self._roles_claim = roles_claim
        self._permissions_claim = permissions_claim
        self._issuer_claim = issuer_claim

    async def authenticate(self, request: Request) -> AuthenticatedPrincipal | None:
        authorization = request.headers.get("authorization")
        if authorization is None:
            return None

        parts = authorization.strip().split()
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
            raise AuthenticationRejected("Authorization must use one Bearer token.")

        try:
            claims = self._verifier.verify(parts[1])
            if isawaitable(claims):
                claims = await claims
        except (AuthenticationProviderUnavailable, AuthenticationRejected):
            raise
        except Exception as exc:
            raise AuthenticationRejected("Bearer token validation failed.") from exc

        if not isinstance(claims, Mapping):
            raise InvalidAuthenticatedPrincipal(
                "Verified token claims must be a mapping."
            )

        subject = claims.get(self._subject_claim)
        actor_id = (
            claims.get(self._actor_id_claim)
            if self._actor_id_claim is not None
            else subject
        )
        issuer = claims.get(self._issuer_claim)

        if not isinstance(subject, str) or not subject.strip():
            raise InvalidAuthenticatedPrincipal(
                "Verified token is missing a valid subject claim."
            )
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise InvalidAuthenticatedPrincipal(
                "Verified token is missing a valid actor identity claim."
            )
        if issuer is not None and (
            not isinstance(issuer, str) or not issuer.strip()
        ):
            raise InvalidAuthenticatedPrincipal(
                "Verified token issuer claim must be a non-empty string."
            )

        roles = claims.get(self._roles_claim, ())
        permissions = claims.get(self._permissions_claim, ())
        if not isinstance(roles, (list, tuple, set, frozenset)):
            raise InvalidAuthenticatedPrincipal(
                "Verified roles claim must be an array of strings."
            )
        if not isinstance(permissions, (list, tuple, set, frozenset)):
            raise InvalidAuthenticatedPrincipal(
                "Verified permissions claim must be an array of strings."
            )

        auth_time = claims.get("auth_time")
        if isinstance(auth_time, bool) or not isinstance(auth_time, int) or auth_time <= 0:
            auth_time = None  # a missing or odd value must never count as a recent login
        return AuthenticatedPrincipal(
            auth_time=auth_time,
            actor_id=actor_id,
            roles=tuple(roles),
            permissions=tuple(permissions),
            issuer=issuer,
            subject=subject,
        )

    async def aclose(self) -> None:
        """Close the verifier's async resources during application shutdown."""
        close = getattr(self._verifier, "aclose", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result


__all__ = ["VerifiedTokenVerifier", "BearerTokenAuthenticationProvider"]
