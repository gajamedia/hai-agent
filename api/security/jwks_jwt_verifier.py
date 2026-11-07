from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import httpx
import jwt
from jwt import InvalidTokenError, PyJWK

from api.security.authentication import AuthenticationProviderUnavailable


_ASYMMETRIC_ALGORITHMS = frozenset(
    {
        "RS256", "RS384", "RS512",
        "PS256", "PS384", "PS512",
        "ES256", "ES384", "ES512",
        "EdDSA",
    }
)
_PRIVATE_JWK_FIELDS = frozenset({"d", "p", "q", "dp", "dq", "qi", "oth", "k"})


class JWKSConfigurationError(ValueError):
    """Invalid or unsafe JWKS verifier configuration."""


class JWKSUnavailable(AuthenticationProviderUnavailable):
    """The configured JWKS endpoint could not provide usable signing keys."""


class AsyncJWKSJWTVerifier:
    """Async JWT verifier using an HTTPS JWKS endpoint and PyJWT.

    This verifier accepts asymmetric signing algorithms only. It validates the
    JWT signature and required issuer, audience, expiration, not-before (when
    present), and subject claims. JWKS keys are cached for a bounded TTL and
    refreshed when an unknown key ID is encountered.

    The caller owns lifecycle when an external AsyncClient is injected.
    """

    def __init__(
        self,
        *,
        jwks_url: str,
        issuer: str,
        audience: str | Sequence[str],
        algorithms: Sequence[str] = ("RS256",),
        leeway_seconds: int = 0,
        cache_ttl_seconds: int = 300,
        timeout_seconds: float = 5.0,
        max_jwks_bytes: int = 1_048_576,
        max_token_length: int = 32_768,
        client: httpx.AsyncClient | None = None,
        allow_insecure_http: bool = False,
    ) -> None:
        parsed = urlparse(jwks_url) if isinstance(jwks_url, str) else None
        if (
            parsed is None
            or parsed.scheme not in ({"https", "http"} if allow_insecure_http else {"https"})
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise JWKSConfigurationError(
                "jwks_url must be an absolute HTTPS URL without embedded credentials."
            )
        if not isinstance(issuer, str) or not issuer.strip():
            raise JWKSConfigurationError("issuer must be a non-empty string.")
        if isinstance(audience, str):
            normalized_audience = (audience.strip(),)
        elif isinstance(audience, Sequence) and not isinstance(audience, (bytes, bytearray)):
            normalized_audience = tuple(
                item.strip() for item in audience
                if isinstance(item, str) and item.strip()
            )
            if len(normalized_audience) != len(audience):
                raise JWKSConfigurationError("audience entries must be non-empty strings.")
        else:
            raise JWKSConfigurationError("audience must be a string or sequence of strings.")
        if not normalized_audience:
            raise JWKSConfigurationError("audience must contain at least one value.")

        if isinstance(algorithms, (str, bytes)) or not algorithms:
            raise JWKSConfigurationError("algorithms must be a non-empty sequence.")
        normalized_algorithms = tuple(dict.fromkeys(algorithms))
        if any(
            not isinstance(algorithm, str)
            or algorithm not in _ASYMMETRIC_ALGORITHMS
            for algorithm in normalized_algorithms
        ):
            raise JWKSConfigurationError(
                "Only explicitly supported asymmetric JWT algorithms are allowed."
            )
        for name, value in (
            ("leeway_seconds", leeway_seconds),
            ("cache_ttl_seconds", cache_ttl_seconds),
            ("max_jwks_bytes", max_jwks_bytes),
            ("max_token_length", max_token_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise JWKSConfigurationError(f"{name} must be a non-negative integer.")
        if cache_ttl_seconds < 1 or max_jwks_bytes < 1 or max_token_length < 1:
            raise JWKSConfigurationError("Cache and size limits must be positive.")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise JWKSConfigurationError("timeout_seconds must be positive.")

        self._jwks_url = jwks_url
        self._issuer = issuer.strip()
        self._audience = normalized_audience
        self._algorithms = normalized_algorithms
        self._leeway = leeway_seconds
        self._cache_ttl = cache_ttl_seconds
        self._max_jwks_bytes = max_jwks_bytes
        self._max_token_length = max_token_length
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            headers={"Accept": "application/json"},
        )
        self._owns_client = client is None
        self._keys: dict[str, dict[str, Any]] = {}
        self._cache_expires_at = 0.0
        self._refresh_lock = asyncio.Lock()
        self._closed = False

    async def verify(self, token: str) -> Mapping[str, Any]:
        if self._closed:
            raise JWKSUnavailable("JWT verifier is closed.")
        if not isinstance(token, str) or not token or len(token) > self._max_token_length:
            raise InvalidTokenError("Token is empty or exceeds the configured size limit.")

        try:
            header = jwt.get_unverified_header(token)
        except Exception as exc:
            raise InvalidTokenError("Malformed JWT header.") from exc

        algorithm = header.get("alg")
        kid = header.get("kid")
        if algorithm not in self._algorithms:
            raise InvalidTokenError("JWT algorithm is not allowed.")
        if not isinstance(kid, str) or not kid.strip():
            raise InvalidTokenError("JWT must contain a non-empty key ID.")

        jwk = await self._get_signing_jwk(kid.strip(), force_refresh=False)
        if jwk is None:
            jwk = await self._get_signing_jwk(kid.strip(), force_refresh=True)
        if jwk is None:
            raise InvalidTokenError("No matching trusted signing key was found.")

        self._validate_jwk_metadata(jwk, algorithm)
        try:
            signing_key = PyJWK.from_dict(jwk, algorithm=algorithm).key
            claims = jwt.decode(
                token,
                key=signing_key,
                algorithms=list(self._algorithms),
                issuer=self._issuer,
                audience=list(self._audience),
                leeway=self._leeway,
                options={
                    "require": ["exp", "iss", "aud", "sub"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )
        except InvalidTokenError:
            raise
        except Exception as exc:
            raise InvalidTokenError("JWT signature or claims validation failed.") from exc

        if not isinstance(claims, Mapping):
            raise InvalidTokenError("JWT claims must be an object.")
        return claims

    async def _get_signing_jwk(
        self,
        kid: str,
        *,
        force_refresh: bool,
    ) -> dict[str, Any] | None:
        now = time.monotonic()
        if not force_refresh and now < self._cache_expires_at:
            return self._keys.get(kid)

        async with self._refresh_lock:
            now = time.monotonic()
            if not force_refresh and now < self._cache_expires_at:
                return self._keys.get(kid)
            await self._refresh_jwks()
            return self._keys.get(kid)

    async def _refresh_jwks(self) -> None:
        try:
            response = await self._client.get(self._jwks_url)
            response.raise_for_status()
            if len(response.content) > self._max_jwks_bytes:
                raise JWKSUnavailable("JWKS response exceeds the configured size limit.")
            payload = response.json()
        except JWKSUnavailable:
            raise
        except Exception as exc:
            raise JWKSUnavailable("Unable to retrieve trusted JWKS keys.") from exc

        if not isinstance(payload, Mapping) or not isinstance(payload.get("keys"), list):
            raise JWKSUnavailable("JWKS response must contain a keys array.")

        parsed: dict[str, dict[str, Any]] = {}
        for item in payload["keys"]:
            if not isinstance(item, Mapping):
                continue
            kid = item.get("kid")
            if not isinstance(kid, str) or not kid.strip():
                continue
            # Duplicate key IDs are ambiguous and must never be resolved by
            # choosing whichever key happened to appear first.
            if kid in parsed:
                raise JWKSUnavailable("JWKS contains duplicate key IDs.")
            parsed[kid] = dict(item)

        if not parsed:
            raise JWKSUnavailable("JWKS contains no usable key entries.")
        self._keys = parsed
        self._cache_expires_at = time.monotonic() + self._cache_ttl

    @staticmethod
    def _validate_jwk_metadata(jwk: Mapping[str, Any], algorithm: str) -> None:
        if any(field in jwk for field in _PRIVATE_JWK_FIELDS):
            raise InvalidTokenError("JWKS must not expose private or symmetric key material.")
        if jwk.get("use") not in (None, "sig"):
            raise InvalidTokenError("JWKS key is not designated for signatures.")
        key_ops = jwk.get("key_ops")
        if key_ops is not None and (
            not isinstance(key_ops, list) or "verify" not in key_ops
        ):
            raise InvalidTokenError("JWKS key does not permit signature verification.")
        key_algorithm = jwk.get("alg")
        if key_algorithm is not None and key_algorithm != algorithm:
            raise InvalidTokenError("JWKS key algorithm does not match the JWT header.")

        expected_kty = (
            "RSA" if algorithm.startswith(("RS", "PS"))
            else "EC" if algorithm.startswith("ES")
            else "OKP"
        )
        if jwk.get("kty") != expected_kty:
            raise InvalidTokenError("JWKS key type does not match the JWT algorithm.")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()


__all__ = ["AsyncJWKSJWTVerifier", "JWKSConfigurationError", "JWKSUnavailable"]
