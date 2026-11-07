from __future__ import annotations

import asyncio
import base64
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from api.security.jwks_jwt_verifier import (
    AsyncJWKSJWTVerifier,
    JWKSConfigurationError,
    JWKSUnavailable,
)


def _b64uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _make_keys(kid: str = "key-1"):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": _b64uint(public_numbers.n),
        "e": _b64uint(public_numbers.e),
    }
    return private, jwk


def _token(private, *, kid="key-1", audience="hai-api", issuer="https://issuer.test", **extra):
    now = int(time.time())
    claims = {
        "sub": "user-123",
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + 300,
        **extra,
    }
    return jwt.encode(claims, private, algorithm="RS256", headers={"kid": kid})


def _verifier(jwks, *, transport=None, **kwargs):
    async_client = httpx.AsyncClient(
        transport=transport or httpx.MockTransport(
            lambda request: httpx.Response(200, json={"keys": jwks})
        )
    )
    verifier = AsyncJWKSJWTVerifier(
        jwks_url="https://issuer.test/.well-known/jwks.json",
        issuer="https://issuer.test",
        audience="hai-api",
        client=async_client,
        **kwargs,
    )
    return verifier, async_client


def test_verifies_signature_and_required_claims():
    private, jwk = _make_keys()
    verifier, client = _verifier([jwk])
    try:
        claims = asyncio.run(verifier.verify(_token(private)))
        assert claims["sub"] == "user-123"
        assert claims["aud"] == "hai-api"
    finally:
        asyncio.run(client.aclose())


def test_rejects_wrong_audience():
    private, jwk = _make_keys()
    verifier, client = _verifier([jwk])
    try:
        with pytest.raises(jwt.InvalidAudienceError):
            asyncio.run(verifier.verify(_token(private, audience="other-api")))
    finally:
        asyncio.run(client.aclose())


def test_rejects_wrong_issuer():
    private, jwk = _make_keys()
    verifier, client = _verifier([jwk])
    try:
        with pytest.raises(jwt.InvalidIssuerError):
            asyncio.run(verifier.verify(_token(private, issuer="https://attacker.test")))
    finally:
        asyncio.run(client.aclose())


def test_rejects_expired_token():
    private, jwk = _make_keys()
    verifier, client = _verifier([jwk])
    try:
        token = _token(private, exp=int(time.time()) - 30)
        with pytest.raises(jwt.ExpiredSignatureError):
            asyncio.run(verifier.verify(token))
    finally:
        asyncio.run(client.aclose())


def test_rejects_wrong_signature():
    trusted_private, trusted_jwk = _make_keys("trusted")
    attacker_private, _ = _make_keys("attacker")
    verifier, client = _verifier([trusted_jwk])
    try:
        token = _token(attacker_private, kid="trusted")
        with pytest.raises(jwt.InvalidSignatureError):
            asyncio.run(verifier.verify(token))
    finally:
        asyncio.run(client.aclose())


def test_unknown_kid_triggers_refresh_and_finds_rotated_key():
    private, jwk = _make_keys("rotated-key")
    calls = {"count": 0}

    _, old_jwk = _make_keys("old-key")
    def handler(request):
        calls["count"] += 1
        return httpx.Response(200, json={"keys": [old_jwk] if calls["count"] == 1 else [jwk]})

    transport = httpx.MockTransport(handler)
    verifier, client = _verifier([old_jwk], transport=transport)
    try:
        claims = asyncio.run(verifier.verify(_token(private, kid="rotated-key")))
        assert claims["sub"] == "user-123"
        assert calls["count"] == 2
    finally:
        asyncio.run(client.aclose())


def test_rejects_algorithm_not_in_allowlist_before_network():
    private, jwk = _make_keys()
    verifier, client = _verifier([jwk], algorithms=("ES256",))
    try:
        with pytest.raises(jwt.InvalidTokenError, match="not allowed"):
            asyncio.run(verifier.verify(_token(private)))
    finally:
        asyncio.run(client.aclose())


def test_rejects_missing_kid():
    private, _ = _make_keys()
    verifier, client = _verifier([])
    try:
        token = jwt.encode(
            {
                "sub": "user-123",
                "iss": "https://issuer.test",
                "aud": "hai-api",
                "iat": int(time.time()),
                "exp": int(time.time()) + 300,
            },
            private,
            algorithm="RS256",
        )
        with pytest.raises(jwt.InvalidTokenError, match="key ID"):
            asyncio.run(verifier.verify(token))
    finally:
        asyncio.run(client.aclose())


def test_rejects_private_key_material_in_jwks():
    private, jwk = _make_keys()
    jwk["d"] = "must-not-be-published"
    verifier, client = _verifier([jwk])
    try:
        with pytest.raises(jwt.InvalidTokenError, match="private"):
            asyncio.run(verifier.verify(_token(private)))
    finally:
        asyncio.run(client.aclose())


def test_rejects_duplicate_kid_in_jwks():
    private, jwk = _make_keys()
    verifier, client = _verifier([jwk, dict(jwk)])
    try:
        with pytest.raises(JWKSUnavailable, match="duplicate"):
            asyncio.run(verifier.verify(_token(private)))
    finally:
        asyncio.run(client.aclose())


def test_rejects_non_https_jwks_url_by_default():
    with pytest.raises(JWKSConfigurationError):
        AsyncJWKSJWTVerifier(
            jwks_url="http://issuer.test/jwks",
            issuer="https://issuer.test",
            audience="hai-api",
            client=httpx.AsyncClient(),
        )


def test_rejects_missing_required_configuration():
    with pytest.raises(JWKSConfigurationError):
        AsyncJWKSJWTVerifier(
            jwks_url="https://issuer.test/jwks",
            issuer="",
            audience="hai-api",
        )


def test_jwks_http_failure_is_not_treated_as_empty_identity():
    private, _ = _make_keys()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, json={"error": "unavailable"})
    )
    verifier, client = _verifier([], transport=transport)
    try:
        with pytest.raises(JWKSUnavailable):
            asyncio.run(verifier.verify(_token(private)))
    finally:
        asyncio.run(client.aclose())
