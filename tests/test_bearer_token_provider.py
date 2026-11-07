from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.security.authentication import (
    AuthenticatedPrincipal,
    AuthenticationRejected,
    InvalidAuthenticatedPrincipal,
)
from api.security.bearer_token_provider import BearerTokenAuthenticationProvider


class FakeVerifiedTokenVerifier:
    """Test double representing a verifier that has already verified a token."""

    def __init__(self, claims=None, error=None):
        self.claims = claims
        self.error = error
        self.seen_tokens = []

    def verify(self, token):
        self.seen_tokens.append(token)
        if self.error:
            raise self.error
        return self.claims


def make_request(headers=None):
    app = FastAPI()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [
            (key.lower().encode(), value.encode())
            for key, value in (headers or {}).items()
        ],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
    }
    return Request(scope, receive=lambda: None)


@pytest.mark.parametrize(
    "header",
    [
        None,
        "Basic abc",
        "Bearer",
        "Bearer one two",
        "Bearer ",
    ],
)
def test_missing_or_malformed_bearer_is_rejected_or_anonymous(header):
    verifier = FakeVerifiedTokenVerifier(
        claims={"sub": "user-1", "iss": "https://issuer.example"}
    )
    provider = BearerTokenAuthenticationProvider(verifier)
    headers = {} if header is None else {"Authorization": header}
    if header is None:
        principal = asyncio.run(provider.authenticate(make_request(headers)))
        assert principal is None
        assert verifier.seen_tokens == []
    else:
        with pytest.raises(AuthenticationRejected):
            asyncio.run(provider.authenticate(make_request(headers)))


def test_maps_only_verified_claims_to_principal():
    verifier = FakeVerifiedTokenVerifier(
        claims={
            "sub": "subject-1",
            "iss": "https://issuer.example",
            "roles": ["operator", "operator"],
            "permissions": ["capability:proposal:read"],
            "untrusted_extra": "ignored",
        }
    )
    provider = BearerTokenAuthenticationProvider(verifier)
    principal = asyncio.run(provider.authenticate(
        make_request({"Authorization": "Bearer signed-token"})
    ))
    assert principal == AuthenticatedPrincipal(
        actor_id="subject-1",
        subject="subject-1",
        issuer="https://issuer.example",
        roles=("operator",),
        permissions=("capability:proposal:read",),
    )
    assert verifier.seen_tokens == ["signed-token"]


def test_custom_actor_claim_is_supported_without_changing_subject():
    verifier = FakeVerifiedTokenVerifier(
        claims={
            "sub": "external-subject",
            "uid": "internal-actor",
            "iss": "https://issuer.example",
        }
    )
    provider = BearerTokenAuthenticationProvider(
        verifier,
        actor_id_claim="uid",
    )
    principal = asyncio.run(provider.authenticate(
        make_request({"Authorization": "Bearer token"})
    ))
    assert principal.actor_id == "internal-actor"
    assert principal.subject == "external-subject"


def test_verifier_failure_is_normalized_without_token_leak():
    verifier = FakeVerifiedTokenVerifier(error=RuntimeError("signature detail"))
    provider = BearerTokenAuthenticationProvider(verifier)
    with pytest.raises(AuthenticationRejected) as error:
        asyncio.run(provider.authenticate(
            make_request({"Authorization": "Bearer secret-token"})
        ))
    assert "secret-token" not in str(error.value)
    assert "signature detail" not in str(error.value)


@pytest.mark.parametrize(
    "claims",
    [
        [],
        {},
        {"sub": ""},
        {"sub": "user", "roles": "admin"},
        {"sub": "user", "permissions": "capability:proposal:read"},
        {"sub": "user", "roles": [1]},
    ],
)
def test_invalid_verified_claim_shapes_fail_closed(claims):
    provider = BearerTokenAuthenticationProvider(
        FakeVerifiedTokenVerifier(claims=claims)
    )
    with pytest.raises(InvalidAuthenticatedPrincipal):
        asyncio.run(provider.authenticate(
            make_request({"Authorization": "Bearer token"})
        ))


def test_provider_is_compatible_with_existing_authentication_contract():
    verifier = FakeVerifiedTokenVerifier(
        claims={"sub": "user-1", "iss": "https://issuer.example"}
    )
    provider = BearerTokenAuthenticationProvider(verifier)
    assert callable(provider.authenticate)


@pytest.mark.parametrize("claim,expected", [(1_700_000_000, 1_700_000_000), (None, None), ("1700000000", None),
                                            (True, None), (-5, None), (0, None), (1.5, None)])
def test_auth_time_is_used_only_when_it_is_a_positive_integer(claim, expected):
    claims = {"sub": "u1", "iss": "https://issuer.example"}
    if claim is not None:
        claims["auth_time"] = claim
    provider = BearerTokenAuthenticationProvider(FakeVerifiedTokenVerifier(claims=claims))
    principal = asyncio.run(provider.authenticate(make_request({"Authorization": "Bearer t"})))
    assert principal.auth_time == expected
