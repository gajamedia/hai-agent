
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import api.security.authentication_middleware as authentication_middleware_module

from api.security.authentication import (
    AuthenticationProviderUnavailable,
    AuthenticatedPrincipal,
)
from api.security.authentication_middleware import (
    CapabilityAuthenticationMiddleware,
)


class StaticAuthenticationProvider:
    def __init__(self, principal):
        self.principal = principal
        self.calls = 0

    async def authenticate(self, request):
        self.calls += 1
        return self.principal


class UnavailableAuthenticationProvider:
    async def authenticate(self, request):
        raise AuthenticationProviderUnavailable(
            "Authentication provider unavailable."
        )


class FailingAuthenticationProvider:
    async def authenticate(self, request):
        raise RuntimeError("Simulated provider failure.")


class SeedIdentityMiddleware:
    """Simulate untrusted upstream middleware injecting identity state."""

    def __init__(
        self,
        app,
        *,
        actor="forged-actor",
        principal=None,
    ):
        self.app = app
        self.actor = actor
        self.principal = principal

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            state = scope.setdefault("state", {})

            if self.actor is not None:
                state["capability_actor"] = self.actor

            if self.principal is not None:
                state["authenticated_principal"] = self.principal

        await self.app(scope, receive, send)


def make_client(
    *,
    provider="valid",
    seed_identity=False,
):
    app = FastAPI()

    @app.get("/api/v1/capability-proposals")
    async def proposals(request: Request):
        principal = getattr(
            request.state,
            "authenticated_principal",
            None,
        )

        actor = getattr(
            request.state,
            "capability_actor",
            None,
        )

        return {
            "status": "ok",
            "actor": actor,
            "principal_actor": (
                principal.actor_id if principal is not None else None
            ),
        }

    @app.get("/api/v1/capability-proposals/{proposal_id}")
    async def proposal_detail(proposal_id: str):
        return {"proposal_id": proposal_id}

    @app.get("/api/v1/capabilities")
    async def capabilities(request: Request):
        principal = getattr(
            request.state,
            "authenticated_principal",
            None,
        )

        return {
            "actor": (
                principal.actor_id if principal is not None else None
            )
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.options("/api/v1/capability-proposals")
    async def proposal_options():
        return {"status": "options"}

    if provider == "valid":
        principal = AuthenticatedPrincipal(
            actor_id="verified-user",
            roles=("client:hai-agent-api:proposal-reader",),
        )
        authentication_provider = StaticAuthenticationProvider(principal)

    elif provider == "reject":
        authentication_provider = StaticAuthenticationProvider(None)

    elif provider == "unavailable":
        authentication_provider = UnavailableAuthenticationProvider()

    elif provider == "failure":
        authentication_provider = FailingAuthenticationProvider()

    elif provider == "missing":
        authentication_provider = None

    else:
        raise ValueError(f"Unknown provider mode: {provider}")

    app.state.authentication_provider = authentication_provider

    # Starlette menempatkan middleware yang terakhir didaftarkan
    # sebagai middleware terluar.
    #
    # Karena itu SeedIdentityMiddleware harus didaftarkan terakhir:
    # 1. SeedIdentityMiddleware menanam identitas palsu.
    # 2. CapabilityAuthenticationMiddleware membersihkan identitas lama.
    # 3. Principal terverifikasi dipasang kembali oleh middleware auth.
    app.add_middleware(CapabilityAuthenticationMiddleware)

    if seed_identity:
        app.add_middleware(
            SeedIdentityMiddleware,
            actor="forged-actor",
            principal=AuthenticatedPrincipal(
                actor_id="forged-principal",
            ),
        )

    return TestClient(app), authentication_provider


def test_valid_provider_authenticates_capability_proposal_endpoint():
    client, provider = make_client()

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 200
    assert response.json()["actor"] == "verified-user"
    assert response.json()["principal_actor"] == "verified-user"
    assert provider.calls == 1


def test_valid_provider_authenticates_capability_lifecycle_endpoint():
    client, provider = make_client()

    with client:
        response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    assert response.json()["actor"] == "verified-user"
    assert provider.calls == 1


def test_middleware_clears_forged_principal_and_actor():
    client, _ = make_client(seed_identity=True)

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 200
    assert response.json()["actor"] == "verified-user"
    assert response.json()["principal_actor"] == "verified-user"


def test_provider_rejection_returns_401():
    client, _ = make_client(provider="reject")

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 401
    assert (
        response.json()["detail"]["error"]
        == "authentication_required"
    )


def test_missing_provider_returns_503():
    client, _ = make_client(provider="missing")

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 503
    assert (
        response.json()["detail"]["error"]
        == "authentication_provider_unavailable"
    )


def test_unavailable_provider_returns_503():
    client, _ = make_client(provider="unavailable")

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 503
    assert (
        response.json()["detail"]["error"]
        == "authentication_provider_unavailable"
    )


def test_provider_runtime_failure_is_normalized_to_401():
    """Unexpected provider errors are normalized by authenticate_request."""

    client, _ = make_client(provider="failure")

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 401
    assert (
        response.json()["detail"]["error"]
        == "authentication_required"
    )


def test_unexpected_middleware_failure_returns_500(monkeypatch):
    """An unexpected error outside authenticate_request returns HTTP 500."""

    client, _ = make_client(provider="valid")

    async def raise_unexpected_error(request):
        raise RuntimeError("Simulated unexpected middleware failure.")

    monkeypatch.setattr(
        authentication_middleware_module,
        "authenticate_request",
        raise_unexpected_error,
    )

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 500
    assert (
        response.json()["detail"]["error"]
        == "authentication_internal_error"
    )


def test_non_capability_endpoint_bypasses_authentication():
    client, provider = make_client(provider="missing")

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    assert provider is None


def test_options_request_bypasses_authentication():
    client, provider = make_client(provider="missing")

    with client:
        response = client.options("/api/v1/capability-proposals")

    assert response.status_code == 200
    assert response.json() == {"status": "options"}
    assert provider is None


def test_similar_but_unprotected_path_bypasses_authentication():
    client, provider = make_client(provider="missing")

    with client:
        response = client.get(
            "/api/v1/capability-proposals-extra"
        )

    assert response.status_code == 404
    assert provider is None


def test_proposal_detail_subpath_is_protected():
    client, _ = make_client(provider="reject")

    with client:
        response = client.get(
            "/api/v1/capability-proposals/proposal-123"
        )

    assert response.status_code == 401


def test_capability_lifecycle_subpath_is_protected():
    client, _ = make_client(provider="reject")

    with client:
        response = client.get(
            "/api/v1/capabilities/application.inspect_room"
        )

    assert response.status_code == 401
