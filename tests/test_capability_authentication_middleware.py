
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.security.authentication import AuthenticatedPrincipal
from api.security.authentication_middleware import CapabilityAuthenticationMiddleware


class Provider:
    def __init__(self, principal):
        self.principal = principal
        self.calls = 0

    async def authenticate(self, request):
        self.calls += 1
        return self.principal


def make_app(provider_marker=True, principal=None):
    app = FastAPI()
    provider = Provider(principal) if provider_marker else None
    app.state.authentication_provider = provider
    app.add_middleware(CapabilityAuthenticationMiddleware)

    @app.get("/api/v1/capabilities")
    async def capabilities():
        return {"ok": True}

    @app.get("/api/v1/capability-proposals")
    async def proposals():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app, provider


def test_middleware_attaches_verified_principal_for_lifecycle_and_proposal_paths():
    principal = AuthenticatedPrincipal(actor_id="verified-user")
    app, provider = make_app(principal=principal)

    with TestClient(app) as client:
        assert client.get("/api/v1/capabilities").status_code == 200
        assert client.get("/api/v1/capability-proposals").status_code == 200

    assert provider.calls == 2


def test_middleware_rejects_missing_provider_for_capability_routes():
    app, _ = make_app(provider_marker=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/capabilities")

    assert response.status_code == 503
    assert (
        response.json()["detail"]["error"]
        == "authentication_provider_unavailable"
    )


def test_middleware_rejects_provider_returning_no_principal():
    app, _ = make_app(principal=None)

    with TestClient(app) as client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "authentication_required"


def test_middleware_discards_forged_identity_state_before_authentication():
    principal = AuthenticatedPrincipal(actor_id="verified-user")
    app, _ = make_app(principal=principal)

    @app.middleware("http")
    async def inject_forged_identity(request, call_next):
        request.state.authenticated_principal = AuthenticatedPrincipal(
            actor_id="forged-user"
        )
        request.state.capability_actor = "forged-user"
        return await call_next(request)

    # This route reads the identity after the middleware boundary.
    @app.get("/api/v1/capabilities/identity")
    async def identity(request: Request):
        return {
            "actor_id": request.state.authenticated_principal.actor_id,
            "legacy_actor": request.state.capability_actor,
        }

    with TestClient(app) as client:
        response = client.get("/api/v1/capabilities/identity")

    assert response.status_code == 200
    assert response.json() == {
        "actor_id": "verified-user",
        "legacy_actor": "verified-user",
    }


def test_health_endpoint_does_not_require_capability_authentication():
    app, provider = make_app(provider_marker=False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert provider is None
