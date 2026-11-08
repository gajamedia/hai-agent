from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.security.authentication import AuthenticatedPrincipal
from api.security.authentication_middleware import (
    APIAuthenticationAuthorizationMiddleware,
    required_permission_for_route,
)
from api.security.authorization import AuthorizationPolicy


class StaticAuthenticationProvider:
    def __init__(self, principal: AuthenticatedPrincipal | None) -> None:
        self.principal = principal

    async def authenticate(self, request):
        return self.principal


def _app(*, principal=None, permissions=(), roles=(), auth_enabled=True):
    app = FastAPI()
    app.state.auth_enabled = auth_enabled
    app.state.authentication_provider = (
        StaticAuthenticationProvider(principal)
        if principal is not None or auth_enabled
        else None
    )
    app.state.authorization_policy = AuthorizationPolicy(
        role_permissions={
            "test:execution-reader": ("execution:state:read",),
        }
    )
    app.add_middleware(APIAuthenticationAuthorizationMiddleware)

    @app.get("/api/v1/execution/state")
    def state():
        return {"status": "ok"}

    @app.get("/api/v1/not-yet-mapped")
    def unmapped():
        return {"status": "should-not-be-reached"}

    return app


def test_route_permission_mapping_is_explicit_and_method_specific():
    assert required_permission_for_route(
        "GET", "/api/v1/execution/state"
    ) == "execution:state:read"
    assert required_permission_for_route(
        "POST", "/api/v1/execution/state"
    ) is None
    assert required_permission_for_route(
        "POST", "/api/v1/execution/reconcile"
    ) == "execution:reconciliation:write"


def test_enabled_authentication_rejects_missing_principal():
    client = TestClient(_app(principal=None))
    response = client.get("/api/v1/execution/state")
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "authentication_required"


def test_enabled_authentication_rejects_missing_permission():
    principal = AuthenticatedPrincipal(actor_id="user-1")
    client = TestClient(_app(principal=principal))
    response = client.get("/api/v1/execution/state")
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "permission_denied"


def test_enabled_authentication_accepts_explicit_permission():
    principal = AuthenticatedPrincipal(
        actor_id="user-1",
        permissions=("execution:state:read",),
    )
    client = TestClient(_app(principal=principal))
    response = client.get("/api/v1/execution/state")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_enabled_authentication_accepts_mapped_role():
    principal = AuthenticatedPrincipal(
        actor_id="user-2",
        roles=("test:execution-reader",),
    )
    client = TestClient(_app(principal=principal))
    response = client.get("/api/v1/execution/state")
    assert response.status_code == 200


def test_enabled_authentication_fails_closed_for_unmapped_api_route():
    principal = AuthenticatedPrincipal(
        actor_id="user-1",
        permissions=("api:all:access",),
    )
    client = TestClient(_app(principal=principal))
    response = client.get("/api/v1/not-yet-mapped")
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "authorization_route_unmapped"


def test_local_development_mode_preserves_non_capability_route_access():
    client = TestClient(_app(auth_enabled=False))
    response = client.get("/api/v1/execution/state")
    assert response.status_code == 200


def test_production_startup_requires_authentication(monkeypatch):
    import pytest
    from fastapi.testclient import TestClient
    from api.main import create_app

    monkeypatch.setenv("HAI_ENV", "production")
    monkeypatch.setenv("HAI_AUTH_ENABLED", "false")
    monkeypatch.delenv("HAI_AUTH_REQUIRED", raising=False)

    app = create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: object(),
    )
    with pytest.raises(RuntimeError, match="requires HAI_AUTH_ENABLED=true"):
        with TestClient(app):
            pass
