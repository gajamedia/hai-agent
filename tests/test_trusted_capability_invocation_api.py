from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import create_app
from api.routes.trusted_capability_invocation import router
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy


class PrincipalAttachMiddleware:
    def __init__(self, app, principal):
        self.app = app
        self.principal = principal

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})[
                "authenticated_principal"
            ] = self.principal

        await self.app(scope, receive, send)


def make_principal(permissions=()):
    return AuthenticatedPrincipal(
        actor_id="c6-test-user",
        roles=(),
        permissions=tuple(permissions),
        issuer="c6-test",
        subject="c6-test-subject",
        is_authenticated=True,
    )


def build_endpoint_app(*, permissions=(), port=None, agent=None):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.authorization_policy = AuthorizationPolicy()
    app.state.trusted_capability_invocation_port = port
    app.state.cognitive_runtime = type(
        "Runtime", (), {"agent": agent}
    )()

    app.add_middleware(
        PrincipalAttachMiddleware,
        principal=make_principal(permissions),
    )

    return app


def test_create_app_registers_canonical_invocation_endpoint():
    app = create_app()
    schema = app.openapi()

    path = "/api/v1/trusted-capability-invocations"

    assert path in schema["paths"]
    assert "post" in schema["paths"][path]


def test_endpoint_rejects_missing_canonical_permission():
    app = build_endpoint_app()

    response = TestClient(app).post(
        "/api/v1/trusted-capability-invocations",
        json={
            "action": "read_business_data",
            "arguments": {},
        },
    )

    assert response.status_code == 403


def test_endpoint_rejects_unexpected_payload_fields():
    app = build_endpoint_app(
        permissions=("capability:canonical:invoke",)
    )

    response = TestClient(app).post(
        "/api/v1/trusted-capability-invocations",
        json={
            "action": "read_business_data",
            "target": "attacker",
        },
    )

    assert response.status_code == 422


def test_endpoint_fails_closed_when_invocation_port_missing():
    app = build_endpoint_app(
        permissions=("capability:canonical:invoke",)
    )

    response = TestClient(app).post(
        "/api/v1/trusted-capability-invocations",
        json={
            "action": "read_business_data",
            "arguments": {},
        },
    )

    assert response.status_code == 503
    assert (
        response.json()["detail"]["error"]
        == "trusted_invocation_unavailable"
    )


def test_endpoint_rejects_unknown_server_mapped_action():
    class Port:
        def supports_action(self, action):
            return False

    app = build_endpoint_app(
        permissions=("capability:canonical:invoke",),
        port=Port(),
    )

    response = TestClient(app).post(
        "/api/v1/trusted-capability-invocations",
        json={
            "action": "unmapped_action",
            "arguments": {},
        },
    )

    assert response.status_code == 404
    assert (
        response.json()["detail"]["error"]
        == "trusted_action_not_found"
    )


def test_endpoint_requires_object_arguments():
    app = build_endpoint_app(
        permissions=("capability:canonical:invoke",)
    )

    response = TestClient(app).post(
        "/api/v1/trusted-capability-invocations",
        json={
            "action": "read_business_data",
            "arguments": [],
        },
    )

    assert response.status_code == 422
