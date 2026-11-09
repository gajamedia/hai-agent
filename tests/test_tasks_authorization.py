"""The task history through the real authorization middleware and the real role map.

The API tests attach a principal directly and so never meet the middleware; this is the path a
signed-in browser takes (HAI_AUTH_ENABLED=true), which once answered 403 route_unmapped.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.security.authentication import AuthenticatedPrincipal
from api.security.authentication_middleware import (
    APIAuthenticationAuthorizationMiddleware, required_permission_for_route,
)
from api.security.authorization import AuthorizationPolicy
from api.security.authorization_config import _role_map_for_keycloak

CLIENT = "hai-api"


class _Provider:
    def __init__(self, principal):
        self.principal = principal

    async def authenticate(self, request):
        return self.principal


def _client(*roles):
    app = FastAPI()
    app.state.auth_enabled = True
    app.state.authentication_provider = _Provider(AuthenticatedPrincipal(
        actor_id="u1", roles=tuple(f"client:{CLIENT}:{r}" for r in roles)))
    app.state.authorization_policy = AuthorizationPolicy(role_permissions=_role_map_for_keycloak(CLIENT))
    app.add_middleware(APIAuthenticationAuthorizationMiddleware)

    @app.get("/api/v1/tasks")
    def listing():
        return {"tasks": []}

    @app.get("/api/v1/tasks/{task_id}")
    def one(task_id: str):
        return {"task_id": task_id}

    return TestClient(app)


def test_the_task_routes_are_mapped_to_the_history_permission():
    assert required_permission_for_route("GET", "/api/v1/tasks") == "task:history:read"
    assert required_permission_for_route("GET", "/api/v1/tasks/abc") == "task:history:read"
    assert required_permission_for_route("POST", "/api/v1/tasks") is None       # read-only
    assert required_permission_for_route("DELETE", "/api/v1/tasks/abc") is None


@pytest.mark.parametrize("role", ["data-reports-user", "data-writer", "data-schema-creator"])
def test_the_roles_that_run_features_may_open_their_history(role):
    client = _client(role)
    assert client.get("/api/v1/tasks").status_code == 200
    assert client.get("/api/v1/tasks/abc").status_code == 200


@pytest.mark.parametrize("roles", [(), ("data-change-approver",), ("database-connection-manager",),
                                   ("cognitive-operator",)])
def test_other_roles_are_refused_not_unmapped(roles):
    response = _client(*roles).get("/api/v1/tasks")
    assert response.status_code == 403
    assert response.json()["detail"]["error"] != "authorization_route_unmapped"
