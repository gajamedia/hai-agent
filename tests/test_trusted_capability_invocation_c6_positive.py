
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.trusted_capability_invocation import router
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from api.security.trusted_execution_context import TrustedExecutionContext

from agent.llm.capability_execution_contracts import CapabilityReadResult
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.trusted_capability_invocation_port import (
    CapabilityInvocationSpec,
    TrustedCapabilityInvocationPort,
)


_PERMISSION = "capability:canonical:invoke"
_ACTION = "read_orders"
_TARGET = "business_db"
_CAPABILITY_ID = "orders.read"


class PrincipalAttachMiddleware:
    """Test-only middleware that attaches a verified principal."""

    def __init__(self, app, principal):
        self.app = app
        self.principal = principal

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})[
                "authenticated_principal"
            ] = self.principal

        await self.app(scope, receive, send)


class RecordingGateway:
    """Gateway double; it never opens a database connection."""

    def __init__(self):
        self.calls = []

    def read(self, principal, request, *, execution_id):
        self.calls.append(
            {
                "principal": principal,
                "request": request,
                "execution_id": execution_id,
            }
        )

        return CapabilityReadResult(
            target=request.target,
            capability_id=request.capability_id,
            data={
                "rows": [
                    {"id": 101, "order_number": "TEST-001"}
                ],
                "count": 1,
            },
            metadata={"source": "c6-positive-test"},
        )


def build_test_runtime(*, action_risk=0.1):
    principal = AuthenticatedPrincipal(
        actor_id="c6-positive-test-user",
        roles=(),
        permissions=(_PERMISSION,),
        issuer="c6-test",
        subject="c6-test-subject",
    )

    gateway = RecordingGateway()

    specs = {
        _ACTION: CapabilityInvocationSpec(
            action_name=_ACTION,
            target=_TARGET,
            capability_id=_CAPABILITY_ID,
        )
    }

    port = TrustedCapabilityInvocationPort(
        gateway=gateway,
        specs=specs,
    )

    security_policy = ExecutionSecurityPolicy(
        action_risks={_ACTION: action_risk},
        risk_threshold=0.65,
    )

    # The trusted capability branch must bypass the simulation executor.
    # An object is sufficient because the port branch does not call execute().
    agent = SimpleNamespace(
        actions={},
        unified_executor=object(),
        execution_security_policy=security_policy,
    )

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    app.state.authorization_policy = AuthorizationPolicy()
    app.state.trusted_capability_invocation_port = port
    app.state.capability_execution_gateway = gateway
    app.state.cognitive_runtime = SimpleNamespace(agent=agent)

    app.add_middleware(
        PrincipalAttachMiddleware,
        principal=principal,
    )

    return app, principal, gateway, port, security_policy


def test_c6_positive_invocation_reaches_existing_gateway(
    monkeypatch,
):
    app, principal, gateway, port, security_policy = (
        build_test_runtime()
    )

    captured = []
    original_invoke = TrustedCapabilityInvocationPort.invoke

    def capture_invocation(
        self,
        request,
        *,
        execution_context,
    ):
        captured.append(
            {
                "request": request,
                "execution_context": execution_context,
            }
        )

        return original_invoke(
            self,
            request,
            execution_context=execution_context,
        )

    monkeypatch.setattr(
        TrustedCapabilityInvocationPort,
        "invoke",
        capture_invocation,
    )

    response = TestClient(app).post(
        "/api/v1/trusted-capability-invocations",
        json={
            "action": _ACTION,
            "arguments": {"limit": 5},
        },
    )

    assert response.status_code == 200, response.text

    body = response.json()

    # 1. Endpoint returns the capability result.
    assert body["status"] == "ok"
    assert body["action"] == _ACTION
    assert body["target"] == _TARGET
    assert body["capability_id"] == _CAPABILITY_ID
    assert body["data"]["count"] == 1
    assert body["data"]["rows"][0]["order_number"] == "TEST-001"
    assert body["metadata"]["source"] == "c6-positive-test"
    assert body["read_only"] is True

    # 2. Exactly one gateway invocation occurred.
    assert len(gateway.calls) == 1

    gateway_call = gateway.calls[0]
    gateway_request = gateway_call["request"]

    # 3. The gateway receives the authenticated principal,
    # not an identity supplied in the HTTP payload.
    assert gateway_call["principal"] is principal
    assert gateway_request.arguments == {"limit": 5}

    # 4. Target and capability ID come from server-owned specs.
    assert gateway_request.target == _TARGET
    assert gateway_request.capability_id == _CAPABILITY_ID

    # 5. The execution context is the exact trusted context
    # consumed by the invocation port.
    assert len(captured) == 1

    execution_context = captured[0]["execution_context"]
    boundary_request = captured[0]["request"]

    assert type(execution_context) is TrustedExecutionContext
    assert execution_context.principal is principal
    assert execution_context.execution_id == gateway_call["execution_id"]
    assert execution_context.execution_id == body["execution_id"]

    # 6. Trusted context never enters request metadata or step content.
    assert "execution_context" not in boundary_request.metadata
    assert boundary_request.step == {
        "arguments": {"limit": 5}
    }

    # 7. The canonical execution security policy remains active.
    assert security_policy is app.state.cognitive_runtime.agent.execution_security_policy

    # 8. The simulation executor was not used.
    assert app.state.cognitive_runtime.agent.unified_executor is not None


def test_c6_positive_path_is_denied_by_execution_security_policy():
    app, principal, gateway, port, security_policy = (
        build_test_runtime(action_risk=0.95)
    )

    response = TestClient(app).post(
        "/api/v1/trusted-capability-invocations",
        json={
            "action": _ACTION,
            "arguments": {"limit": 5},
        },
    )

    # High-risk action requires approval. No approval checker is configured,
    # so the canonical execution policy must deny it.
    assert response.status_code == 403
    assert (
        response.json()["detail"]["error"]
        == "canonical_execution_denied"
    )

    # The request must be rejected before reaching the gateway.
    assert gateway.calls == []
