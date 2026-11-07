from __future__ import annotations

import pytest

from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from agent.core.action import Action
from agent.llm.authenticated_capability_execution_gateway import (
    AuthenticatedCapabilityExecutionGateway,
    AuthenticatedCapabilityReadRequest,
)
from agent.llm.capability_execution_contracts import CapabilityAuthorizationError
from agent.llm.capability_registry import TrustedCapabilityRegistry
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.room_environment_adapter import RoomEnvironmentAdapter


PERMISSION = "room-environment:state:read"
CAPABILITY = RoomEnvironmentAdapter.OBSERVE_CAPABILITY


class FakeEnvironment:
    def __init__(self):
        self.calls = 0

    def observe(self):
        self.calls += 1
        return {"box": "closed"}


def make_registry(*, permissions=(PERMISSION,), capability_id=CAPABILITY):
    registry = TrustedCapabilityRegistry()
    action = Action(
        name="read_orders",
        description="Read allowlisted business orders",
        handler=lambda **_: {"success": True},
    )
    registry.register_trusted(
        action=action,
        capability_id=capability_id,
        risk_level="low",
        permissions=permissions,
    )
    return registry


def make_gateway(environment, *, permissions=(PERMISSION,), auth_policy=None):
    registry = make_registry(permissions=permissions)
    adapter = RoomEnvironmentAdapter(environment)
    gateway = AuthenticatedCapabilityExecutionGateway(
        [adapter],
        capability_registry=registry,
        authorization_policy=auth_policy or AuthorizationPolicy(),
        execution_security_policy=ExecutionSecurityPolicy(
            action_risks={CAPABILITY: 0.1},
        ),
    )
    return gateway, registry


def request():
    return AuthenticatedCapabilityReadRequest(
        target="room_environment",
        capability_id=CAPABILITY,
    )


def test_read_requires_verified_principal_and_does_not_accept_actor_id_from_request():
    environment = FakeEnvironment()
    gateway, _ = make_gateway(environment)

    with pytest.raises(CapabilityAuthorizationError):
        gateway.read(None, request())

    assert environment.calls == 0
    assert "actor_id" not in AuthenticatedCapabilityReadRequest.__dataclass_fields__


def test_explicit_permission_claim_allows_read():
    environment = FakeEnvironment()
    gateway, _ = make_gateway(environment)
    principal = AuthenticatedPrincipal(
        actor_id="operator-17",
        permissions=(PERMISSION,),
    )

    result = gateway.read(principal, request())

    assert result.data == {"box": "closed"}
    assert result.metadata["read_only"] is True
    assert environment.calls == 1


def test_role_permission_mapping_allows_read():
    environment = FakeEnvironment()
    gateway, _ = make_gateway(
        environment,
        auth_policy=AuthorizationPolicy(
            role_permissions={"client:hai:business-reader": (PERMISSION,)}
        ),
    )
    principal = AuthenticatedPrincipal(
        actor_id="operator-18",
        roles=("client:hai:business-reader",),
    )

    assert gateway.read(principal, request()).data == {"box": "closed"}


def test_missing_permission_denies_before_adapter_access():
    environment = FakeEnvironment()
    gateway, _ = make_gateway(environment)
    principal = AuthenticatedPrincipal(actor_id="operator-no-permission")

    with pytest.raises(CapabilityAuthorizationError):
        gateway.read(principal, request())

    assert environment.calls == 0


def test_explicit_policy_deny_overrides_permission_claim():
    environment = FakeEnvironment()
    gateway, _ = make_gateway(
        environment,
        auth_policy=AuthorizationPolicy(denied_permissions=frozenset({PERMISSION})),
    )
    principal = AuthenticatedPrincipal(
        actor_id="operator-denied",
        permissions=(PERMISSION,),
    )

    with pytest.raises(CapabilityAuthorizationError):
        gateway.read(principal, request())

    assert environment.calls == 0


def test_disabled_or_revoked_capability_is_denied():
    for transition in ("disable", "revoke"):
        environment = FakeEnvironment()
        gateway, registry = make_gateway(environment)
        if transition == "disable":
            registry.disable(CAPABILITY)
        else:
            registry.revoke(CAPABILITY)

        principal = AuthenticatedPrincipal(
            actor_id="operator-17",
            permissions=(PERMISSION,),
        )
        with pytest.raises(CapabilityAuthorizationError):
            gateway.read(principal, request())
        assert environment.calls == 0


def test_missing_registry_permission_denies_even_if_principal_has_claim():
    environment = FakeEnvironment()
    gateway, _ = make_gateway(environment, permissions=())
    principal = AuthenticatedPrincipal(
        actor_id="operator-17",
        permissions=(PERMISSION,),
    )

    with pytest.raises(CapabilityAuthorizationError):
        gateway.read(principal, request())
    assert environment.calls == 0


def test_execution_security_policy_denial_blocks_read():
    environment = FakeEnvironment()
    registry = make_registry()
    gateway = AuthenticatedCapabilityExecutionGateway(
        [RoomEnvironmentAdapter(environment)],
        capability_registry=registry,
        authorization_policy=AuthorizationPolicy(),
        execution_security_policy=ExecutionSecurityPolicy(
            action_risks={CAPABILITY: 0.95},
            risk_threshold=0.65,
        ),
    )
    principal = AuthenticatedPrincipal(
        actor_id="operator-17",
        permissions=(PERMISSION,),
    )

    with pytest.raises(CapabilityAuthorizationError):
        gateway.read(principal, request())
    assert environment.calls == 0


def test_audit_failure_in_execution_security_policy_fails_closed():
    environment = FakeEnvironment()

    def broken_audit(_event):
        raise OSError("audit unavailable")

    registry = make_registry()
    gateway = AuthenticatedCapabilityExecutionGateway(
        [RoomEnvironmentAdapter(environment)],
        capability_registry=registry,
        authorization_policy=AuthorizationPolicy(),
        execution_security_policy=ExecutionSecurityPolicy(
            action_risks={CAPABILITY: 0.1},
            audit_sink=broken_audit,
        ),
    )
    principal = AuthenticatedPrincipal(
        actor_id="operator-17",
        permissions=(PERMISSION,),
    )

    with pytest.raises(CapabilityAuthorizationError):
        gateway.read(principal, request())
    assert environment.calls == 0


def test_gateway_does_not_publish_raw_gateway_or_adapters():
    environment = FakeEnvironment()
    gateway, _ = make_gateway(environment)

    assert not hasattr(gateway, "adapters")
    assert not hasattr(gateway, "adapter")
    assert not hasattr(gateway, "execute")
    assert not hasattr(gateway, "write")
