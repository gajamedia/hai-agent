from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread

import pytest

from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from agent.core.action import Action
from agent.llm.authenticated_environment_execution_gateway import (
    AuthenticatedEnvironmentExecutionGateway,
    AuthenticatedEnvironmentExecutionRequest,
    EnvironmentExecutionDenied,
    EnvironmentExecutionOutcomeUnknown,
)
from agent.llm.capability_registry import TrustedCapabilityRegistry
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.world.environment_lifecycle import (
    EnvironmentInstanceStatus,
    EnvironmentLifecycleManager,
)
from agent.world.environment_contracts import (
    EnvironmentActionRequest,
    EnvironmentActionResult,
    EnvironmentDescriptor,
    EnvironmentMode,
    EnvironmentObservation,
)

PERMISSION = "warehouse:inventory:move"
CAPABILITY = "warehouse.move_stock"
INSTANCE = "warehouse-001"


class FakeAdapter:
    def __init__(self, *, supports_idempotency=True, fail_execute=False, fail_after=False):
        self.descriptor = EnvironmentDescriptor(
            environment_type="warehouse",
            domain="inventory",
            mode=EnvironmentMode.SIMULATION,
            supports_observation=True,
            supports_idempotency=supports_idempotency,
        )
        self.revision = "r1"
        self.calls = 0
        self.fail_execute = fail_execute
        self.fail_after = fail_after

    def observe(self, *, instance_id):
        if self.fail_after and self.calls:
            raise OSError("observation unavailable")
        return EnvironmentObservation(
            environment_id="warehouse",
            instance_id=instance_id,
            revision=self.revision,
            state={"stock": 8},
        )

    def execute(self, *, instance_id, request: EnvironmentActionRequest):
        self.calls += 1
        if self.fail_execute:
            raise OSError("device timeout")
        previous = self.revision
        self.revision = "r2"
        return EnvironmentActionResult(
            success=True,
            environment_id="warehouse",
            instance_id=instance_id,
            previous_revision=previous,
            current_revision=self.revision,
            output={"moved": 2},
        )

    def reset(self, *, instance_id):
        self.revision = "r1"
        return self.observe(instance_id=instance_id)


def make_gateway(adapter=None, *, permission=True, risk=0.1, audit_sink=None, include_lifecycle=False):
    adapter = adapter or FakeAdapter()
    lifecycle = EnvironmentLifecycleManager()
    lifecycle.register_provider("warehouse", adapter.descriptor, lambda: adapter)
    lifecycle.create_instance("warehouse", INSTANCE)
    registry = TrustedCapabilityRegistry()
    action = Action(
        name="move_stock",
        description="Move stock between trusted warehouse locations",
        handler=lambda **_: {"success": True},
    )
    registry.register_trusted(
        action=action,
        capability_id=CAPABILITY,
        risk_level="low",
        permissions=(PERMISSION,) if permission else (),
    )
    gateway = AuthenticatedEnvironmentExecutionGateway(
        environment_lifecycle_manager=lifecycle,
        capability_binding_validator=lambda environment_type, capability_id: (
            environment_type == "warehouse" and capability_id == CAPABILITY
        ),
        capability_registry=registry,
        authorization_policy=AuthorizationPolicy(),
        execution_security_policy=ExecutionSecurityPolicy(
            action_risks={CAPABILITY: risk},
            audit_sink=audit_sink,
        ),
    )
    if include_lifecycle:
        return gateway, registry, adapter, lifecycle
    return gateway, registry, adapter


def principal(*permissions):
    return AuthenticatedPrincipal(actor_id="operator-1", permissions=permissions)


def request(**kwargs):
    return AuthenticatedEnvironmentExecutionRequest(
        instance_id=INSTANCE,
        capability_id=CAPABILITY,
        arguments={"from": "A", "to": "B", "quantity": 2},
        **kwargs,
    )


def test_success_requires_auth_and_returns_fresh_observation_evidence():
    gateway, _, adapter = make_gateway()
    result = gateway.execute(principal(PERMISSION), request(idempotency_key="move-1"))
    assert result.success is True
    assert result.observation_before.revision == "r1"
    assert result.observation_after.revision == "r2"
    assert result.action_result.output["moved"] == 2
    assert adapter.calls == 1


def test_missing_or_unverified_principal_is_denied_before_adapter():
    gateway, _, adapter = make_gateway()
    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(None, request())
    assert adapter.calls == 0


def test_missing_permission_is_denied_before_adapter():
    gateway, _, adapter = make_gateway()
    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(principal(), request())
    assert adapter.calls == 0


def test_capability_without_registry_permission_is_denied():
    gateway, _, adapter = make_gateway(permission=False)
    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(principal(PERMISSION), request())
    assert adapter.calls == 0


@pytest.mark.parametrize("transition", ["disable", "revoke"])
def test_disabled_or_revoked_capability_is_denied(transition):
    gateway, registry, adapter = make_gateway()
    getattr(registry, transition)(CAPABILITY)
    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(principal(PERMISSION), request())
    assert adapter.calls == 0


def test_execution_risk_policy_denial_blocks_adapter():
    gateway, _, adapter = make_gateway(risk=0.99)
    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(principal(PERMISSION), request())
    assert adapter.calls == 0


def test_expected_revision_mismatch_blocks_adapter():
    gateway, _, adapter = make_gateway()
    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(principal(PERMISSION), request(expected_revision="stale"))
    assert adapter.calls == 0


def test_idempotency_key_requires_provider_support():
    gateway, _, adapter = make_gateway(FakeAdapter(supports_idempotency=False))
    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(principal(PERMISSION), request(idempotency_key="move-1"))
    assert adapter.calls == 0


def test_adapter_exception_is_unknown_and_never_reported_as_failure_or_success():
    gateway, _, adapter = make_gateway(FakeAdapter(fail_execute=True))
    with pytest.raises(EnvironmentExecutionOutcomeUnknown, match="Do not retry blindly"):
        gateway.execute(principal(PERMISSION), request())
    assert adapter.calls == 1


def test_post_action_observation_failure_is_unknown():
    gateway, _, adapter = make_gateway(FakeAdapter(fail_after=True))
    with pytest.raises(EnvironmentExecutionOutcomeUnknown, match="cannot be verified"):
        gateway.execute(principal(PERMISSION), request())
    assert adapter.calls == 1


def test_outcome_audit_failure_is_unknown():
    def broken_audit(_event):
        if _event.event_type == "execution_outcome":
            raise OSError("audit unavailable")

    gateway, _, adapter = make_gateway(audit_sink=broken_audit)
    with pytest.raises(EnvironmentExecutionOutcomeUnknown, match="Do not retry blindly"):
        gateway.execute(principal(PERMISSION), request())
    assert adapter.calls == 1


def test_request_does_not_accept_actor_identity_or_caller_risk():
    fields = AuthenticatedEnvironmentExecutionRequest.__dataclass_fields__
    assert "actor_id" not in fields
    assert "risk" not in fields


def test_gateway_does_not_publish_resolver_or_adapter():
    gateway, _, _ = make_gateway()
    assert not hasattr(gateway, "adapter")
    assert not hasattr(gateway, "adapters")
    assert not hasattr(gateway, "target_resolver")
    assert not hasattr(gateway, "adapter")


def test_capability_not_bound_to_environment_is_denied():
    adapter = FakeAdapter()
    registry = TrustedCapabilityRegistry()
    registry.register_trusted(
        action=Action(name="move_stock", description="Move stock", handler=lambda **_: True),
        capability_id=CAPABILITY,
        risk_level="low",
        permissions=(PERMISSION,),
    )
    lifecycle = EnvironmentLifecycleManager()
    lifecycle.register_provider("warehouse", adapter.descriptor, lambda: adapter)
    lifecycle.create_instance("warehouse", INSTANCE)
    gateway = AuthenticatedEnvironmentExecutionGateway(
        environment_lifecycle_manager=lifecycle,
        capability_binding_validator=lambda _environment_type, _capability_id: False,
        capability_registry=registry,
        authorization_policy=AuthorizationPolicy(),
        execution_security_policy=ExecutionSecurityPolicy(action_risks={CAPABILITY: 0.1}),
    )
    with pytest.raises(EnvironmentExecutionDenied, match="not registered for this environment"):
        gateway.execute(principal(PERMISSION), request())
    assert adapter.calls == 0


class BlockingAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.execute_entered = Event()
        self.release_execute = Event()

    def execute(self, *, instance_id, request):
        self.execute_entered.set()
        if not self.release_execute.wait(timeout=5):
            raise TimeoutError("test adapter release timed out")
        return super().execute(instance_id=instance_id, request=request)


def test_close_waits_for_execution_lease_and_blocks_new_execution():
    adapter = BlockingAdapter()
    gateway, _, _, lifecycle = make_gateway(adapter, include_lifecycle=True)
    execution_errors = []
    close_finished = Event()

    def run_execution():
        try:
            gateway.execute(principal(PERMISSION), request())
        except BaseException as exc:
            execution_errors.append(exc)

    def close_instance():
        lifecycle.close_instance(INSTANCE)
        close_finished.set()

    execution_thread = Thread(target=run_execution)
    execution_thread.start()
    assert adapter.execute_entered.wait(timeout=2)

    close_thread = Thread(target=close_instance)
    close_thread.start()
    for _ in range(200):
        if lifecycle.get_instance(INSTANCE).status == EnvironmentInstanceStatus.CLOSING:
            break
        Event().wait(0.005)
    assert lifecycle.get_instance(INSTANCE).status == EnvironmentInstanceStatus.CLOSING
    assert not close_finished.is_set()

    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(principal(PERMISSION), request())

    adapter.release_execute.set()
    execution_thread.join(timeout=3)
    close_thread.join(timeout=3)
    assert not execution_thread.is_alive()
    assert not close_thread.is_alive()
    assert execution_errors == []
    assert close_finished.is_set()
    assert lifecycle.get_instance(INSTANCE).status == EnvironmentInstanceStatus.CLOSED


def test_closed_instance_is_denied_before_adapter():
    gateway, _, adapter = make_gateway()
    gateway._environment_lifecycle_manager.close_instance(INSTANCE)
    with pytest.raises(EnvironmentExecutionDenied):
        gateway.execute(principal(PERMISSION), request())
    assert adapter.calls == 0


def test_environment_lease_is_released_after_adapter_exception():
    gateway, _, adapter, lifecycle = make_gateway(
        FakeAdapter(fail_execute=True), include_lifecycle=True
    )
    with pytest.raises(EnvironmentExecutionOutcomeUnknown):
        gateway.execute(principal(PERMISSION), request())
    closed = lifecycle.close_instance(INSTANCE)
    assert closed.status == EnvironmentInstanceStatus.CLOSED
    assert adapter.calls == 1
