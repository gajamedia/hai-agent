from __future__ import annotations

from threading import Event, Thread

import pytest

from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from agent.core.action import Action
from agent.llm.authenticated_capability_execution_gateway import (
    AuthenticatedCapabilityExecutionGateway,
    AuthenticatedCapabilityReadRequest,
)
from agent.llm.capability_execution_contracts import CapabilityAuthorizationError
from agent.llm.capability_registry import (
    CapabilityNotFoundError,
    CapabilityStatus,
    TrustedCapabilityRegistry,
)
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.room_environment_adapter import RoomEnvironmentAdapter


PERMISSION = "room-environment:state:read"
CAPABILITY = RoomEnvironmentAdapter.OBSERVE_CAPABILITY


class BlockingEnvironment:
    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.calls = 0

    def observe(self):
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the blocked read")
        return {"box": "closed"}


def make_registry(capability_id=CAPABILITY, action_name="read_orders"):
    registry = TrustedCapabilityRegistry()
    action = Action(
        name=action_name,
        description="Read allowlisted business orders",
        handler=lambda **_: {"success": True},
    )
    registry.register_trusted(
        action=action,
        capability_id=capability_id,
        risk_level="low",
        permissions=(PERMISSION,),
    )
    return registry


def make_gateway(environment, registry):
    return AuthenticatedCapabilityExecutionGateway(
        [RoomEnvironmentAdapter(environment)],
        capability_registry=registry,
        authorization_policy=AuthorizationPolicy(),
        execution_security_policy=ExecutionSecurityPolicy(
            action_risks={CAPABILITY: 0.1},
        ),
    )


def make_request():
    return AuthenticatedCapabilityReadRequest(
        target="room_environment",
        capability_id=CAPABILITY,
    )


def make_principal():
    return AuthenticatedPrincipal(
        actor_id="operator-17",
        permissions=(PERMISSION,),
    )


@pytest.mark.parametrize("transition", ["disable", "revoke"])
def test_lifecycle_transition_waits_for_inflight_authenticated_read(transition):
    environment = BlockingEnvironment()
    registry = make_registry()
    gateway = make_gateway(environment, registry)
    read_results = []
    read_errors = []
    transition_finished = Event()
    transition_started = Event()

    def run_read():
        try:
            read_results.append(gateway.read(make_principal(), make_request()))
        except BaseException as exc:  # surfaced in the test thread
            read_errors.append(exc)

    def run_transition():
        transition_started.set()
        getattr(registry, transition)(CAPABILITY)
        transition_finished.set()

    reader = Thread(target=run_read, daemon=True)
    reader.start()
    assert environment.started.wait(timeout=2), "adapter read did not start"

    lifecycle = Thread(target=run_transition, daemon=True)
    lifecycle.start()
    assert transition_started.wait(timeout=2)
    # The adapter is still blocked, so the lifecycle transition must not finish.
    assert not transition_finished.wait(timeout=0.15)

    environment.release.set()
    reader.join(timeout=2)
    lifecycle.join(timeout=2)

    assert not reader.is_alive()
    assert not lifecycle.is_alive()
    assert read_errors == []
    assert len(read_results) == 1
    assert transition_finished.is_set()
    expected = (
        CapabilityStatus.DISABLED
        if transition == "disable"
        else CapabilityStatus.REVOKED
    )
    assert registry.get(CAPABILITY, include_inactive=True).status == expected

    # Once disable/revoke has returned, a new read cannot reach the adapter.
    with pytest.raises(CapabilityAuthorizationError):
        gateway.read(make_principal(), make_request())
    assert environment.calls == 1


def test_execution_lease_rejects_unknown_or_inactive_capability():
    registry = make_registry()
    with pytest.raises(CapabilityNotFoundError):
        with registry.execution_lease("missing.capability"):
            pytest.fail("unknown capability must not receive a lease")

    registry.disable(CAPABILITY)
    with pytest.raises(CapabilityNotFoundError):
        with registry.execution_lease(CAPABILITY):
            pytest.fail("inactive capability must not receive a lease")


def test_lifecycle_transition_for_another_capability_is_not_blocked():
    registry = make_registry()
    other_action = Action(
        name="read_inventory",
        description="Read inventory",
        handler=lambda **_: {"success": True},
    )
    registry.register_trusted(
        capability_id="business.inventory.read",
        action=other_action,
        risk_level="low",
        permissions=(PERMISSION,),
    )

    lease_entered = Event()
    release_lease = Event()
    transition_finished = Event()

    def hold_first_lease():
        with registry.execution_lease(CAPABILITY):
            lease_entered.set()
            assert release_lease.wait(timeout=5)

    def disable_other():
        registry.disable("business.inventory.read")
        transition_finished.set()

    reader = Thread(target=hold_first_lease, daemon=True)
    reader.start()
    assert lease_entered.wait(timeout=2)

    lifecycle = Thread(target=disable_other, daemon=True)
    lifecycle.start()
    assert transition_finished.wait(timeout=1), (
        "a lease for one capability must not block another capability"
    )

    release_lease.set()
    reader.join(timeout=2)
    lifecycle.join(timeout=2)
    assert not reader.is_alive()
    assert not lifecycle.is_alive()


def test_apply_lifecycle_transition_waits_for_inflight_read_before_sync():
    environment = BlockingEnvironment()
    registry = make_registry()
    gateway = make_gateway(environment, registry)
    read_errors = []
    transition_finished = Event()
    sync_called = Event()

    def run_read():
        try:
            gateway.read(make_principal(), make_request())
        except BaseException as exc:
            read_errors.append(exc)

    def run_transition():
        registry.apply_lifecycle_transition(
            CAPABILITY,
            CapabilityStatus.DISABLED,
            synchronize=sync_called.set,
        )
        transition_finished.set()

    reader = Thread(target=run_read, daemon=True)
    reader.start()
    assert environment.started.wait(timeout=2)

    lifecycle = Thread(target=run_transition, daemon=True)
    lifecycle.start()
    assert not sync_called.wait(timeout=0.15)
    assert not transition_finished.is_set()

    environment.release.set()
    reader.join(timeout=2)
    lifecycle.join(timeout=2)
    assert not reader.is_alive()
    assert not lifecycle.is_alive()
    assert read_errors == []
    assert sync_called.is_set()
    assert transition_finished.is_set()
    assert registry.get(CAPABILITY, include_inactive=True).status == CapabilityStatus.DISABLED


def test_apply_lifecycle_transition_keeps_lease_until_synchronization_finishes():
    registry = make_registry()
    sync_started = Event()
    release_sync = Event()
    transition_finished = Event()

    def transition():
        registry.apply_lifecycle_transition(
            CAPABILITY,
            CapabilityStatus.DISABLED,
            synchronize=lambda: (
                sync_started.set(),
                release_sync.wait(timeout=5),
            ),
        )
        transition_finished.set()

    worker = Thread(target=transition, daemon=True)
    worker.start()
    assert sync_started.wait(timeout=2)

    # While lifecycle synchronization is in progress, a new execution lease
    # cannot observe a partially synchronized ACTIVE/DISABLED state.
    lease_acquired = Event()

    def acquire_lease():
        try:
            with registry.execution_lease(CAPABILITY):
                lease_acquired.set()
        except CapabilityNotFoundError:
            # Expected after the disable transition is committed.
            pass

    lease_worker = Thread(target=acquire_lease, daemon=True)
    lease_worker.start()
    assert not lease_acquired.wait(timeout=0.15)

    release_sync.set()
    worker.join(timeout=2)
    lease_worker.join(timeout=2)
    assert not worker.is_alive()
    assert not lease_worker.is_alive()
    assert transition_finished.is_set()
    assert not lease_acquired.is_set()
    assert registry.get(CAPABILITY, include_inactive=True).status == CapabilityStatus.DISABLED
