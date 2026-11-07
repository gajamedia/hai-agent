from dataclasses import FrozenInstanceError

import pytest

from agent.world.environment_contracts import (
    EnvironmentActionRequest,
    EnvironmentActionResult,
    EnvironmentDescriptor,
    EnvironmentMode,
    EnvironmentObservation,
)
from agent.world.environment_lifecycle import (
    EnvironmentInstanceError,
    EnvironmentInstanceStatus,
    EnvironmentLifecycleManager,
    EnvironmentRegistrationError,
)


class FakeAdapter:
    def __init__(self, descriptor):
        self._descriptor = descriptor

    @property
    def descriptor(self):
        return self._descriptor

    def observe(self, instance_id):
        return EnvironmentObservation(
            environment_id="test-env", instance_id=instance_id,
            revision="0", state={"ready": True},
        )

    def execute(self, instance_id, request):
        return EnvironmentActionResult(
            success=True, environment_id="test-env",
            instance_id=instance_id, previous_revision="0",
            current_revision="1", output={"ok": True},
        )

    def reset(self, instance_id):
        return self.observe(instance_id=instance_id)


def make_descriptor(provider="test.sim"):
    return EnvironmentDescriptor(
        environment_type=provider, domain="testing",
        mode=EnvironmentMode.SIMULATION, version="1.0",
        supports_observation=True, supports_reset=True,
        supports_idempotency=False, metadata={"source": "test"},
    )


def register(manager, descriptor=None, factory=None, provider="test.sim"):
    descriptor = descriptor or make_descriptor(provider)
    factory = factory or (lambda: FakeAdapter(descriptor))
    return manager.register_provider(provider, descriptor, factory)


def test_register_and_discover_trusted_provider():
    manager = EnvironmentLifecycleManager()
    register(manager)
    found = manager.discover_providers()
    assert len(found) == 1
    assert found[0].provider_id == "test.sim"
    assert found[0].active_instance_count == 0


def test_duplicate_provider_registration_is_rejected():
    manager = EnvironmentLifecycleManager()
    register(manager)
    with pytest.raises(EnvironmentRegistrationError):
        register(manager)


def test_descriptor_provider_must_match_registration_id():
    manager = EnvironmentLifecycleManager()
    with pytest.raises(EnvironmentRegistrationError):
        register(manager, descriptor=make_descriptor("other"), provider="test.sim")


def test_unregistered_provider_cannot_create_instance():
    with pytest.raises(EnvironmentInstanceError):
        EnvironmentLifecycleManager().create_instance("test.sim", "instance-1")


def test_instance_lifecycle_and_discovery_counts():
    manager = EnvironmentLifecycleManager()
    register(manager)
    created = manager.create_instance("test.sim", "instance-1")
    assert created.status == EnvironmentInstanceStatus.ACTIVE
    assert manager.discover_providers()[0].active_instance_count == 1
    closed = manager.close_instance("instance-1")
    assert closed.status == EnvironmentInstanceStatus.CLOSED
    assert manager.discover_providers()[0].active_instance_count == 0
    assert manager.list_instances() == ()
    assert manager.list_instances(include_closed=True) == (closed,)


def test_duplicate_instance_id_is_rejected_even_after_close():
    manager = EnvironmentLifecycleManager()
    register(manager)
    manager.create_instance("test.sim", "instance-1")
    manager.close_instance("instance-1")
    with pytest.raises(EnvironmentInstanceError):
        manager.create_instance("test.sim", "instance-1")


def test_cannot_unregister_provider_with_active_instance():
    manager = EnvironmentLifecycleManager()
    register(manager)
    manager.create_instance("test.sim", "instance-1")
    with pytest.raises(EnvironmentRegistrationError):
        manager.unregister_provider("test.sim")
    manager.close_instance("instance-1")
    manager.unregister_provider("test.sim")
    assert manager.discover_providers() == ()


def test_factory_failure_does_not_register_instance():
    manager = EnvironmentLifecycleManager()
    register(manager, factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(EnvironmentInstanceError):
        manager.create_instance("test.sim", "instance-1")
    assert manager.list_instances() == ()


def test_factory_must_return_protocol_adapter():
    manager = EnvironmentLifecycleManager()
    register(manager, factory=lambda: object())
    with pytest.raises(EnvironmentInstanceError):
        manager.create_instance("test.sim", "instance-1")
    assert manager.list_instances() == ()


def test_adapter_descriptor_must_match_registration():
    manager = EnvironmentLifecycleManager()
    descriptor = make_descriptor()
    wrong = make_descriptor("different.provider")
    register(manager, descriptor=descriptor, factory=lambda: FakeAdapter(wrong))
    with pytest.raises(EnvironmentInstanceError):
        manager.create_instance("test.sim", "instance-1")


def test_instance_snapshot_is_immutable_and_does_not_expose_adapter():
    manager = EnvironmentLifecycleManager()
    register(manager)
    snapshot = manager.create_instance("test.sim", "instance-1")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        snapshot.status = EnvironmentInstanceStatus.CLOSED
    assert not hasattr(snapshot, "adapter")


def test_close_unknown_or_already_closed_instance_fails():
    manager = EnvironmentLifecycleManager()
    with pytest.raises(EnvironmentInstanceError):
        manager.close_instance("missing")
    register(manager)
    manager.create_instance("test.sim", "instance-1")
    manager.close_instance("instance-1")
    with pytest.raises(EnvironmentInstanceError):
        manager.close_instance("instance-1")


def test_identifiers_are_validated():
    manager = EnvironmentLifecycleManager()
    with pytest.raises(ValueError):
        manager.register_provider("BAD ID", make_descriptor("BAD ID"), lambda: None)
    register(manager)
    with pytest.raises(ValueError):
        manager.create_instance("test.sim", "../unsafe")


def test_instance_lease_blocks_close_until_released():
    from threading import Event, Thread
    import time

    manager = EnvironmentLifecycleManager()
    register(manager)
    manager.create_instance("test.sim", "leased-1")
    lease_entered = Event()
    release_lease = Event()
    close_finished = Event()

    def hold_lease():
        with manager.lease_instance("leased-1") as lease:
            assert lease.snapshot.status == EnvironmentInstanceStatus.ACTIVE
            assert isinstance(lease.adapter, FakeAdapter)
            lease_entered.set()
            assert release_lease.wait(timeout=3)

    def close():
        manager.close_instance("leased-1")
        close_finished.set()

    worker = Thread(target=hold_lease)
    worker.start()
    assert lease_entered.wait(timeout=2)
    closer = Thread(target=close)
    closer.start()
    for _ in range(200):
        if manager.get_instance("leased-1").status == EnvironmentInstanceStatus.CLOSING:
            break
        time.sleep(0.005)
    assert manager.get_instance("leased-1").status == EnvironmentInstanceStatus.CLOSING
    assert not close_finished.is_set()
    with pytest.raises(EnvironmentInstanceError):
        with manager.lease_instance("leased-1"):
            pass
    release_lease.set()
    worker.join(timeout=2)
    closer.join(timeout=2)
    assert not worker.is_alive() and not closer.is_alive()
    assert close_finished.is_set()
    assert manager.get_instance("leased-1").status == EnvironmentInstanceStatus.CLOSED


def test_provider_cannot_unregister_while_instance_is_closing():
    from threading import Event, Thread
    import time

    manager = EnvironmentLifecycleManager()
    register(manager)
    manager.create_instance("test.sim", "leased-2")
    entered = Event()
    release = Event()

    def hold():
        with manager.lease_instance("leased-2"):
            entered.set()
            assert release.wait(timeout=3)

    worker = Thread(target=hold)
    worker.start()
    assert entered.wait(timeout=2)
    closer = Thread(target=lambda: manager.close_instance("leased-2"))
    closer.start()
    for _ in range(200):
        if manager.get_instance("leased-2").status == EnvironmentInstanceStatus.CLOSING:
            break
        time.sleep(0.005)
    with pytest.raises(EnvironmentRegistrationError):
        manager.unregister_provider("test.sim")
    release.set()
    worker.join(timeout=2)
    closer.join(timeout=2)
    assert not worker.is_alive() and not closer.is_alive()
    manager.unregister_provider("test.sim")
