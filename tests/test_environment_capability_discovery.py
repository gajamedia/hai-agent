import pytest

from agent.core.action import Action
from agent.llm.capability_registry import (
    CapabilityStatus, CapabilityTrustLevel, TrustedCapabilityRegistry,
)
from agent.world.environment_contracts import EnvironmentDescriptor, EnvironmentMode
from agent.world.environment_lifecycle import EnvironmentLifecycleManager
from agent.world.environment_capability_discovery import (
    EnvironmentCapabilityBindingError,
    EnvironmentCapabilityDiscovery,
    EnvironmentCapabilityDiscoveryError,
    EnvironmentCapabilityState,
)


class Adapter:
    def __init__(self, descriptor):
        self.descriptor = descriptor

    def observe(self, instance_id):
        raise AssertionError("Discovery must not observe or execute an adapter.")

    def execute(self, instance_id, request):
        raise AssertionError("Discovery must never execute an adapter.")

    def reset(self, instance_id):
        raise AssertionError("Discovery must never reset an adapter.")


def descriptor(env_type="demo.sim"):
    return EnvironmentDescriptor(
        environment_type=env_type, domain="demo", mode=EnvironmentMode.SIMULATION,
        supports_observation=True, supports_reset=True,
    )


def make_registry():
    registry = TrustedCapabilityRegistry()
    action = Action(
        name="open_gate", description="Open the gate",
        handler=lambda **kwargs: {"opened": True},
    )
    record = registry.register_trusted(
        capability_id="demo.open_gate", action=action,
        risk_level="medium", permissions=("gate:open",),
    )
    return registry, record


def setup():
    lifecycle = EnvironmentLifecycleManager()
    desc = descriptor()
    lifecycle.register_provider("demo.sim", desc, lambda: Adapter(desc))
    lifecycle.create_instance("demo.sim", "sim-1")
    registry, record = make_registry()
    discovery = EnvironmentCapabilityDiscovery(
        environments=lifecycle, registry=registry,
        bindings={"demo.sim": ("demo.open_gate",)},
    )
    return lifecycle, registry, record, discovery


def test_discovers_only_application_bound_active_trusted_capabilities():
    _, _, record, discovery = setup()
    report = discovery.discover("sim-1")
    assert report.available_capability_ids == ("demo.open_gate",)
    assert report.excluded_capability_ids == ()
    candidate = report.candidates[0]
    assert candidate.action_name == record.action_name
    assert candidate.permissions == ("gate:open",)
    assert candidate.state == EnvironmentCapabilityState.AVAILABLE
    assert not hasattr(candidate, "action")
    assert not hasattr(candidate, "handler")


def test_disabled_capability_is_excluded():
    _, registry, _, discovery = setup()
    registry.disable("demo.open_gate")
    report = discovery.discover("sim-1")
    assert report.available_capability_ids == ()
    assert report.excluded_capability_ids == ("demo.open_gate",)
    assert report.candidates[0].state == EnvironmentCapabilityState.INACTIVE


def test_revoked_capability_is_excluded():
    _, registry, _, discovery = setup()
    registry.revoke("demo.open_gate")
    assert discovery.discover("sim-1").available_capability_ids == ()


def test_missing_bound_capability_is_reported_not_registered():
    lifecycle = EnvironmentLifecycleManager()
    desc = descriptor()
    lifecycle.register_provider("demo.sim", desc, lambda: Adapter(desc))
    lifecycle.create_instance("demo.sim", "sim-1")
    registry = TrustedCapabilityRegistry()
    discovery = EnvironmentCapabilityDiscovery(
        environments=lifecycle, registry=registry,
        bindings={"demo.sim": ("demo.unknown",)},
    )
    report = discovery.discover("sim-1")
    assert report.candidates[0].state == EnvironmentCapabilityState.MISSING
    assert registry.list_all() == ()


def test_environment_without_application_binding_discovers_nothing():
    lifecycle, registry, _, _ = setup()
    discovery = EnvironmentCapabilityDiscovery(
        environments=lifecycle, registry=registry,
        bindings={"another.sim": ("demo.open_gate",)},
    )
    report = discovery.discover("sim-1")
    assert report.candidates == ()
    assert report.available_capability_ids == ()


def test_closed_environment_cannot_discover():
    lifecycle, registry, _, _ = setup()
    lifecycle.close_instance("sim-1")
    discovery = EnvironmentCapabilityDiscovery(
        environments=lifecycle, registry=registry,
        bindings={"demo.sim": ("demo.open_gate",)},
    )
    with pytest.raises(EnvironmentCapabilityDiscoveryError):
        discovery.discover("sim-1")


def test_unknown_instance_fails_closed():
    _, _, _, discovery = setup()
    with pytest.raises(EnvironmentCapabilityDiscoveryError):
        discovery.discover("missing")


def test_duplicate_capability_binding_rejected():
    lifecycle = EnvironmentLifecycleManager()
    registry = TrustedCapabilityRegistry()
    with pytest.raises(EnvironmentCapabilityBindingError):
        EnvironmentCapabilityDiscovery(
            environments=lifecycle, registry=registry,
            bindings={"demo.sim": ("demo.x", "demo.x")},
        )


def test_string_is_not_accepted_as_capability_id_sequence():
    lifecycle = EnvironmentLifecycleManager()
    registry = TrustedCapabilityRegistry()
    with pytest.raises(EnvironmentCapabilityBindingError):
        EnvironmentCapabilityDiscovery(
            environments=lifecycle, registry=registry,
            bindings={"demo.sim": "demo.x"},
        )


def test_empty_bindings_rejected():
    lifecycle = EnvironmentLifecycleManager()
    registry = TrustedCapabilityRegistry()
    with pytest.raises(EnvironmentCapabilityBindingError):
        EnvironmentCapabilityDiscovery(
            environments=lifecycle, registry=registry, bindings={},
        )


def test_binding_configuration_is_immutable():
    lifecycle = EnvironmentLifecycleManager()
    registry = TrustedCapabilityRegistry()
    source = {"demo.sim": ["demo.x"]}
    discovery = EnvironmentCapabilityDiscovery(
        environments=lifecycle, registry=registry, bindings=source,
    )
    source["demo.sim"].append("demo.y")
    assert discovery.bindings["demo.sim"] == ("demo.x",)
    with pytest.raises(TypeError):
        discovery.bindings["new"] = ("demo.z",)


def test_discovery_does_not_mutate_registry_or_execute_actions():
    _, registry, record, discovery = setup()
    before_version = registry.snapshot().version
    report = discovery.discover("sim-1")
    assert registry.snapshot().version == before_version
    assert report.candidates[0].capability_id == record.capability_id
