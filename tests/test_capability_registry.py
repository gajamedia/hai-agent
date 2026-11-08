
import pytest

from agent.core.action import Action
from agent.llm.capability_registry import (
    CapabilityNotFoundError,
    CapabilityRegistrationError,
    CapabilityStatus,
    CapabilityTrustLevel,
    CapabilityValidationError,
    TrustedCapabilityRegistry,
)


def make_action(name="inspect_room"):
    return Action(
        name=name,
        description="Inspect the current room.",
        handler=lambda **kwargs: {
            "success": True,
        },
    )


def register_inspect_capability(registry):
    return registry.register_trusted(
        capability_id="environment.inspect",
        action=make_action(),
        risk_level="low",
        permissions=("environment.read",),
    )


def test_register_trusted_capability():
    registry = TrustedCapabilityRegistry()

    record = register_inspect_capability(registry)

    assert record.capability_id == "environment.inspect"
    assert record.action_name == "inspect_room"
    assert record.trust_level == CapabilityTrustLevel.APPLICATION
    assert record.status == CapabilityStatus.ACTIVE
    assert record.permissions == ("environment.read",)


def test_register_builtin_capability():
    registry = TrustedCapabilityRegistry()

    record = registry.register_builtin(
        make_action("open_box"),
    )

    assert record.capability_id == "builtin.open_box"
    assert record.trust_level == CapabilityTrustLevel.BUILTIN
    assert registry.contains_action("open_box")


def test_duplicate_capability_id_is_rejected():
    registry = TrustedCapabilityRegistry()

    register_inspect_capability(registry)

    with pytest.raises(CapabilityRegistrationError):
        registry.register_trusted(
            capability_id="environment.inspect",
            action=make_action("inspect_hall"),
            risk_level="low",
        )


def test_duplicate_action_name_is_rejected():
    registry = TrustedCapabilityRegistry()

    register_inspect_capability(registry)

    with pytest.raises(CapabilityRegistrationError):
        registry.register_trusted(
            capability_id="environment.inspect_again",
            action=make_action("inspect_room"),
            risk_level="low",
        )


def test_invalid_capability_id_is_rejected():
    registry = TrustedCapabilityRegistry()

    with pytest.raises(CapabilityValidationError):
        registry.register_trusted(
            capability_id="Run Python",
            action=make_action(),
            risk_level="low",
        )


def test_invalid_action_name_is_rejected():
    registry = TrustedCapabilityRegistry()

    with pytest.raises(CapabilityValidationError):
        registry.register_trusted(
            capability_id="environment.inspect",
            action=make_action("Run Python"),
            risk_level="low",
        )


def test_invalid_risk_level_is_rejected():
    registry = TrustedCapabilityRegistry()

    with pytest.raises(CapabilityValidationError):
        registry.register_trusted(
            capability_id="environment.inspect",
            action=make_action(),
            risk_level="unrestricted",
        )


def test_invalid_permission_is_rejected():
    registry = TrustedCapabilityRegistry()

    with pytest.raises(CapabilityValidationError):
        registry.register_trusted(
            capability_id="environment.inspect",
            action=make_action(),
            risk_level="low",
            permissions=("environment.read", " "),
        )


def test_string_is_not_accepted_as_permissions_iterable():
    registry = TrustedCapabilityRegistry()

    with pytest.raises(CapabilityValidationError):
        registry.register_trusted(
            capability_id="environment.inspect",
            action=make_action(),
            risk_level="low",
            permissions="environment.read",
        )


def test_unregistered_capability_cannot_be_retrieved():
    registry = TrustedCapabilityRegistry()

    with pytest.raises(CapabilityNotFoundError):
        registry.get("environment.missing")


def test_disabled_capability_is_not_active():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    record = registry.disable("environment.inspect")

    assert record.status == CapabilityStatus.DISABLED
    assert registry.is_active("environment.inspect") is False
    assert registry.contains_action("inspect_room") is False


def test_disabled_capability_remains_in_registry_history():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    registry.disable("environment.inspect")

    record = registry.get(
        "environment.inspect",
        include_inactive=True,
    )

    assert record.status == CapabilityStatus.DISABLED


def test_revoked_capability_cannot_be_reactivated():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    registry.revoke("environment.inspect")

    with pytest.raises(CapabilityRegistrationError):
        registry.disable("environment.inspect")

    with pytest.raises(CapabilityRegistrationError):
        registry.activate("environment.inspect")


def test_builtin_cannot_be_revoked():
    registry = TrustedCapabilityRegistry(
        builtin_actions=[
            make_action("open_box"),
        ],
    )

    with pytest.raises(CapabilityRegistrationError):
        registry.revoke("builtin.open_box")


def test_snapshot_is_read_only():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    snapshot = registry.snapshot()

    with pytest.raises(TypeError):
        snapshot.records["new.capability"] = object()


def test_snapshot_version_changes_after_registration():
    registry = TrustedCapabilityRegistry()
    initial_version = registry.snapshot().version

    register_inspect_capability(registry)

    assert registry.snapshot().version > initial_version


def test_list_active_excludes_revoked_capabilities():
    registry = TrustedCapabilityRegistry()

    registry.register_trusted(
        capability_id="environment.inspect",
        action=make_action("inspect_room"),
        risk_level="low",
    )

    registry.register_trusted(
        capability_id="environment.inspect_hall",
        action=make_action("inspect_hall"),
        risk_level="low",
    )

    registry.revoke("environment.inspect")

    active = registry.list_active()

    assert len(active) == 1
    assert active[0].capability_id == "environment.inspect_hall"


def test_get_by_action_returns_registered_capability():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    record = registry.get_by_action("inspect_room")

    assert record.capability_id == "environment.inspect"


def test_revoked_capability_remains_indexed_but_inactive():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    registry.revoke("environment.inspect")

    assert registry.contains_action("inspect_room") is False

    record = registry.get_by_action(
        "inspect_room",
        include_inactive=True,
    )

    assert record.status == CapabilityStatus.REVOKED


def test_disabled_capability_can_be_reactivated():
    registry = TrustedCapabilityRegistry()
    original = register_inspect_capability(registry)

    disabled = registry.disable("environment.inspect")
    activated = registry.activate("environment.inspect")

    assert original.status == CapabilityStatus.ACTIVE
    assert disabled.status == CapabilityStatus.DISABLED
    assert activated.status == CapabilityStatus.ACTIVE
    assert activated.version == original.version + 2
    assert registry.is_active("environment.inspect") is True
    assert registry.contains_action("inspect_room") is True


def test_repeated_disable_is_idempotent():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    first = registry.disable("environment.inspect")
    registry_version = registry.snapshot().version

    second = registry.disable("environment.inspect")

    assert second is first
    assert second.version == first.version
    assert registry.snapshot().version == registry_version


def test_repeated_activate_is_idempotent():
    registry = TrustedCapabilityRegistry()
    original = register_inspect_capability(registry)

    registry_version = registry.snapshot().version
    activated = registry.activate("environment.inspect")

    assert activated is original
    assert activated.version == original.version
    assert registry.snapshot().version == registry_version


def test_repeated_revoke_is_idempotent():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    first = registry.revoke("environment.inspect")
    registry_version = registry.snapshot().version

    second = registry.revoke("environment.inspect")

    assert second is first
    assert second.version == first.version
    assert registry.snapshot().version == registry_version


def test_disabled_capability_can_be_revoked():
    registry = TrustedCapabilityRegistry()
    register_inspect_capability(registry)

    registry.disable("environment.inspect")
    revoked = registry.revoke("environment.inspect")

    assert revoked.status == CapabilityStatus.REVOKED
    assert registry.is_active("environment.inspect") is False


def test_builtin_cannot_be_disabled_or_activated():
    registry = TrustedCapabilityRegistry(
        builtin_actions=[
            make_action("open_box"),
        ],
    )

    with pytest.raises(CapabilityRegistrationError):
        registry.disable("builtin.open_box")

    with pytest.raises(CapabilityRegistrationError):
        registry.activate("builtin.open_box")


def test_lifecycle_change_increments_registry_and_record_versions():
    registry = TrustedCapabilityRegistry()
    original = register_inspect_capability(registry)

    initial_registry_version = registry.snapshot().version

    disabled = registry.disable("environment.inspect")

    assert disabled.version == original.version + 1
    assert registry.snapshot().version == initial_registry_version + 1

    activated = registry.activate("environment.inspect")

    assert activated.version == disabled.version + 1
    assert registry.snapshot().version == initial_registry_version + 2


def test_activate_unknown_capability_raises():
    registry = TrustedCapabilityRegistry()

    with pytest.raises(CapabilityNotFoundError):
        registry.activate("environment.missing")
