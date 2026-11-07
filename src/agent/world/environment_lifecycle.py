"""Application-controlled environment provider discovery and instance lifecycle.

The manager owns adapter instances. Callers receive immutable metadata snapshots;
execution code may obtain a short-lived lease which pins one active adapter until
its context exits. Closing an instance first prevents new leases, then waits for
existing leases to drain before releasing the adapter reference.
"""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator

from .environment_contracts import EnvironmentAdapter, EnvironmentDescriptor


class EnvironmentLifecycleError(RuntimeError):
    """Base error for environment provider and instance lifecycle operations."""


class EnvironmentRegistrationError(EnvironmentLifecycleError):
    """Raised when a provider registration is invalid or conflicts."""


class EnvironmentInstanceError(EnvironmentLifecycleError):
    """Raised when an instance cannot be created, found, leased, or closed."""


class EnvironmentInstanceStatus(str, Enum):
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True)
class EnvironmentProviderSnapshot:
    provider_id: str
    descriptor: EnvironmentDescriptor
    active_instance_count: int


@dataclass(frozen=True)
class EnvironmentInstanceSnapshot:
    instance_id: str
    provider_id: str
    descriptor: EnvironmentDescriptor
    status: EnvironmentInstanceStatus


@dataclass(frozen=True)
class EnvironmentInstanceLease:
    """Short-lived adapter handle valid only inside ``lease_instance`` context.

    The adapter is intentionally not included in lifecycle snapshots or list APIs.
    Consumers must not retain this object after the context manager exits.
    """

    snapshot: EnvironmentInstanceSnapshot
    adapter: EnvironmentAdapter = field(repr=False, compare=False)


@dataclass(frozen=True)
class _ProviderRegistration:
    provider_id: str
    descriptor: EnvironmentDescriptor
    factory: Callable[[], EnvironmentAdapter]


@dataclass
class _ManagedInstance:
    snapshot: EnvironmentInstanceSnapshot
    adapter: EnvironmentAdapter | None
    active_leases: int = 0


_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_.-]{1,127}$")
_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _validate_identifier(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{label} must be a valid non-empty identifier.")
    return value


class EnvironmentLifecycleManager:
    """Thread-safe registry for trusted providers and managed adapter instances.

    Provider registration is an application/deployment responsibility. There is
    intentionally no API for model-generated proposals to register providers.
    Adapter references are exposed only through an active instance lease.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._providers: dict[str, _ProviderRegistration] = {}
        self._instances: dict[str, _ManagedInstance] = {}

    def register_provider(
        self,
        provider_id: str,
        descriptor: EnvironmentDescriptor,
        factory: Callable[[], EnvironmentAdapter],
    ) -> EnvironmentProviderSnapshot:
        provider_id = _validate_identifier(provider_id, _PROVIDER_ID, "provider_id")
        if not isinstance(descriptor, EnvironmentDescriptor):
            raise EnvironmentRegistrationError("descriptor must be EnvironmentDescriptor.")
        if descriptor.environment_type != provider_id:
            raise EnvironmentRegistrationError(
                "descriptor.environment_type must match provider_id."
            )
        if not callable(factory):
            raise EnvironmentRegistrationError("factory must be callable.")

        with self._lock:
            if provider_id in self._providers:
                raise EnvironmentRegistrationError(
                    f"Environment provider '{provider_id}' is already registered."
                )
            self._providers[provider_id] = _ProviderRegistration(
                provider_id=provider_id,
                descriptor=descriptor,
                factory=factory,
            )
            return EnvironmentProviderSnapshot(provider_id, descriptor, 0)

    def unregister_provider(self, provider_id: str) -> None:
        provider_id = _validate_identifier(provider_id, _PROVIDER_ID, "provider_id")
        with self._lock:
            if provider_id not in self._providers:
                raise EnvironmentRegistrationError(
                    f"Environment provider '{provider_id}' is not registered."
                )
            if any(
                item.snapshot.provider_id == provider_id
                and item.snapshot.status != EnvironmentInstanceStatus.CLOSED
                for item in self._instances.values()
            ):
                raise EnvironmentRegistrationError(
                    "Cannot unregister a provider while active or closing instances exist."
                )
            del self._providers[provider_id]

    def discover_providers(self) -> tuple[EnvironmentProviderSnapshot, ...]:
        with self._lock:
            snapshots = []
            for provider_id, registration in sorted(self._providers.items()):
                count = sum(
                    1 for item in self._instances.values()
                    if item.snapshot.provider_id == provider_id
                    and item.snapshot.status != EnvironmentInstanceStatus.CLOSED
                )
                snapshots.append(EnvironmentProviderSnapshot(
                    provider_id=provider_id,
                    descriptor=registration.descriptor,
                    active_instance_count=count,
                ))
            return tuple(snapshots)

    def create_instance(self, provider_id: str, instance_id: str) -> EnvironmentInstanceSnapshot:
        provider_id = _validate_identifier(provider_id, _PROVIDER_ID, "provider_id")
        instance_id = _validate_identifier(instance_id, _INSTANCE_ID, "instance_id")
        with self._lock:
            registration = self._providers.get(provider_id)
            if registration is None:
                raise EnvironmentInstanceError(
                    f"Environment provider '{provider_id}' is not registered."
                )
            if instance_id in self._instances:
                raise EnvironmentInstanceError(
                    f"Environment instance '{instance_id}' has already been used."
                )
            try:
                adapter = registration.factory()
            except Exception as exc:
                raise EnvironmentInstanceError(
                    f"Factory for provider '{provider_id}' failed."
                ) from exc
            if not isinstance(adapter, EnvironmentAdapter):
                raise EnvironmentInstanceError(
                    "Provider factory must return an EnvironmentAdapter."
                )
            if adapter.descriptor != registration.descriptor:
                raise EnvironmentInstanceError(
                    "Adapter descriptor does not match its trusted registration."
                )
            snapshot = EnvironmentInstanceSnapshot(
                instance_id=instance_id,
                provider_id=provider_id,
                descriptor=registration.descriptor,
                status=EnvironmentInstanceStatus.ACTIVE,
            )
            self._instances[instance_id] = _ManagedInstance(snapshot=snapshot, adapter=adapter)
            return snapshot

    def get_instance(self, instance_id: str) -> EnvironmentInstanceSnapshot:
        instance_id = _validate_identifier(instance_id, _INSTANCE_ID, "instance_id")
        with self._lock:
            managed = self._instances.get(instance_id)
            if managed is None:
                raise EnvironmentInstanceError(
                    f"Environment instance '{instance_id}' does not exist."
                )
            return managed.snapshot

    def list_instances(
        self, *, include_closed: bool = False
    ) -> tuple[EnvironmentInstanceSnapshot, ...]:
        if not isinstance(include_closed, bool):
            raise ValueError("include_closed must be a bool.")
        with self._lock:
            values = [
                item.snapshot for item in self._instances.values()
                if include_closed or item.snapshot.status != EnvironmentInstanceStatus.CLOSED
            ]
            return tuple(sorted(values, key=lambda item: item.instance_id))

    @contextmanager
    def lease_instance(self, instance_id: str) -> Iterator[EnvironmentInstanceLease]:
        """Pin an active adapter; close blocks until this context has exited."""
        instance_id = _validate_identifier(instance_id, _INSTANCE_ID, "instance_id")
        with self._condition:
            managed = self._instances.get(instance_id)
            if managed is None or managed.snapshot.status != EnvironmentInstanceStatus.ACTIVE:
                raise EnvironmentInstanceError(
                    f"Environment instance '{instance_id}' is unknown, closing, or closed."
                )
            adapter = managed.adapter
            if adapter is None:
                raise EnvironmentInstanceError("Active environment instance has no adapter.")
            managed.active_leases += 1
            lease = EnvironmentInstanceLease(managed.snapshot, adapter)
        try:
            yield lease
        finally:
            with self._condition:
                current = self._instances.get(instance_id)
                if current is not None:
                    current.active_leases -= 1
                    if current.active_leases < 0:
                        current.active_leases = 0
                        raise RuntimeError("Environment instance lease count underflow.")
                self._condition.notify_all()

    def close_instance(self, instance_id: str) -> EnvironmentInstanceSnapshot:
        """Block new leases, drain current leases, then release the adapter."""
        instance_id = _validate_identifier(instance_id, _INSTANCE_ID, "instance_id")
        with self._condition:
            managed = self._instances.get(instance_id)
            if managed is None:
                raise EnvironmentInstanceError(
                    f"Environment instance '{instance_id}' does not exist."
                )
            if managed.snapshot.status != EnvironmentInstanceStatus.ACTIVE:
                raise EnvironmentInstanceError(
                    f"Environment instance '{instance_id}' is already closing or closed."
                )
            managed.snapshot = EnvironmentInstanceSnapshot(
                instance_id=managed.snapshot.instance_id,
                provider_id=managed.snapshot.provider_id,
                descriptor=managed.snapshot.descriptor,
                status=EnvironmentInstanceStatus.CLOSING,
            )
            self._condition.notify_all()
            try:
                while managed.active_leases:
                    self._condition.wait()
            except BaseException:
                # If shutdown/cancellation interrupts the wait, restore the
                # instance to service; its adapter was never released.
                managed.snapshot = EnvironmentInstanceSnapshot(
                    instance_id=managed.snapshot.instance_id,
                    provider_id=managed.snapshot.provider_id,
                    descriptor=managed.snapshot.descriptor,
                    status=EnvironmentInstanceStatus.ACTIVE,
                )
                self._condition.notify_all()
                raise
            managed.snapshot = EnvironmentInstanceSnapshot(
                instance_id=managed.snapshot.instance_id,
                provider_id=managed.snapshot.provider_id,
                descriptor=managed.snapshot.descriptor,
                status=EnvironmentInstanceStatus.CLOSED,
            )
            managed.adapter = None
            self._condition.notify_all()
            return managed.snapshot


__all__ = [
    "EnvironmentLifecycleError",
    "EnvironmentRegistrationError",
    "EnvironmentInstanceError",
    "EnvironmentInstanceStatus",
    "EnvironmentProviderSnapshot",
    "EnvironmentInstanceSnapshot",
    "EnvironmentInstanceLease",
    "EnvironmentLifecycleManager",
]
