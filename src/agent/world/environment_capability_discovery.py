"""Resolve environment capability bindings against the trusted registry.

Bindings are application-owned configuration. Environment metadata and model
proposals never register capabilities, create handlers, or authorize execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Mapping

from agent.llm.capability_registry import (
    CapabilityStatus,
    CapabilityTrustLevel,
    TrustedCapabilityRegistry,
)
from agent.world.environment_lifecycle import (
    EnvironmentInstanceError,
    EnvironmentInstanceStatus,
    EnvironmentLifecycleManager,
)


class EnvironmentCapabilityDiscoveryError(RuntimeError):
    """Base error for environment-to-capability discovery."""


class EnvironmentCapabilityBindingError(EnvironmentCapabilityDiscoveryError):
    """Raised for invalid trusted application bindings."""


class EnvironmentCapabilityState(str, Enum):
    AVAILABLE = "available"
    INACTIVE = "inactive"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class EnvironmentCapabilityCandidate:
    environment_instance_id: str
    environment_type: str
    capability_id: str
    action_name: str
    description: str
    risk_level: str
    permissions: tuple[str, ...]
    trust_level: CapabilityTrustLevel
    registry_version: int
    state: EnvironmentCapabilityState


@dataclass(frozen=True, slots=True)
class EnvironmentCapabilityDiscoveryReport:
    environment_instance_id: str
    environment_type: str
    registry_version: int
    candidates: tuple[EnvironmentCapabilityCandidate, ...]
    available_capability_ids: tuple[str, ...]
    excluded_capability_ids: tuple[str, ...]


class EnvironmentCapabilityDiscovery:
    """Discover only pre-registered capabilities explicitly bound by the app.

    `bindings` maps environment_type -> capability IDs. It is supplied by
    trusted application configuration, never by the environment or an LLM.
    The resolver returns metadata only; it does not expose Action handlers,
    mutate Agent/Planner, acquire execution leases, or execute operations.
    """

    _ALLOWED_TRUST = frozenset({
        CapabilityTrustLevel.BUILTIN,
        CapabilityTrustLevel.APPLICATION,
        CapabilityTrustLevel.APPROVED_PROPOSAL,
    })

    def __init__(
        self,
        *,
        environments: EnvironmentLifecycleManager,
        registry: TrustedCapabilityRegistry,
        bindings: Mapping[str, tuple[str, ...] | list[str]],
    ) -> None:
        if not isinstance(environments, EnvironmentLifecycleManager):
            raise EnvironmentCapabilityBindingError(
                "environments must be an EnvironmentLifecycleManager."
            )
        if not isinstance(registry, TrustedCapabilityRegistry):
            raise EnvironmentCapabilityBindingError(
                "registry must be a TrustedCapabilityRegistry."
            )
        if not isinstance(bindings, Mapping) or not bindings:
            raise EnvironmentCapabilityBindingError(
                "bindings must be a non-empty application-owned mapping."
            )

        normalized: dict[str, tuple[str, ...]] = {}
        for environment_type, capability_ids in bindings.items():
            if not isinstance(environment_type, str) or not environment_type.strip():
                raise EnvironmentCapabilityBindingError(
                    "environment_type keys must be non-empty strings."
                )
            if isinstance(capability_ids, (str, bytes)) or not isinstance(
                capability_ids, (tuple, list)
            ):
                raise EnvironmentCapabilityBindingError(
                    "Each binding must be a list or tuple of capability IDs."
                )
            ids = tuple(capability_ids)
            if not ids or any(
                not isinstance(item, str) or not item.strip() for item in ids
            ):
                raise EnvironmentCapabilityBindingError(
                    "Capability ID bindings must be non-empty strings."
                )
            if len(set(ids)) != len(ids):
                raise EnvironmentCapabilityBindingError(
                    "Duplicate capability IDs are not allowed in a binding."
                )
            normalized[environment_type.strip()] = ids

        self.environments = environments
        self.registry = registry
        self._bindings = MappingProxyType(normalized)
        self._lock = RLock()

    @property
    def bindings(self) -> Mapping[str, tuple[str, ...]]:
        return self._bindings

    def discover(self, instance_id: str) -> EnvironmentCapabilityDiscoveryReport:
        with self._lock:
            try:
                instance = self.environments.get_instance(instance_id)
            except EnvironmentInstanceError as exc:
                raise EnvironmentCapabilityDiscoveryError(
                    "Environment instance is unavailable."
                ) from exc

            if instance.status != EnvironmentInstanceStatus.ACTIVE:
                raise EnvironmentCapabilityDiscoveryError(
                    "Capabilities cannot be discovered for a closed environment."
                )

            environment_type = instance.descriptor.environment_type
            capability_ids = self._bindings.get(environment_type, ())
            snapshot = self.registry.snapshot()
            candidates = []
            available = []
            excluded = []

            for capability_id in capability_ids:
                record = snapshot.records.get(capability_id)
                if record is None:
                    candidates.append(EnvironmentCapabilityCandidate(
                        environment_instance_id=instance_id,
                        environment_type=environment_type,
                        capability_id=capability_id,
                        action_name="",
                        description="",
                        risk_level="",
                        permissions=(),
                        trust_level=CapabilityTrustLevel.APPLICATION,
                        registry_version=snapshot.version,
                        state=EnvironmentCapabilityState.MISSING,
                    ))
                    excluded.append(capability_id)
                    continue

                if record.status != CapabilityStatus.ACTIVE or record.trust_level not in self._ALLOWED_TRUST:
                    state = EnvironmentCapabilityState.INACTIVE
                    excluded.append(capability_id)
                else:
                    state = EnvironmentCapabilityState.AVAILABLE
                    available.append(capability_id)

                candidates.append(EnvironmentCapabilityCandidate(
                    environment_instance_id=instance_id,
                    environment_type=environment_type,
                    capability_id=record.capability_id,
                    action_name=record.action_name,
                    description=record.description,
                    risk_level=record.risk_level,
                    permissions=tuple(record.permissions),
                    trust_level=record.trust_level,
                    registry_version=snapshot.version,
                    state=state,
                ))

            if self.registry.snapshot().version != snapshot.version:
                raise EnvironmentCapabilityDiscoveryError(
                    "Capability registry changed during discovery."
                )

            return EnvironmentCapabilityDiscoveryReport(
                environment_instance_id=instance_id,
                environment_type=environment_type,
                registry_version=snapshot.version,
                candidates=tuple(candidates),
                available_capability_ids=tuple(available),
                excluded_capability_ids=tuple(excluded),
            )


__all__ = [
    "EnvironmentCapabilityDiscoveryError",
    "EnvironmentCapabilityBindingError",
    "EnvironmentCapabilityState",
    "EnvironmentCapabilityCandidate",
    "EnvironmentCapabilityDiscoveryReport",
    "EnvironmentCapabilityDiscovery",
]
