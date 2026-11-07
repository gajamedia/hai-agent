"""Generic, provider-neutral contracts for hAI Agent environments.

These contracts describe environment metadata and I/O. They do not authorize
actions, register capabilities, or provide an execution bypass. Production
execution must continue through the application's trusted capability gateway
and canonical executor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Protocol, runtime_checkable
from uuid import UUID


class EnvironmentContractError(ValueError):
    """Raised when generic environment contract data is invalid."""


class EnvironmentMode(str, Enum):
    SIMULATION = "simulation"
    REAL_WORLD = "real_world"


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnvironmentContractError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _validate_json_value(value: Any, path: str = "value") -> None:
    """Accept only finite JSON-compatible values; reject executable objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise EnvironmentContractError(f"{path} must be finite.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise EnvironmentContractError(f"{path} keys must be non-empty strings.")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise EnvironmentContractError(
        f"{path} contains unsupported value type {type(value).__name__}."
    )


def _freeze_json(value: Any) -> Any:
    from types import MappingProxyType
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _frozen_json_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentContractError(f"{field_name} must be a mapping.")
    copied = deepcopy(dict(value))
    _validate_json_value(copied, field_name)
    # Freeze nested mappings and arrays as well as the outer dataclass.
    return _freeze_json(copied)


@dataclass(frozen=True, slots=True)
class EnvironmentDescriptor:
    """Trusted application-supplied description of one environment provider."""

    environment_type: str
    domain: str
    mode: EnvironmentMode
    version: str = "1"
    supports_observation: bool = True
    supports_reset: bool = False
    supports_idempotency: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment_type", _non_empty_string(
            self.environment_type, "environment_type"
        ))
        object.__setattr__(self, "domain", _non_empty_string(self.domain, "domain"))
        object.__setattr__(self, "version", _non_empty_string(self.version, "version"))
        if not isinstance(self.mode, EnvironmentMode):
            raise EnvironmentContractError("mode must be an EnvironmentMode.")
        for name in ("supports_observation", "supports_reset", "supports_idempotency"):
            if not isinstance(getattr(self, name), bool):
                raise EnvironmentContractError(f"{name} must be a boolean.")
        object.__setattr__(self, "metadata", _frozen_json_mapping(
            self.metadata, "metadata"
        ))


@dataclass(frozen=True, slots=True)
class EnvironmentObservation:
    """Snapshot returned by an adapter; revision is provider-owned and opaque."""

    environment_id: str
    instance_id: str
    revision: str
    state: Mapping[str, Any]
    observed_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("environment_id", "instance_id", "revision"):
            object.__setattr__(self, name, _non_empty_string(
                getattr(self, name), name
            ))
        if self.observed_at is not None:
            object.__setattr__(self, "observed_at", _non_empty_string(
                self.observed_at, "observed_at"
            ))
        object.__setattr__(self, "state", _frozen_json_mapping(self.state, "state"))


@dataclass(frozen=True, slots=True)
class EnvironmentActionRequest:
    """Adapter operation envelope, not an authorization grant.

    action_id must resolve to an application-registered trusted capability.
    Callers must not treat this request object as proof of permission.
    """

    action_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    expected_revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _non_empty_string(
            self.action_id, "action_id"
        ))
        object.__setattr__(self, "arguments", _frozen_json_mapping(
            self.arguments, "arguments"
        ))
        for name in ("idempotency_key", "expected_revision"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _non_empty_string(value, name))


@dataclass(frozen=True, slots=True)
class EnvironmentActionResult:
    """Outcome envelope; success must be verified against a fresh observation."""

    success: bool
    environment_id: str
    instance_id: str
    previous_revision: str
    current_revision: str
    output: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    outcome_known: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise EnvironmentContractError("success must be a boolean.")
        if not isinstance(self.outcome_known, bool):
            raise EnvironmentContractError("outcome_known must be a boolean.")
        for name in (
            "environment_id", "instance_id", "previous_revision",
            "current_revision",
        ):
            object.__setattr__(self, name, _non_empty_string(
                getattr(self, name), name
            ))
        if self.error_code is not None:
            object.__setattr__(self, "error_code", _non_empty_string(
                self.error_code, "error_code"
            ))
        object.__setattr__(self, "output", _frozen_json_mapping(
            self.output, "output"
        ))
        if not self.outcome_known and self.success:
            raise EnvironmentContractError(
                "An unknown outcome cannot be reported as successful."
            )


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Provider interface. Implementations are injected by trusted application code."""

    @property
    def descriptor(self) -> EnvironmentDescriptor:
        ...

    def observe(self, *, instance_id: str) -> EnvironmentObservation:
        ...

    def execute(
        self,
        *,
        instance_id: str,
        request: EnvironmentActionRequest,
    ) -> EnvironmentActionResult:
        ...

    def reset(self, *, instance_id: str) -> EnvironmentObservation:
        ...


__all__ = [
    "EnvironmentActionRequest",
    "EnvironmentActionResult",
    "EnvironmentAdapter",
    "EnvironmentContractError",
    "EnvironmentDescriptor",
    "EnvironmentMode",
    "EnvironmentObservation",
]
