from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


class CapabilityExecutionError(RuntimeError):
    """Base error for the read-only capability execution boundary."""


class CapabilityRequestValidationError(CapabilityExecutionError):
    """Raised when a capability read request is malformed."""


class CapabilityAuthorizationError(CapabilityExecutionError):
    """Raised when the application does not authorize a capability read."""


class CapabilityResolutionError(CapabilityExecutionError):
    """Raised when a target or capability is not registered."""


class CapabilityAdapterError(CapabilityExecutionError):
    """Raised when a trusted adapter cannot complete a read operation."""


@dataclass(frozen=True)
class CapabilityReadRequest:
    """Immutable request envelope for a read-only capability operation."""

    target: str
    capability_id: str
    actor_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("target", "capability_id", "actor_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise CapabilityRequestValidationError(
                    f"{field_name} must be a non-empty string."
                )
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.arguments, Mapping):
            raise CapabilityRequestValidationError("arguments must be a mapping.")
        # Detach the request from caller-owned top-level mutation.
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class CapabilityReadResult:
    """Result envelope returned by a trusted read adapter."""

    target: str
    capability_id: str
    data: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CapabilityReadAdapter(Protocol):
    """Application-owned adapter contract; deliberately has no write method."""

    @property
    def target(self) -> str: ...

    def supports(self, capability_id: str) -> bool: ...

    def read(self, capability_id: str, arguments: Mapping[str, Any]) -> Any: ...


__all__ = [
    "CapabilityExecutionError",
    "CapabilityRequestValidationError",
    "CapabilityAuthorizationError",
    "CapabilityResolutionError",
    "CapabilityAdapterError",
    "CapabilityReadRequest",
    "CapabilityReadResult",
    "CapabilityReadAdapter",
]
