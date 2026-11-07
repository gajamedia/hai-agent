from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class CapabilityProposalError(RuntimeError):
    """Base error for capability proposal operations."""


class CapabilityProposalValidationError(CapabilityProposalError):
    """Raised when proposal metadata is invalid."""


class CapabilityProposalStateError(CapabilityProposalError):
    """Raised when a proposal state transition is not allowed."""


class ProposalCategory(str, Enum):
    READ_ONLY = "read_only"
    STATE_CHANGING = "state_changing"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REGISTERED = "registered"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    # Prevent callers from mutating nested proposal metadata after submission.
    return _freeze_value(dict(value))


@dataclass(frozen=True)
class CapabilityProposal:
    proposal_id: str
    capability_id: str
    action_name: str
    description: str
    category: ProposalCategory
    risk_level: str
    permissions: tuple[str, ...] = ()
    preconditions: Mapping[str, Any] = field(default_factory=dict)
    effects: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: str = ""
    approved_by: str | None = None
    rejected_by: str | None = None
    rejection_reason: str | None = None
    registered_by: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "preconditions", _freeze_mapping(self.preconditions))
        object.__setattr__(self, "effects", _freeze_mapping(self.effects))
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))
        object.__setattr__(self, "permissions", tuple(self.permissions))


__all__ = [
    "CapabilityProposalError",
    "CapabilityProposalValidationError",
    "CapabilityProposalStateError",
    "ProposalCategory",
    "ProposalStatus",
    "CapabilityProposal",
]
