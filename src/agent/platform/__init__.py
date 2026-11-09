"""Shared platform for agent features: the contract and the registry.

See docs/product/PHASE1_FOUNDATION_AUDIT.md. Nothing here executes an operation.
"""

from .contracts import (
    ApprovalPolicy,
    ContractError,
    Effect,
    Feature,
    Operation,
    TERMINAL_STATUSES,
    TaskStatus,
    Verdict,
    Verification,
    can_transition,
)
from .registry import OperationDisabled, OperationRegistry, RegistryError, UnknownOperation

__all__ = [
    "ApprovalPolicy", "ContractError", "Effect", "Feature", "Operation", "OperationDisabled",
    "OperationRegistry", "RegistryError", "TERMINAL_STATUSES", "TaskStatus", "UnknownOperation",
    "Verdict", "Verification", "can_transition",
]
