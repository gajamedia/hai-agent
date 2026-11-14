"""Shared platform for agent features: contract, registry, durable task store and pipeline.

See docs/product/PHASE1_FOUNDATION_AUDIT.md. Features declare operations; every run goes
through PlatformPipeline (authorize, plan, preview, approve, execute, verify, audit).
"""

from .contracts import (
    ApprovalPolicy,
    ContractError,
    Effect,
    Feature,
    Operation,
    PlanUnsupported,
    TERMINAL_STATUSES,
    TaskStatus,
    Verdict,
    Verification,
    can_transition,
)
from .pipeline import (
    ApprovalRequired, ConfirmationMismatch, ExecutionFailed, PermissionDenied, PipelineError, PlanError,
    PlatformPipeline, PolicyViolation, RunContext, RunOutcome, StepUpRequired, TaskExpired, TooManyTasks,
)
from .registry import OperationDisabled, OperationRegistry, RegistryError, UnknownOperation
from .store import SQLiteTaskStore, TaskConflict, TaskNotFound, TaskRecord

__all__ = [
    "ApprovalRequired", "ConfirmationMismatch", "ExecutionFailed", "PermissionDenied", "PipelineError",
    "PlanError", "PlatformPipeline", "PolicyViolation", "RunContext", "RunOutcome", "SQLiteTaskStore",
    "PlanUnsupported", "StepUpRequired", "TaskConflict", "TaskExpired", "TooManyTasks", "TaskNotFound", "TaskRecord",
    "ApprovalPolicy", "ContractError", "Effect", "Feature", "Operation", "OperationDisabled",
    "OperationRegistry", "RegistryError", "TERMINAL_STATUSES", "TaskStatus", "UnknownOperation",
    "Verdict", "Verification", "can_transition",
]
