"""Contract for agent features.

A *feature* (for example the Database Agent) declares one or more *operations*.
This module only describes them and checks that the description is complete and
safe. It runs nothing: no operation is executed here, no credential or
connection is created, and importing it has no side effects.

The rules enforced here are the ones from docs/product/PHASE1_FOUNDATION_AUDIT.md:

* an operation that changes the world (``write`` / ``external``) must offer a
  preview, name its approval policy, and either verify its result or say why it
  cannot;
* the risk score is a number between 0 and 1;
* identifiers are stable, lower-case and namespaced by feature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


class ContractError(ValueError):
    """The feature or operation description is invalid."""


class PlanUnsupported(Exception):
    """Raised by an operation's ``plan`` when the request cannot be turned into a plan.

    That is a normal answer, not a failure: the pipeline records it, creates no task, and
    re-raises it so the caller can show ``reason`` (and any ``details``) to the person.
    """

    def __init__(self, reason: str, details: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = dict(details or {})


class Effect(str, Enum):
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"

    @property
    def changes_world(self) -> bool:
        return self is not Effect.READ


@dataclass(frozen=True)
class ApprovalPolicy:
    """What a person must do before an operation may run.

    ``required`` is true when any of the other conditions is set. An operation
    with no condition set declares, explicitly, that it needs no approval.
    """

    confirm_phrase: bool = False
    fresh_login: bool = False
    second_approver: bool = False

    @property
    def required(self) -> bool:
        return self.confirm_phrase or self.fresh_login or self.second_approver

    def covers(self, baseline: "ApprovalPolicy") -> bool:
        """True when this policy demands at least everything ``baseline`` demands."""
        return ((self.confirm_phrase or not baseline.confirm_phrase)
                and (self.fresh_login or not baseline.fresh_login)
                and (self.second_approver or not baseline.second_approver))

    def as_dict(self) -> dict:
        return {"confirm_phrase": self.confirm_phrase, "fresh_login": self.fresh_login,
                "second_approver": self.second_approver}

    @classmethod
    def from_dict(cls, data) -> "ApprovalPolicy":
        return cls(confirm_phrase=bool(data["confirm_phrase"]), fresh_login=bool(data["fresh_login"]),
                   second_approver=bool(data["second_approver"]))

    @classmethod
    def none(cls) -> "ApprovalPolicy":
        return cls()


class TaskStatus(str, Enum):
    """Lifecycle shared by every operation run."""

    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_STATUSES


TERMINAL_STATUSES = frozenset({
    TaskStatus.COMPLETED, TaskStatus.UNVERIFIED, TaskStatus.FAILED,
    TaskStatus.CANCELLED, TaskStatus.REJECTED,
})

_TRANSITIONS: Mapping[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PLANNED: frozenset({
        TaskStatus.AWAITING_APPROVAL, TaskStatus.APPROVED,
        TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.AWAITING_APPROVAL: frozenset({
        TaskStatus.APPROVED, TaskStatus.REJECTED, TaskStatus.CANCELLED}),
    TaskStatus.APPROVED: frozenset({TaskStatus.EXECUTING, TaskStatus.CANCELLED}),
    TaskStatus.EXECUTING: frozenset({TaskStatus.VERIFYING, TaskStatus.FAILED}),
    TaskStatus.VERIFYING: frozenset({
        TaskStatus.COMPLETED, TaskStatus.UNVERIFIED, TaskStatus.FAILED}),
}


def can_transition(current: TaskStatus, new: TaskStatus) -> bool:
    """True when ``current -> new`` is a legal lifecycle step (terminal states have none)."""
    return new in _TRANSITIONS.get(current, frozenset())


class Verdict(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"


@dataclass(frozen=True)
class Verification:
    """Outcome of checking a result against observed state, with the evidence used."""

    verdict: Verdict
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, Verdict):
            raise ContractError("verdict must be a Verdict")
        if not isinstance(self.evidence, Mapping):
            raise ContractError("evidence must be a mapping")
        if self.verdict is Verdict.VERIFIED and not self.evidence:
            raise ContractError("a verified result must name its evidence")


def _check_name(kind: str, value: Any) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ContractError(
            f"{kind} must be lower-case letters, digits or underscores, starting with a letter "
            f"(at most 48 characters): {value!r}")


@dataclass(frozen=True)
class Operation:
    """One thing a feature can do.

    The callables are supplied by the feature and are only ever called by the
    platform pipeline, never directly:

    * ``plan``     turns a request into a validated, structured plan (no side effects);
    * ``preview``  describes what the plan would change, for a person to review;
    * ``execute``  carries out exactly the approved plan;
    * ``verify``   checks the result against observed state and returns a ``Verification``.
    """

    feature: str
    name: str
    effect: Effect
    permission: str
    risk: float
    approval: ApprovalPolicy
    description: str = ""
    plan: Optional[Callable[..., Any]] = None
    preview: Optional[Callable[..., Any]] = None
    execute: Optional[Callable[..., Any]] = None
    verify: Optional[Callable[..., Verification]] = None
    verify_waived_reason: Optional[str] = None
    # Permission a *different* person needs to approve a run (required when approval.second_approver).
    approver_permission: Optional[str] = None
    # Permission needed to *plan* (submit) when it differs from the one needed to run.
    plan_permission: Optional[str] = None
    # Decides the approval policy of one plan (for example "a delete needs a second person").
    # The result must include everything ``approval`` demands; it may only add conditions.
    approval_for: Optional[Callable[..., ApprovalPolicy]] = None
    # The plan and preview hold business data. They are erased from the task store when the
    # task ends (only a hash of the plan is kept).
    sensitive: bool = False
    # Cap on tasks of this operation that are waiting at the same time (None: the pipeline's cap).
    max_active: Optional[int] = None

    def __post_init__(self) -> None:
        _check_name("feature", self.feature)
        _check_name("operation name", self.name)
        if not isinstance(self.effect, Effect):
            raise ContractError("effect must be an Effect")
        if not isinstance(self.permission, str) or not self.permission.strip():
            raise ContractError("permission must be non-empty text")
        if isinstance(self.risk, bool) or not isinstance(self.risk, (int, float)) \
                or not 0.0 <= float(self.risk) <= 1.0:
            raise ContractError("risk must be a number between 0 and 1")
        if not isinstance(self.approval, ApprovalPolicy):
            raise ContractError("approval must be an ApprovalPolicy")
        for field_name in ("plan", "preview", "execute", "verify", "approval_for"):
            value = getattr(self, field_name)
            if value is not None and not callable(value):
                raise ContractError(f"{field_name} must be callable")
        if self.verify_waived_reason is not None and not self.verify_waived_reason.strip():
            raise ContractError("verify_waived_reason must not be blank")
        if self.verify is not None and self.verify_waived_reason is not None:
            raise ContractError("an operation with a verifier cannot also waive verification")
        if self.execute is None:
            raise ContractError("execute is required")
        if self.approval.second_approver and not (
                isinstance(self.approver_permission, str) and self.approver_permission.strip()):
            raise ContractError(f"{self.id}: second_approver needs an approver_permission")
        if self.approver_permission is not None and not self.approval.second_approver \
                and self.approval_for is None:
            raise ContractError(f"{self.id}: approver_permission without second_approver")
        if self.plan_permission is not None and not self.plan_permission.strip():
            raise ContractError("plan_permission must not be blank")
        if self.max_active is not None and (
                isinstance(self.max_active, bool) or not isinstance(self.max_active, int)
                or self.max_active < 1):
            raise ContractError("max_active must be a positive integer")
        if self.effect.changes_world:
            if self.preview is None:
                raise ContractError(
                    f"{self.id}: an operation that changes the world must offer a preview")
            if self.verify is None and self.verify_waived_reason is None:
                raise ContractError(
                    f"{self.id}: verify or an explicit verify_waived_reason is required")

    @property
    def id(self) -> str:
        return f"{self.feature}.{self.name}"


@dataclass(frozen=True)
class Feature:
    """A named group of operations, for example ``database``."""

    name: str
    description: str
    operations: tuple[Operation, ...]

    def __post_init__(self) -> None:
        _check_name("feature", self.name)
        ops = tuple(self.operations)
        object.__setattr__(self, "operations", ops)
        if not ops:
            raise ContractError("a feature must declare at least one operation")
        seen: set[str] = set()
        for op in ops:
            if not isinstance(op, Operation):
                raise ContractError("operations must be Operation instances")
            if op.feature != self.name:
                raise ContractError(f"{op.id} does not belong to feature {self.name!r}")
            if op.id in seen:
                raise ContractError(f"duplicate operation: {op.id}")
            seen.add(op.id)


__all__ = [
    "ApprovalPolicy", "ContractError", "Effect", "Feature", "Operation", "PlanUnsupported", "TERMINAL_STATUSES",
    "TaskStatus", "Verdict", "Verification", "can_transition",
]
