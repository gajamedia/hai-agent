"""The one path every operation run goes through.

    submit  -> authorize, check policy, plan, preview          (no side effects)
    approve -> a different person, when the plan's approval policy asks for it
    run     -> re-authorize, fresh login, exact phrase, execute exactly the stored plan, verify
    cancel  -> owner withdraws a task that has not started

Rules this module enforces for every feature:

* fail closed: an unknown or disabled operation, a missing permission, a failing
  authorizer, an unwritable audit record, or an invalid plan all stop the run;
* the server keeps the plan; callers only ever send the task id, so nothing can
  change between the preview a person reviewed and the execution;
* the approval policy of a plan is decided when it is planned and stored with the task;
  it can only be stricter than the operation's declared baseline;
* a task runs at most once (compare-and-set on its version), and a finished task is only
  read again with the very idempotency key it ran under;
* a result is ``completed`` only when the verifier confirmed it (or the operation declared,
  in its contract, that it cannot be verified);
* every step is audited in the same transaction as the status change it records, and the
  plan and preview of an operation marked ``sensitive`` are erased when its task ends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

from .contracts import (
    ApprovalPolicy, Operation, PlanUnsupported, TaskStatus, Verdict, Verification,
)
from .registry import OperationRegistry, RegistryError
from .store import SQLiteTaskStore, TaskConflict, TaskNotFound, TaskRecord

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Base class for pipeline refusals."""


class PermissionDenied(PipelineError):
    pass


class PolicyViolation(PipelineError):
    pass


class PlanError(PipelineError):
    pass


class TooManyTasks(PolicyViolation):
    pass


class ApprovalRequired(PipelineError):
    pass


class ConfirmationMismatch(PipelineError):
    pass


class StepUpRequired(PipelineError):
    pass


class TaskExpired(PipelineError):
    pass


class ExecutionFailed(PipelineError):
    """The operation raised while executing; the task is recorded as failed."""


@dataclass(frozen=True)
class RunContext:
    """What an operation callable may know about the run. No credentials."""

    principal: Any
    actor_id: str
    operation_id: str
    task_id: Optional[str] = None
    approved_by: Optional[str] = None


@dataclass(frozen=True)
class RunOutcome:
    task: TaskRecord
    replayed: bool = False


Authorizer = Callable[[Any, str], Any]
Setting = Union[bool, float, Callable[[], Any]]


def _plan_hash(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _json_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanError(f"{what} must be a mapping")
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise PlanError(f"{what} must be JSON serializable: {exc}") from exc
    return dict(value)


class PlatformPipeline:
    def __init__(self, *, registry: OperationRegistry, store: SQLiteTaskStore, authorize: Authorizer,
                 clock: Callable[[], float] = time.time, ttl_seconds: float = 900,
                 approval_risk_threshold: float = 0.5, allow_self_approval: Setting = False,
                 step_up_max_age_seconds: Setting = 300,
                 max_active_per_operation: Optional[int] = None) -> None:
        """``allow_self_approval`` and ``step_up_max_age_seconds`` may be callables, read at the
        moment they are needed, so a server setting changed while it runs takes effect."""
        if not callable(authorize):
            raise TypeError("authorize must be callable")
        self.registry, self.store, self._authorize = registry, store, authorize
        self._clock, self.ttl = clock, ttl_seconds
        self.approval_risk_threshold = approval_risk_threshold
        self.allow_self_approval = allow_self_approval
        self.step_up_max_age = step_up_max_age_seconds
        self.max_active_per_operation = max_active_per_operation

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _setting(value: Setting) -> Any:
        return value() if callable(value) else value

    @staticmethod
    def _actor(principal: Any) -> str:
        actor = getattr(principal, "actor_id", None)
        if not isinstance(actor, str) or not actor:
            raise PermissionDenied("the caller has no identity")
        return actor

    def _refuse(self, event: str, *, operation_id: Optional[str], task_id: Optional[str] = None,
                actor: Optional[str] = None, **detail: Any) -> None:
        """Audit a refusal. The refusal stands even if the audit record cannot be written."""
        try:
            self.store.audit(event, task_id=task_id, operation_id=operation_id, actor=actor,
                             detail=detail)
        except Exception:  # noqa: BLE001
            logger.exception("Could not audit refusal %s", event)

    def _require(self, principal: Any, permission: str, *, operation_id: str,
                 task_id: Optional[str] = None) -> str:
        actor = self._actor(principal)
        try:
            self._authorize(principal, permission)
        except Exception as exc:  # noqa: BLE001  any failure of the check is a denial
            self._refuse("permission_denied", operation_id=operation_id, task_id=task_id,
                         actor=actor, permission=permission)
            raise PermissionDenied(f"missing permission: {permission}") from exc
        return actor

    def _operation(self, operation_id: str, *, actor: Optional[str] = None,
                   task_id: Optional[str] = None) -> Operation:
        try:
            return self.registry.get(operation_id)
        except RegistryError as exc:
            self._refuse("operation_refused", operation_id=operation_id, task_id=task_id,
                         actor=actor, reason=str(exc))
            raise

    @staticmethod
    def _context(principal: Any, actor: str, op: Operation, task_id: Optional[str] = None,
                 approved_by: Optional[str] = None) -> RunContext:
        return RunContext(principal=principal, actor_id=actor, operation_id=op.id, task_id=task_id,
                          approved_by=approved_by)

    @staticmethod
    def _policy(task: TaskRecord) -> ApprovalPolicy:
        """The approval policy this task was planned under (never recomputed later)."""
        return ApprovalPolicy.from_dict(task.approval)

    def _move(self, task: TaskRecord, op: Optional[Operation], new_status: TaskStatus, *,
              actor: Optional[str], event: str, detail: Optional[Mapping[str, Any]] = None,
              **fields: Any) -> TaskRecord:
        """One status change, its audit record and (for sensitive operations that end) the erasure of
        the plan, all in one transaction."""
        scrub = bool(op is not None and op.sensitive and new_status.terminal)
        return self.store.transition(
            task.task_id, expected_version=task.version, new_status=new_status, actor=actor,
            audit_event=event, audit_detail=detail or {}, scrub=scrub, **fields)

    def seconds_left(self, task: TaskRecord) -> int:
        return max(0, int(task.expires_at - self._clock()))

    def _sweep(self) -> None:
        """End tasks that waited too long, so their data does not stay in the store."""
        try:
            for task in self.store.list_expired_waiting():
                try:
                    op = self.registry.lookup(task.operation_id)
                except RegistryError:
                    op = None
                self._move(task, op, TaskStatus.CANCELLED, actor="system", event="task_expired")
        except Exception:  # noqa: BLE001
            logger.exception("Could not sweep expired tasks")

    # -- submit ------------------------------------------------------------

    def submit(self, operation_id: str, request: Mapping[str, Any], principal: Any, *,
               idempotency_key: Optional[str] = None) -> TaskRecord:
        actor = self._actor(principal)
        op = self._operation(operation_id, actor=actor)
        self._require(principal, op.plan_permission or op.permission, operation_id=op.id)
        risky = op.effect.changes_world and op.risk >= self.approval_risk_threshold
        if risky and not op.approval.required and op.approval_for is None:
            self._refuse("policy_violation", operation_id=op.id, actor=actor,
                         reason="risky operation declares no approval")
            raise PolicyViolation(
                f"{op.id}: risk {op.risk} needs an approval policy (threshold "
                f"{self.approval_risk_threshold})")
        if idempotency_key is not None:
            key = idempotency_key.strip()
            if not key or len(key) > 255:
                raise PolicyViolation("idempotency key must contain 1 to 255 non-blank characters")
            existing = self.store.find_by_run_key(actor, op.id, key)
            if existing is not None:
                return existing
            idempotency_key = key

        self._sweep()
        cap = op.max_active if op.max_active is not None else self.max_active_per_operation
        if cap is not None and self.store.count_active(op.id) >= cap:
            self._refuse("too_many_tasks", operation_id=op.id, actor=actor)
            raise TooManyTasks("too many pending tasks; try again shortly")

        ctx = self._context(principal, actor, op)
        try:
            raw_plan = op.plan(request, ctx) if op.plan is not None else request
            plan = _json_mapping(raw_plan, "plan")
            policy = op.approval_for(plan, ctx) if op.approval_for is not None else op.approval
            if not isinstance(policy, ApprovalPolicy) or not policy.covers(op.approval):
                raise PolicyViolation(f"{op.id}: the approval policy of a plan may only add conditions")
            if policy.second_approver and not op.approver_permission:
                raise PolicyViolation(f"{op.id}: a second approver needs an approver_permission")
            if risky and not policy.required:
                raise PolicyViolation(f"{op.id}: risk {op.risk} needs an approval policy")
            preview = confirm_phrase = None
            if op.preview is not None:
                preview = _json_mapping(op.preview(plan, ctx), "preview")
            if policy.confirm_phrase:
                phrase = (preview or {}).get("confirm_phrase")
                if not isinstance(phrase, str) or not phrase.strip():
                    raise PlanError("the preview must give the confirm_phrase this operation requires")
                confirm_phrase = phrase.strip()
        except PlanUnsupported as exc:
            self._refuse("plan_unsupported", operation_id=op.id, actor=actor, reason=exc.reason)
            raise
        except PipelineError as exc:
            self._refuse("plan_failed", operation_id=op.id, actor=actor, reason=str(exc))
            raise
        except Exception as exc:  # noqa: BLE001
            self._refuse("plan_failed", operation_id=op.id, actor=actor, error=type(exc).__name__)
            raise PlanError(str(exc)) from exc

        status = TaskStatus.AWAITING_APPROVAL if policy.second_approver else TaskStatus.APPROVED
        return self.store.create(
            operation_id=op.id, owner=actor, status=status, plan=plan, preview=preview,
            confirm_phrase=confirm_phrase, approval=policy.as_dict(), run_key=idempotency_key,
            ttl_seconds=self.ttl, audit_event="task_planned",
            audit_detail={"effect": op.effect.value, "risk": op.risk, "plan_sha256": _plan_hash(plan),
                          "approval": policy.as_dict()})

    # -- visibility --------------------------------------------------------

    def _can_approve(self, principal: Any, op: Operation, task: TaskRecord) -> bool:
        if not self._policy(task).second_approver or op.approver_permission is None:
            return False
        try:
            self._authorize(principal, op.approver_permission)
        except Exception:  # noqa: BLE001
            return False
        return True

    def get_task(self, task_id: str, principal: Any) -> TaskRecord:
        """The owner, or someone allowed to approve it, may look; anyone else sees 'not found'."""
        actor = self._actor(principal)
        task = self.store.get(task_id)
        if task.owner == actor:
            return task
        try:
            op = self.registry.lookup(task.operation_id)
        except RegistryError:
            raise TaskNotFound("task not found") from None
        if self._can_approve(principal, op, task):
            return task
        raise TaskNotFound("task not found")

    def waiting_for_approval(self, principal: Any) -> list[TaskRecord]:
        actor = self._actor(principal)
        include_own = bool(self._setting(self.allow_self_approval))
        found = []
        for task in self.store.list_by_status([TaskStatus.AWAITING_APPROVAL]):
            if task.expires_at <= self._clock():
                continue
            if task.owner == actor and not include_own:
                continue
            try:
                op = self.registry.lookup(task.operation_id)
            except RegistryError:
                continue
            if self._can_approve(principal, op, task):
                found.append(task)
        return found

    # -- approve / cancel --------------------------------------------------

    def _expire_if_due(self, task: TaskRecord, op: Operation, actor: str) -> None:
        if task.expires_at <= self._clock() and task.status in (
                TaskStatus.PLANNED, TaskStatus.AWAITING_APPROVAL, TaskStatus.APPROVED):
            self._move(task, op, TaskStatus.CANCELLED, actor=actor, event="task_expired")
            raise TaskExpired("this task expired; plan it again")

    def _require_fresh_login(self, principal: Any, op: Operation, actor: str,
                             task_id: str) -> None:
        auth_time = getattr(principal, "auth_time", None)
        window = self._setting(self.step_up_max_age)
        if auth_time is None or not (0 <= self._clock() - auth_time <= window):
            self._refuse("step_up_required", operation_id=op.id, task_id=task_id, actor=actor)
            raise StepUpRequired("sign in again to confirm it is you")

    def approve(self, task_id: str, principal: Any, *, reject: bool = False) -> TaskRecord:
        actor = self._actor(principal)
        task = self.store.get(task_id)
        op = self._operation(task.operation_id, actor=actor, task_id=task_id)
        policy = self._policy(task)
        if not policy.second_approver or op.approver_permission is None:
            raise TaskConflict("this operation does not use a second approver")
        self._require(principal, op.approver_permission, operation_id=op.id, task_id=task_id)
        if not reject and policy.fresh_login:
            self._require_fresh_login(principal, op, actor, task_id)
        if task.status is not TaskStatus.AWAITING_APPROVAL:
            raise TaskConflict("this task is not waiting for approval")
        self._expire_if_due(task, op, actor)
        self_approved = task.owner == actor
        if self_approved and not self._setting(self.allow_self_approval):
            self._refuse("self_approval_refused", operation_id=op.id, task_id=task_id, actor=actor)
            raise PolicyViolation("you cannot approve your own task")
        return self._move(
            task, op, TaskStatus.REJECTED if reject else TaskStatus.APPROVED, actor=actor,
            event="task_rejected" if reject else "task_approved",
            detail={"self_approved": self_approved}, approved_by=actor)

    def cancel(self, task_id: str, principal: Any) -> TaskRecord:
        actor = self._actor(principal)
        task = self.store.get(task_id)
        if task.owner != actor:
            self._refuse("cancel_refused", operation_id=task.operation_id, task_id=task_id, actor=actor)
            raise TaskNotFound("task not found")
        try:
            op = self.registry.lookup(task.operation_id)
        except RegistryError:
            op = None
        return self._move(task, op, TaskStatus.CANCELLED, actor=actor, event="task_cancelled")

    # -- run ---------------------------------------------------------------

    def run(self, task_id: str, principal: Any, *, confirm: Optional[str] = None,
            idempotency_key: Optional[str] = None) -> RunOutcome:
        actor = self._actor(principal)
        exec_key = None
        if idempotency_key is not None:
            exec_key = idempotency_key.strip()
            if not exec_key or len(exec_key) > 255:
                raise PolicyViolation("idempotency key must contain 1 to 255 non-blank characters")
        task = self.store.get(task_id)
        if task.owner != actor:
            self._refuse("run_refused", operation_id=task.operation_id, task_id=task_id, actor=actor,
                         reason="not the owner")
            raise TaskNotFound("task not found")
        if task.result is not None and task.status in (
                TaskStatus.COMPLETED, TaskStatus.UNVERIFIED, TaskStatus.FAILED):
            # The operation ran and its result is stored (a failed *verification* keeps it; an
            # operation that raised has none). Only the same key may read it again; anything else
            # is a second run.
            if exec_key is not None and exec_key == task.exec_key:
                return RunOutcome(task, replayed=True)
            raise TaskConflict("this task was already executed; plan it again to run it anew")
        op = self._operation(task.operation_id, actor=actor, task_id=task_id)
        self._require(principal, op.permission, operation_id=op.id, task_id=task_id)
        policy = self._policy(task)
        if task.status is TaskStatus.AWAITING_APPROVAL:
            raise ApprovalRequired("this task needs approval from someone else first")
        if task.status is not TaskStatus.APPROVED:
            raise TaskConflict(f"this task cannot run from state {task.status.value}")
        self._expire_if_due(task, op, actor)

        if policy.fresh_login:
            self._require_fresh_login(principal, op, actor, task_id)
        if policy.confirm_phrase and (
                not isinstance(confirm, str) or confirm.strip() != task.confirm_phrase):
            self._refuse("confirmation_mismatch", operation_id=op.id, task_id=task_id, actor=actor)
            raise ConfirmationMismatch("the confirmation phrase does not match")

        # Compare-and-set: only one caller can move the task to EXECUTING.
        task = self._move(task, op, TaskStatus.EXECUTING, actor=actor, event="task_executing",
                          exec_key=exec_key)
        ctx = self._context(principal, actor, op, task_id, task.approved_by)
        try:
            raw = op.execute(task.plan, ctx)
            result = _json_mapping(raw if isinstance(raw, Mapping) else {"value": raw}, "result")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Operation %s failed", op.id)
            self._move(
                task, op, TaskStatus.FAILED, actor=actor, event="task_failed",
                detail={"error": type(exc).__name__},
                error=f"{type(exc).__name__}: the operation raised; if it changes data, check the "
                      f"target to see whether the change happened")
            raise ExecutionFailed(f"{op.id} failed") from exc
        task = self._move(task, op, TaskStatus.VERIFYING, actor=actor, event="task_executed",
                          result=result)
        return RunOutcome(self._verify(op, task, ctx, actor))

    def _verify(self, op: Operation, task: TaskRecord, ctx: RunContext, actor: str) -> TaskRecord:
        if op.verify is None:
            reason = op.verify_waived_reason
            verification = {"verdict": "waived", "reason": reason} if reason \
                else {"verdict": "not_applicable"}
            return self._move(task, op, TaskStatus.COMPLETED, actor=actor, event="task_completed",
                              detail=verification, verification=verification)
        try:
            outcome = op.verify(task.plan, task.result, ctx)
            if not isinstance(outcome, Verification):
                raise TypeError("verify must return a Verification")
            record = {"verdict": outcome.verdict.value,
                      "evidence": _json_mapping(outcome.evidence, "evidence")}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Verifier of %s failed", op.id)
            record = {"verdict": Verdict.UNVERIFIED.value,
                      "evidence": {"verifier_error": type(exc).__name__}}
        status = {Verdict.VERIFIED.value: TaskStatus.COMPLETED,
                  Verdict.UNVERIFIED.value: TaskStatus.UNVERIFIED,
                  Verdict.FAILED.value: TaskStatus.FAILED}[record["verdict"]]
        return self._move(
            task, op, status, actor=actor, event=f"task_{status.value}",
            detail={"verdict": record["verdict"]}, verification=record,
            error="verification failed" if status is TaskStatus.FAILED else None)

    # -- restart recovery --------------------------------------------------

    def recover_interrupted(self) -> list[TaskRecord]:
        """Call once at startup. Nothing is re-run: a task that was executing when the process
        stopped is marked failed (outcome unknown), one that was being verified is unverified."""
        recovered = []
        for task in self.store.list_by_status([TaskStatus.EXECUTING, TaskStatus.VERIFYING]):
            try:
                op = self.registry.lookup(task.operation_id)
            except RegistryError:
                op = None
            if task.status is TaskStatus.EXECUTING:
                new, error = TaskStatus.FAILED, \
                    "interrupted by a restart; the outcome is unknown, check the target"
            else:
                new, error = TaskStatus.UNVERIFIED, "interrupted by a restart before verification"
            recovered.append(self._move(task, op, new, actor="system", event="task_recovered",
                                        error=error))
        self._sweep()
        return recovered


__all__ = [
    "ApprovalRequired", "ConfirmationMismatch", "ExecutionFailed", "PermissionDenied", "PipelineError",
    "PlanError", "PlatformPipeline", "PolicyViolation", "RunContext", "RunOutcome", "StepUpRequired",
    "TaskExpired", "TooManyTasks",
]
