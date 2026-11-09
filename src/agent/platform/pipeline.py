"""The one path every operation run goes through.

    submit  -> authorize, check policy, plan, preview          (no side effects)
    approve -> a different person, when the operation asks for it
    run     -> re-authorize, fresh login, exact phrase, execute exactly the stored plan, verify
    cancel  -> owner withdraws a task that has not started

Rules this module enforces for every feature:

* fail closed: an unknown or disabled operation, a missing permission, a failing
  authorizer, an unwritable audit record, or an invalid plan all stop the run;
* the server keeps the plan; callers only ever send the task id, so nothing can
  change between the preview a person reviewed and the execution;
* a task runs at most once (compare-and-set on its version), and a replay of a
  finished task returns the stored outcome instead of running again;
* a result is ``completed`` only when the verifier confirmed it (or the
  operation declared, in its contract, that it cannot be verified);
* every step is audited in the same transaction as the status change it records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .contracts import Operation, TaskStatus, Verdict, Verification
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


@dataclass(frozen=True)
class RunOutcome:
    task: TaskRecord
    replayed: bool = False


Authorizer = Callable[[Any, str], Any]


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
                 approval_risk_threshold: float = 0.5, allow_self_approval: bool = False,
                 step_up_max_age_seconds: float = 300) -> None:
        if not callable(authorize):
            raise TypeError("authorize must be callable")
        self.registry, self.store, self._authorize = registry, store, authorize
        self._clock, self.ttl = clock, ttl_seconds
        self.approval_risk_threshold = approval_risk_threshold
        self.allow_self_approval = allow_self_approval
        self.step_up_max_age = step_up_max_age_seconds

    # -- helpers -----------------------------------------------------------

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

    def _context(self, principal: Any, actor: str, op: Operation,
                 task_id: Optional[str] = None) -> RunContext:
        return RunContext(principal=principal, actor_id=actor, operation_id=op.id, task_id=task_id)

    # -- submit ------------------------------------------------------------

    def submit(self, operation_id: str, request: Mapping[str, Any], principal: Any, *,
               idempotency_key: Optional[str] = None) -> TaskRecord:
        actor = self._actor(principal)
        op = self._operation(operation_id, actor=actor)
        self._require(principal, op.permission, operation_id=op.id)
        if op.effect.changes_world and op.risk >= self.approval_risk_threshold \
                and not op.approval.required:
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

        ctx = self._context(principal, actor, op)
        try:
            raw_plan = op.plan(request, ctx) if op.plan is not None else request
            plan = _json_mapping(raw_plan, "plan")
            preview = confirm_phrase = None
            if op.preview is not None:
                preview = _json_mapping(op.preview(plan, ctx), "preview")
            if op.approval.confirm_phrase:
                phrase = (preview or {}).get("confirm_phrase")
                if not isinstance(phrase, str) or not phrase.strip():
                    raise PlanError("the preview must give the confirm_phrase this operation requires")
                confirm_phrase = phrase.strip()
        except PipelineError as exc:
            self._refuse("plan_failed", operation_id=op.id, actor=actor, reason=str(exc))
            raise
        except Exception as exc:  # noqa: BLE001
            self._refuse("plan_failed", operation_id=op.id, actor=actor, error=type(exc).__name__)
            raise PlanError(str(exc)) from exc

        status = TaskStatus.AWAITING_APPROVAL if op.approval.second_approver else TaskStatus.APPROVED
        return self.store.create(
            operation_id=op.id, owner=actor, status=status, plan=plan, preview=preview,
            confirm_phrase=confirm_phrase, run_key=idempotency_key, ttl_seconds=self.ttl,
            audit_event="task_planned",
            audit_detail={"effect": op.effect.value, "risk": op.risk, "plan_sha256": _plan_hash(plan),
                          "approval": {"confirm_phrase": op.approval.confirm_phrase,
                                       "fresh_login": op.approval.fresh_login,
                                       "second_approver": op.approval.second_approver}})

    # -- visibility --------------------------------------------------------

    def _can_approve(self, principal: Any, op: Operation) -> bool:
        if not op.approval.second_approver or op.approver_permission is None:
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
            op = self.registry.get(task.operation_id)
        except RegistryError:
            raise TaskNotFound("task not found") from None
        if self._can_approve(principal, op):
            return task
        raise TaskNotFound("task not found")

    def waiting_for_approval(self, principal: Any) -> list[TaskRecord]:
        actor = self._actor(principal)
        found = []
        for task in self.store.list_by_status([TaskStatus.AWAITING_APPROVAL]):
            if task.expires_at <= self._clock():
                continue
            if task.owner == actor and not self.allow_self_approval:
                continue
            try:
                op = self.registry.get(task.operation_id)
            except RegistryError:
                continue
            if self._can_approve(principal, op):
                found.append(task)
        return found

    # -- approve / cancel --------------------------------------------------

    def _expire_if_due(self, task: TaskRecord, actor: str) -> None:
        if task.expires_at <= self._clock() and not task.status.terminal \
                and task.status in (TaskStatus.PLANNED, TaskStatus.AWAITING_APPROVAL,
                                    TaskStatus.APPROVED):
            self.store.transition(task.task_id, expected_version=task.version,
                                  new_status=TaskStatus.CANCELLED, actor=actor,
                                  audit_event="task_expired", audit_detail={})
            raise TaskExpired("this task expired; plan it again")

    def approve(self, task_id: str, principal: Any, *, reject: bool = False) -> TaskRecord:
        actor = self._actor(principal)
        task = self.store.get(task_id)
        op = self._operation(task.operation_id, actor=actor, task_id=task_id)
        if not op.approval.second_approver or op.approver_permission is None:
            raise TaskConflict("this operation does not use a second approver")
        self._require(principal, op.approver_permission, operation_id=op.id, task_id=task_id)
        if task.status is not TaskStatus.AWAITING_APPROVAL:
            raise TaskConflict("this task is not waiting for approval")
        self._expire_if_due(task, actor)
        self_approved = task.owner == actor
        if self_approved and not self.allow_self_approval:
            self._refuse("self_approval_refused", operation_id=op.id, task_id=task_id, actor=actor)
            raise PolicyViolation("you cannot approve your own task")
        return self.store.transition(
            task_id, expected_version=task.version,
            new_status=TaskStatus.REJECTED if reject else TaskStatus.APPROVED, actor=actor,
            audit_event="task_rejected" if reject else "task_approved",
            audit_detail={"self_approved": self_approved}, approved_by=actor)

    def cancel(self, task_id: str, principal: Any) -> TaskRecord:
        actor = self._actor(principal)
        task = self.store.get(task_id)
        if task.owner != actor:
            self._refuse("cancel_refused", operation_id=task.operation_id, task_id=task_id, actor=actor)
            raise TaskNotFound("task not found")
        return self.store.transition(task_id, expected_version=task.version,
                                     new_status=TaskStatus.CANCELLED, actor=actor,
                                     audit_event="task_cancelled", audit_detail={})

    # -- run ---------------------------------------------------------------

    def run(self, task_id: str, principal: Any, *, confirm: Optional[str] = None) -> RunOutcome:
        actor = self._actor(principal)
        task = self.store.get(task_id)
        if task.owner != actor:
            self._refuse("run_refused", operation_id=task.operation_id, task_id=task_id, actor=actor,
                         reason="not the owner")
            raise TaskNotFound("task not found")
        if task.status in (TaskStatus.COMPLETED, TaskStatus.UNVERIFIED):
            return RunOutcome(task, replayed=True)
        op = self._operation(task.operation_id, actor=actor, task_id=task_id)
        self._require(principal, op.permission, operation_id=op.id, task_id=task_id)
        if task.status is TaskStatus.AWAITING_APPROVAL:
            raise ApprovalRequired("this task needs approval from someone else first")
        if task.status is not TaskStatus.APPROVED:
            raise TaskConflict(f"this task cannot run from state {task.status.value}")
        self._expire_if_due(task, actor)

        if op.approval.fresh_login:
            auth_time = getattr(principal, "auth_time", None)
            now = self._clock()
            if auth_time is None or not (0 <= now - auth_time <= self.step_up_max_age):
                self._refuse("step_up_required", operation_id=op.id, task_id=task_id, actor=actor)
                raise StepUpRequired("sign in again to confirm it is you")
        if op.approval.confirm_phrase and (
                not isinstance(confirm, str) or confirm.strip() != task.confirm_phrase):
            self._refuse("confirmation_mismatch", operation_id=op.id, task_id=task_id, actor=actor)
            raise ConfirmationMismatch("the confirmation phrase does not match")

        # Compare-and-set: only one caller can move the task to EXECUTING.
        task = self.store.transition(task_id, expected_version=task.version,
                                     new_status=TaskStatus.EXECUTING, actor=actor,
                                     audit_event="task_executing", audit_detail={})
        ctx = self._context(principal, actor, op, task_id)
        try:
            raw = op.execute(task.plan, ctx)
            result = _json_mapping(raw if isinstance(raw, Mapping) else {"value": raw}, "result")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Operation %s failed", op.id)
            self.store.transition(
                task_id, expected_version=task.version, new_status=TaskStatus.FAILED, actor=actor,
                audit_event="task_failed", audit_detail={"error": type(exc).__name__},
                error=f"{type(exc).__name__}: the operation raised; if it changes data, check the "
                      f"target to see whether the change happened")
            raise ExecutionFailed(f"{op.id} failed") from exc
        task = self.store.transition(task_id, expected_version=task.version,
                                     new_status=TaskStatus.VERIFYING, actor=actor,
                                     audit_event="task_executed", audit_detail={}, result=result)
        return RunOutcome(self._verify(op, task, ctx, actor))

    def _verify(self, op: Operation, task: TaskRecord, ctx: RunContext, actor: str) -> TaskRecord:
        if op.verify is None:
            reason = op.verify_waived_reason
            verification = {"verdict": "waived", "reason": reason} if reason \
                else {"verdict": "not_applicable"}
            return self.store.transition(
                task.task_id, expected_version=task.version, new_status=TaskStatus.COMPLETED,
                actor=actor, audit_event="task_completed", audit_detail=verification,
                verification=verification)
        try:
            outcome = op.verify(task.plan, task.result, ctx)
            if not isinstance(outcome, Verification):
                raise TypeError("verify must return a Verification")
            record = {"verdict": outcome.verdict.value, "evidence": _json_mapping(outcome.evidence, "evidence")}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Verifier of %s failed", op.id)
            record = {"verdict": Verdict.UNVERIFIED.value,
                      "evidence": {"verifier_error": type(exc).__name__}}
        status = {Verdict.VERIFIED.value: TaskStatus.COMPLETED,
                  Verdict.UNVERIFIED.value: TaskStatus.UNVERIFIED,
                  Verdict.FAILED.value: TaskStatus.FAILED}[record["verdict"]]
        return self.store.transition(
            task.task_id, expected_version=task.version, new_status=status, actor=actor,
            audit_event=f"task_{status.value}", audit_detail={"verdict": record["verdict"]},
            verification=record,
            error="verification failed" if status is TaskStatus.FAILED else None)

    # -- restart recovery --------------------------------------------------

    def recover_interrupted(self) -> list[TaskRecord]:
        """Call once at startup. Nothing is re-run: a task that was executing when the process
        stopped is marked failed (outcome unknown), one that was being verified is unverified."""
        recovered = []
        for task in self.store.list_by_status([TaskStatus.EXECUTING, TaskStatus.VERIFYING]):
            if task.status is TaskStatus.EXECUTING:
                new, error = TaskStatus.FAILED, \
                    "interrupted by a restart; the outcome is unknown, check the target"
            else:
                new, error = TaskStatus.UNVERIFIED, "interrupted by a restart before verification"
            recovered.append(self.store.transition(
                task.task_id, expected_version=task.version, new_status=new, actor="system",
                audit_event="task_recovered", audit_detail={}, error=error))
        return recovered


__all__ = [
    "ApprovalRequired", "ConfirmationMismatch", "ExecutionFailed", "PermissionDenied", "PipelineError",
    "PlanError", "PlatformPipeline", "PolicyViolation", "RunContext", "RunOutcome", "StepUpRequired",
    "TaskExpired",
]
