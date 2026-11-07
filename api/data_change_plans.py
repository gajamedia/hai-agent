from __future__ import annotations

"""Short-lived store for data-change plans waiting for a confirmation (and maybe an approval).

States: planned -> (approved) -> running -> done. A plan can only be run by its owner, once, after the
exact confirmation phrase, and, when the rules ask for it, only after someone *else* approved it.
The server keeps the change; the client only ever sends the plan id, so nothing can be altered between
the preview the user looked at and the execution.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class ChangePlanNotFound(LookupError):
    """Unknown, expired, cancelled, or not visible to this person (indistinguishable on purpose)."""


class ChangePlanConflict(RuntimeError):
    """The plan is not in a state that allows this action."""


@dataclass
class StoredChange:
    plan_id: str
    owner: str
    database: str
    question: str
    request: Any
    expected_rows: int
    summary: dict[str, Any]
    preview: dict[str, Any]
    confirm_phrase: str
    requires_approval: bool
    service: Any
    expires_at: float
    state: str = "planned"        # planned | approved | rejected | running | done | unknown
    approved_by: str | None = None
    run_key: str | None = None
    result: dict[str, Any] | None = field(default=None, repr=False)


class ChangePlanStore:
    def __init__(self, *, ttl_seconds: float = 900, max_plans: int = 100,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl, self.max_plans, self._clock = ttl_seconds, max_plans, clock
        self._plans: dict[str, StoredChange] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = self._clock()
        for key in [k for k, p in self._plans.items() if p.expires_at <= now and p.state != "running"]:
            del self._plans[key]

    def create(self, **fields: Any) -> StoredChange:
        with self._lock:
            self._purge()
            if len(self._plans) >= self.max_plans:
                raise ChangePlanConflict("too many pending changes; try again shortly")
            plan = StoredChange(plan_id=uuid.uuid4().hex, expires_at=self._clock() + self.ttl, **fields)
            self._plans[plan.plan_id] = plan
            return plan

    def seconds_left(self, plan: StoredChange) -> int:
        return max(0, int(plan.expires_at - self._clock()))

    def _owned(self, plan_id: str, owner: str) -> StoredChange:
        self._purge()
        plan = self._plans.get(plan_id)
        if plan is None or plan.owner != owner:
            raise ChangePlanNotFound("change plan not found or expired")
        return plan

    def get(self, plan_id: str, owner: str) -> StoredChange:
        with self._lock:
            return self._owned(plan_id, owner)

    def waiting_for_approval(self, viewer: str) -> list[StoredChange]:
        """Plans someone else made that still need a second person (never your own)."""
        with self._lock:
            self._purge()
            return [p for p in self._plans.values()
                    if p.state == "planned" and p.requires_approval and p.owner != viewer]

    def approve(self, plan_id: str, approver: str, *, reject: bool = False) -> StoredChange:
        with self._lock:
            self._purge()
            plan = self._plans.get(plan_id)
            if plan is None:
                raise ChangePlanNotFound("change plan not found or expired")
            if plan.owner == approver:
                raise ChangePlanConflict("you cannot approve your own change")
            if not plan.requires_approval or plan.state != "planned":
                raise ChangePlanConflict("this change is not waiting for approval")
            plan.state = "rejected" if reject else "approved"
            plan.approved_by = approver
            return plan

    def begin_run(self, plan_id: str, owner: str, key: str | None) -> tuple[str, StoredChange]:
        with self._lock:
            plan = self._owned(plan_id, owner)
            if plan.state == "done" and key is not None and key == plan.run_key:
                return "replay", plan
            if plan.state == "running":
                raise ChangePlanConflict("this change is already running")
            if plan.state == "rejected":
                raise ChangePlanConflict("this change was rejected")
            if plan.requires_approval and plan.state != "approved":
                raise ChangePlanConflict("this change needs approval from someone else first")
            if plan.state not in {"planned", "approved"}:
                raise ChangePlanConflict("this change was already executed; plan it again")
            plan.state, plan.run_key = "running", key
            return "run", plan

    def finish(self, plan: StoredChange, result: dict[str, Any]) -> None:
        with self._lock:
            plan.state, plan.result = "done", result
            plan.expires_at = self._clock() + self.ttl

    def mark_unknown(self, plan: StoredChange) -> None:
        with self._lock:
            plan.state = "unknown"

    def release(self, plan: StoredChange, state: str = "planned") -> None:
        """Nothing was executed (refused or rolled back): the owner may try again."""
        with self._lock:
            if plan.state == "running":
                plan.state, plan.run_key = ("approved" if plan.approved_by else state), None

    def cancel(self, plan_id: str, owner: str) -> None:
        with self._lock:
            plan = self._owned(plan_id, owner)
            if plan.state == "running":
                raise ChangePlanConflict("this change is running")
            del self._plans[plan.plan_id]


__all__ = ["ChangePlanConflict", "ChangePlanNotFound", "ChangePlanStore", "StoredChange"]
