from __future__ import annotations

"""Short-lived store for report plans awaiting the user's go-ahead.

A plan is created from a question (no data is read), shown to the user, and run
at most once. Plans live only in memory: after a restart or expiry the user simply
asks again. The server keeps the plan; the client only ever sends its id, so a
plan cannot be altered between review and execution.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class PlanNotFound(LookupError):
    """Unknown, expired, cancelled, or owned by someone else (indistinguishable on purpose)."""


class PlanConflict(RuntimeError):
    """The plan cannot be run now (already running or already executed)."""


@dataclass
class StoredPlan:
    plan_id: str
    owner: str
    database: str | None
    question: str
    plan_result: Any
    summary: dict[str, Any]
    service: Any
    expires_at: float
    state: str = "planned"  # planned | running | done | unknown
    run_key: str | None = None
    result: dict[str, Any] | None = field(default=None, repr=False)


class PlanStore:
    def __init__(self, *, ttl_seconds: float = 900, max_plans: int = 200,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl, self.max_plans, self._clock = ttl_seconds, max_plans, clock
        self._plans: dict[str, StoredPlan] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = self._clock()
        for key in [k for k, p in self._plans.items() if p.expires_at <= now]:
            del self._plans[key]

    def create(self, *, owner: str, database: str | None, question: str, plan_result: Any,
               summary: dict[str, Any], service: Any) -> StoredPlan:
        with self._lock:
            self._purge()
            if len(self._plans) >= self.max_plans:
                raise PlanConflict("too many pending plans; try again shortly")
            plan = StoredPlan(uuid.uuid4().hex, owner, database, question, plan_result,
                              summary, service, self._clock() + self.ttl)
            self._plans[plan.plan_id] = plan
            return plan

    def _get(self, plan_id: str, owner: str) -> StoredPlan:
        self._purge()
        plan = self._plans.get(plan_id)
        if plan is None or plan.owner != owner:
            raise PlanNotFound("plan not found or expired")
        return plan

    def seconds_left(self, plan: StoredPlan) -> int:
        return max(0, int(plan.expires_at - self._clock()))

    def begin_run(self, plan_id: str, owner: str, key: str | None) -> tuple[str, StoredPlan]:
        """Return ("run", plan) to execute, or ("replay", plan) to answer from the stored result."""
        with self._lock:
            plan = self._get(plan_id, owner)
            if plan.state == "planned":
                plan.state, plan.run_key = "running", key
                return "run", plan
            if plan.state == "running":
                raise PlanConflict("this plan is already running")
            if plan.state == "done" and key is not None and key == plan.run_key:
                return "replay", plan
            raise PlanConflict("this plan was already executed; ask again to run a new report")

    def finish(self, plan: StoredPlan, result: dict[str, Any]) -> None:
        with self._lock:
            plan.state, plan.result = "done", result
            plan.expires_at = self._clock() + self.ttl  # keep the result for idempotent replay

    def mark_unknown(self, plan: StoredPlan) -> None:
        with self._lock:
            plan.state = "unknown"

    def release(self, plan: StoredPlan) -> None:
        """Execution never started (for example the connection changed): allow a retry."""
        with self._lock:
            if plan.state == "running":
                plan.state, plan.run_key = "planned", None

    def cancel(self, plan_id: str, owner: str) -> None:
        with self._lock:
            plan = self._get(plan_id, owner)
            if plan.state == "running":
                raise PlanConflict("this plan is running")
            del self._plans[plan.plan_id]


__all__ = ["PlanConflict", "PlanNotFound", "PlanStore", "StoredPlan"]
