from __future__ import annotations

"""Question -> report service: LLM planner + job runner behind one call.

The service owns no credentials and makes no authorization decision; the caller
(API route) authenticates the principal and passes a trusted execution context.
Every run uses a server-generated idempotency key, so a client cannot collide
with or replay another run's workflow.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Mapping

MAX_QUESTION_LENGTH = 1000


@dataclass(frozen=True)
class AskOutcome:
    status: str  # "succeeded" | "unsupported" | "failed"
    question: str
    reason: str | None = None
    workflow_id: str | None = None
    error_code: str | None = None
    failed_task_key: str | None = None
    artifact: Mapping[str, Any] | None = None
    plan: Mapping[str, Any] | None = None
    provider: str | None = None
    attempts: int = 0
    rows: tuple[Mapping[str, Any], ...] = ()


class TableReportService:
    def __init__(self, *, planner: Any, runner: Any, experience: Any = None) -> None:
        self.planner = planner
        self.runner = runner
        self.experience = experience  # optional ConnectionExperience (best effort)

    def ask(self, question: str, *, execution_context: Any) -> AskOutcome:
        question = (question or "").strip()
        if not question or len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(f"question must be 1-{MAX_QUESTION_LENGTH} characters")
        plan = self.planner.plan(question)
        if not plan.ok:
            return AskOutcome("unsupported", question, reason=plan.reason,
                              provider=plan.provider, attempts=plan.attempts)
        req = plan.request
        summary = {
            "table": req.table, "group_by": list(req.group_by),
            "metrics": [{"output": o, "operation": op, "column": c} for o, op, c in req.metrics],
            "filters": dict(req.filters), "format": req.report_format,
        }
        outcome = self.runner.run(
            goal=question, idempotency_key=f"ask-{uuid.uuid4().hex}",
            request=req, execution_context=execution_context)
        status = "succeeded" if outcome.succeeded else "failed"
        self._remember(question, summary, status, outcome.error_code)
        return AskOutcome(
            status, question,
            reason=outcome.message, workflow_id=outcome.workflow_id,
            error_code=outcome.error_code, failed_task_key=outcome.failed_task_key,
            artifact=outcome.artifact, plan=summary, provider=plan.provider,
            attempts=plan.attempts, rows=outcome.rows)

    def _remember(self, question: str, plan: Mapping[str, Any], status: str,
                  error_code: str | None) -> None:
        # Learning must never break or delay an answer the user already earned.
        if self.experience is None:
            return
        try:
            self.experience.record(question, plan, status, error_code)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning("Could not record job experience.", exc_info=True)


__all__ = ["AskOutcome", "TableReportService", "MAX_QUESTION_LENGTH"]
