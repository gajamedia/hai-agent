"""Independent, read-only evaluation of autonomous test outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from .contracts import (
    CaseStatus,
    GoalEvaluation,
    ROOM_STATE_SCHEMA,
    ScenarioDefinition,
    TestCaseResult,
)


class EvaluationInputError(ValueError):
    """Raised when supplied state evidence is malformed."""


class IndependentEvaluator:
    """Evaluate a case using observed state only; never trust execution claims."""

    def evaluate(
        self,
        scenario: ScenarioDefinition,
        final_state: Optional[Mapping[str, Any]],
        *,
        case_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        recovery_attempts: int = 0,
        error: Optional[str] = None,
        evidence: tuple[str, ...] = (),
    ) -> TestCaseResult:
        if not isinstance(scenario, ScenarioDefinition):
            raise TypeError("scenario must be ScenarioDefinition")

        observed = self._normalize_observation(final_state)
        goal = dict(scenario.goal)
        initial = dict(scenario.initial_state)

        mismatches = [
            key for key, expected in goal.items()
            if key in observed and observed[key] != expected
        ]
        missing = [key for key in goal if key not in observed]

        if mismatches:
            outcome: Optional[bool] = False
            status = CaseStatus.FAILED
            reason = (
                "Observed final state contradicts the goal for: "
                + ", ".join(sorted(mismatches))
            )
        elif missing:
            outcome = None
            status = CaseStatus.INCONCLUSIVE
            reason = (
                "Insufficient final-state evidence; missing goal fields: "
                + ", ".join(sorted(missing))
            )
        else:
            outcome = True
            status = CaseStatus.PASSED
            reason = "All goal fields match the independently observed final state."

        normalized_evidence = list(evidence)
        normalized_evidence.append(
            f"Observed {len(observed)} final-state field(s); evaluated {len(goal)} goal field(s)."
        )
        if error:
            normalized_evidence.append("Execution error was recorded as context, not as goal evidence.")

        evaluation = GoalEvaluation(
            goal=goal,
            initial_state=initial,
            final_state=observed,
            goal_satisfied=outcome,
            evidence=tuple(normalized_evidence),
            reason=reason,
        )
        return TestCaseResult(
            case_id=case_id or scenario.scenario_id,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            evaluation=evaluation,
            recovery_attempts=recovery_attempts,
            error=error,
            metadata={
                "evaluator": "IndependentEvaluator",
                "evaluator_version": 1,
                "decision_basis": "observed_final_state_only",
            },
        )

    @staticmethod
    def _normalize_observation(
        final_state: Optional[Mapping[str, Any]],
    ) -> dict[str, str]:
        if final_state is None:
            return {}
        if not isinstance(final_state, Mapping):
            raise EvaluationInputError("final_state must be a mapping or None")

        unknown = set(final_state) - set(ROOM_STATE_SCHEMA)
        if unknown:
            raise EvaluationInputError(
                "final_state contains unknown field(s): " + ", ".join(sorted(unknown))
            )

        normalized: dict[str, str] = {}
        for key, value in final_state.items():
            if not isinstance(value, str) or value not in ROOM_STATE_SCHEMA[key]:
                raise EvaluationInputError(
                    f"invalid observed value for {key!r}: {value!r}"
                )
            normalized[key] = value
        return normalized


def evaluate_test_case(
    scenario: ScenarioDefinition,
    final_state: Optional[Mapping[str, Any]],
    **kwargs: Any,
) -> TestCaseResult:
    """Functional convenience wrapper around IndependentEvaluator."""
    return IndependentEvaluator().evaluate(scenario, final_state, **kwargs)


__all__ = [
    "EvaluationInputError",
    "IndependentEvaluator",
    "evaluate_test_case",
]
