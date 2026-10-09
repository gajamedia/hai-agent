from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .goal_acceptance import (
    GoalAcceptanceDecision,
    GoalAcceptanceEngine,
    GoalAcceptanceResult,
    AgentGoalAcceptanceIntegrator,
)
from .goal_proposal import GoalCandidate
from .goal_evaluation import (
    GoalEvaluationEngine,
    GoalEvaluationResult,
)


# ============================================================================
# ERRORS
# ============================================================================


class GoalEvaluationAcceptanceError(Exception):
    """Base error for goal evaluation + acceptance integration."""


class GoalEvaluationAcceptanceConfigurationError(
    GoalEvaluationAcceptanceError
):
    """Raised when configuration is invalid."""


class GoalEvaluationAcceptanceInputError(
    GoalEvaluationAcceptanceError
):
    """Raised when input data is invalid."""


# ============================================================================
# POLICY
# ============================================================================


@dataclass(frozen=True)
class GoalEvaluationAcceptancePolicy:
    """
    Policy controlling the bridge between goal evaluation and goal acceptance.

    Important:
    - GoalEvaluationEngine remains responsible for evaluation.
    - GoalAcceptanceEngine remains responsible for lifecycle acceptance.
    - This class only coordinates the two stages.
    """

    minimum_evaluation_score: Optional[float] = None

    require_recommendation: bool = False

    prioritize_by_evaluation: bool = True

    max_candidates: Optional[int] = None

    def __post_init__(self) -> None:
        if self.minimum_evaluation_score is not None:
            value = float(self.minimum_evaluation_score)

            if not 0.0 <= value <= 1.0:
                raise GoalEvaluationAcceptanceConfigurationError(
                    "minimum_evaluation_score must be between 0 and 1"
                )

            object.__setattr__(
                self,
                "minimum_evaluation_score",
                value,
            )

        if self.max_candidates is not None:
            value = int(self.max_candidates)

            if value < 1:
                raise GoalEvaluationAcceptanceConfigurationError(
                    "max_candidates must be >= 1"
                )

            object.__setattr__(
                self,
                "max_candidates",
                value,
            )

        object.__setattr__(
            self,
            "require_recommendation",
            bool(self.require_recommendation),
        )

        object.__setattr__(
            self,
            "prioritize_by_evaluation",
            bool(self.prioritize_by_evaluation),
        )


# ============================================================================
# DECISION
# ============================================================================


@dataclass(frozen=True)
class GoalEvaluationAcceptanceDecision:
    """
    Immutable result of evaluating one GoalCandidate before acceptance.
    """

    evaluation: GoalEvaluationResult

    eligible: bool

    acceptance_decision: Optional[GoalAcceptanceDecision] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate(self) -> GoalCandidate:
        return self.evaluation.candidate

    @property
    def evaluation_score(self) -> float:
        return _evaluation_score(self.evaluation)

    @property
    def evaluation_recommendation(self) -> str:
        return _evaluation_recommendation(self.evaluation)

    @property
    def accepted(self) -> bool:
        if self.acceptance_decision is None:
            return False

        return bool(self.acceptance_decision.accepted)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata or {}),
        )


# ============================================================================
# RESULT
# ============================================================================


@dataclass(frozen=True)
class GoalEvaluationAcceptanceResult:
    """
    Immutable result of applying one candidate through:

        evaluation
            ↓
        evaluation gate
            ↓
        acceptance evaluation
            ↓
        lifecycle application
    """

    decision: GoalEvaluationAcceptanceDecision

    acceptance_result: Optional[GoalAcceptanceResult] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate(self) -> GoalCandidate:
        return self.decision.candidate

    @property
    def accepted(self) -> bool:
        if self.acceptance_result is None:
            return False

        return bool(self.acceptance_result.accepted)

    @property
    def goal(self) -> Any:
        if self.acceptance_result is None:
            return None

        return self.acceptance_result.goal

    @property
    def current_goal_id(self) -> Optional[str]:
        if self.acceptance_result is None:
            return None

        return self.acceptance_result.current_goal_id

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata or {}),
        )


# ============================================================================
# BATCH
# ============================================================================


@dataclass(frozen=True)
class GoalEvaluationAcceptanceBatch:
    """
    Immutable collection of evaluation/acceptance results.
    """

    results: tuple[GoalEvaluationAcceptanceResult, ...]

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def accepted(self) -> tuple[GoalEvaluationAcceptanceResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.accepted
        )

    @property
    def rejected(self) -> tuple[GoalEvaluationAcceptanceResult, ...]:
        return tuple(
            result
            for result in self.results
            if not result.accepted
        )

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def best(self) -> Optional[GoalEvaluationAcceptanceResult]:
        if not self.results:
            return None

        return self.results[0]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "results",
            tuple(self.results),
        )


# ============================================================================
# ENGINE
# ============================================================================


class GoalEvaluationAcceptanceEngine:
    """
    Coordinates GoalEvaluationEngine and GoalAcceptanceEngine.

    Responsibilities:

    1. Evaluate candidates.
    2. Apply optional evaluation gates.
    3. Rank evaluated candidates.
    4. Apply max_candidates.
    5. Delegate lifecycle acceptance to GoalAcceptanceEngine.
    6. Delegate lifecycle mutation to AgentGoalAcceptanceIntegrator.

    This class does NOT own lifecycle state.
    """

    def __init__(
        self,
        evaluation_engine: Optional[GoalEvaluationEngine] = None,
        acceptance_engine: Optional[GoalAcceptanceEngine] = None,
        policy: Optional[GoalEvaluationAcceptancePolicy] = None,
    ) -> None:

        self.evaluation_engine = (
            evaluation_engine
            if evaluation_engine is not None
            else GoalEvaluationEngine()
        )

        self.acceptance_engine = (
            acceptance_engine
            if acceptance_engine is not None
            else GoalAcceptanceEngine()
        )

        self.policy = (
            policy
            if policy is not None
            else GoalEvaluationAcceptancePolicy()
        )

    # ------------------------------------------------------------------
    # SINGLE EVALUATION
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidate: GoalCandidate,
    ) -> GoalEvaluationAcceptanceDecision:
        """
        Evaluate one candidate.

        No lifecycle mutation occurs here.
        """

        self._validate_candidate(candidate)

        evaluation = self.evaluation_engine.evaluate(candidate)

        score = _evaluation_score(evaluation)
        recommendation = _evaluation_recommendation(evaluation)

        eligible = self._passes_evaluation_gate(
            score=score,
            recommendation=recommendation,
        )

        metadata = {
            "evaluation_score": score,
            "evaluation_recommendation": recommendation,
            "eligible": eligible,
        }

        return GoalEvaluationAcceptanceDecision(
            evaluation=evaluation,
            eligible=eligible,
            acceptance_decision=None,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # MANY EVALUATIONS
    # ------------------------------------------------------------------

    def evaluate_many(
        self,
        candidates: Iterable[GoalCandidate],
    ) -> tuple[GoalEvaluationAcceptanceDecision, ...]:
        """
        Evaluate and rank multiple candidates.

        max_candidates is ALWAYS applied after ranking.

        This is the important fix for:

            assert len(decisions) == max_candidates
        """

        candidate_list = tuple(candidates)

        if not candidate_list:
            return ()

        decisions = [
            self.evaluate(candidate)
            for candidate in candidate_list
        ]

        if self.policy.prioritize_by_evaluation:
            decisions.sort(
                key=self._ranking_key,
                reverse=True,
            )

        # --------------------------------------------------------------
        # IMPORTANT:
        # Apply max_candidates AFTER ranking.
        # --------------------------------------------------------------

        if self.policy.max_candidates is not None:
            decisions = decisions[
                : self.policy.max_candidates
            ]

        return tuple(decisions)

    # ------------------------------------------------------------------
    # ACCEPT SINGLE
    # ------------------------------------------------------------------

    def accept(
        self,
        candidate: GoalCandidate,
        integration: AgentGoalAcceptanceIntegrator,
    ) -> GoalEvaluationAcceptanceResult:
        """
        Evaluate and, when eligible, delegate lifecycle acceptance.
        """

        self._validate_candidate(candidate)

        if integration is None:
            raise GoalEvaluationAcceptanceInputError(
                "integration cannot be None"
            )

        evaluation_decision = self.evaluate(candidate)

        # --------------------------------------------------------------
        # Evaluation gate
        # --------------------------------------------------------------

        if not evaluation_decision.eligible:
            return GoalEvaluationAcceptanceResult(
                decision=evaluation_decision,
                acceptance_result=None,
                metadata={
                    **dict(evaluation_decision.metadata),
                    "stage": "evaluation",
                    "accepted": False,
                },
            )

        # --------------------------------------------------------------
        # Acceptance gate
        #
        # IMPORTANT:
        # GoalAcceptanceEngine.evaluate() is side-effect free.
        # --------------------------------------------------------------

        acceptance_decision = (
            self.acceptance_engine.evaluate(
                candidate,
                integration.manager,
            )
        )

        enriched_decision = GoalEvaluationAcceptanceDecision(
            evaluation=evaluation_decision.evaluation,
            eligible=True,
            acceptance_decision=acceptance_decision,
            metadata={
                **dict(evaluation_decision.metadata),
                "acceptance_reason": acceptance_decision.reason.value,
                "acceptance_accepted": acceptance_decision.accepted,
            },
        )

        # --------------------------------------------------------------
        # Acceptance rejected.
        #
        # No lifecycle mutation.
        # --------------------------------------------------------------

        if not acceptance_decision.accepted:
            return GoalEvaluationAcceptanceResult(
                decision=enriched_decision,
                acceptance_result=None,
                metadata={
                    **dict(enriched_decision.metadata),
                    "stage": "acceptance",
                    "accepted": False,
                },
            )

        # --------------------------------------------------------------
        # Final lifecycle application.
        # --------------------------------------------------------------

        acceptance_result = integration.accept(
            candidate
        )

        return GoalEvaluationAcceptanceResult(
            decision=enriched_decision,
            acceptance_result=acceptance_result,
            metadata={
                **dict(enriched_decision.metadata),
                "stage": "applied",
                "accepted": bool(
                    acceptance_result.accepted
                ),
            },
        )

    # ------------------------------------------------------------------
    # ACCEPT MANY
    # ------------------------------------------------------------------

    def accept_many(
        self,
        candidates: Iterable[GoalCandidate],
        integration: AgentGoalAcceptanceIntegrator,
        *,
        atomic: bool = False,
    ) -> GoalEvaluationAcceptanceBatch:
        """
        Evaluate and accept multiple candidates.

        atomic=False:
            Each candidate is processed independently.

        atomic=True:
            All candidates are evaluated and acceptance-gated first.
            Lifecycle mutation only starts if every candidate passes.

        There is intentionally NO rollback implementation because
        GoalLifecycleManager does not expose transactional rollback
        semantics.
        """

        candidate_list = tuple(candidates)

        if not candidate_list:
            return GoalEvaluationAcceptanceBatch(
                results=()
            )

        if integration is None:
            raise GoalEvaluationAcceptanceInputError(
                "integration cannot be None"
            )

        # --------------------------------------------------------------
        # First evaluate all candidates.
        # --------------------------------------------------------------

        decisions = self.evaluate_many(
            candidate_list
        )

        # --------------------------------------------------------------
        # Non-atomic mode.
        # --------------------------------------------------------------

        if not atomic:
            results = tuple(
                self._accept_evaluated_decision(
                    decision,
                    integration,
                )
                for decision in decisions
            )

            return GoalEvaluationAcceptanceBatch(
                results=results
            )

        # --------------------------------------------------------------
        # Atomic validation phase.
        #
        # No lifecycle mutation happens here.
        # --------------------------------------------------------------

        validated: list[
            GoalEvaluationAcceptanceDecision
        ] = []

        for decision in decisions:

            if not decision.eligible:
                result = GoalEvaluationAcceptanceResult(
                    decision=decision,
                    acceptance_result=None,
                    metadata={
                        **dict(decision.metadata),
                        "stage": "evaluation",
                        "accepted": False,
                    },
                )

                return GoalEvaluationAcceptanceBatch(
                    results=(
                        result,
                    )
                )

            acceptance_decision = (
                self.acceptance_engine.evaluate(
                    decision.candidate,
                    integration.manager,
                )
            )

            enriched = GoalEvaluationAcceptanceDecision(
                evaluation=decision.evaluation,
                eligible=True,
                acceptance_decision=acceptance_decision,
                metadata={
                    **dict(decision.metadata),
                    "acceptance_reason": (
                        acceptance_decision.reason.value
                    ),
                    "acceptance_accepted": (
                        acceptance_decision.accepted
                    ),
                },
            )

            if not acceptance_decision.accepted:
                result = GoalEvaluationAcceptanceResult(
                    decision=enriched,
                    acceptance_result=None,
                    metadata={
                        **dict(enriched.metadata),
                        "stage": "acceptance",
                        "accepted": False,
                    },
                )

                return GoalEvaluationAcceptanceBatch(
                    results=(result,)
                )

            validated.append(enriched)

        # --------------------------------------------------------------
        # All validation succeeded.
        #
        # Now mutation is allowed.
        # --------------------------------------------------------------

        results: list[
            GoalEvaluationAcceptanceResult
        ] = []

        for decision in validated:

            acceptance_result = integration.accept(
                decision.candidate
            )

            results.append(
                GoalEvaluationAcceptanceResult(
                    decision=decision,
                    acceptance_result=acceptance_result,
                    metadata={
                        **dict(decision.metadata),
                        "stage": "applied",
                        "accepted": bool(
                            acceptance_result.accepted
                        ),
                    },
                )
            )

        return GoalEvaluationAcceptanceBatch(
            results=tuple(results)
        )

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _accept_evaluated_decision(
        self,
        decision: GoalEvaluationAcceptanceDecision,
        integration: AgentGoalAcceptanceIntegrator,
    ) -> GoalEvaluationAcceptanceResult:

        if not decision.eligible:
            return GoalEvaluationAcceptanceResult(
                decision=decision,
                acceptance_result=None,
                metadata={
                    **dict(decision.metadata),
                    "stage": "evaluation",
                    "accepted": False,
                },
            )

        acceptance_decision = (
            self.acceptance_engine.evaluate(
                decision.candidate,
                integration.manager,
            )
        )

        enriched = GoalEvaluationAcceptanceDecision(
            evaluation=decision.evaluation,
            eligible=True,
            acceptance_decision=acceptance_decision,
            metadata={
                **dict(decision.metadata),
                "acceptance_reason": (
                    acceptance_decision.reason.value
                ),
                "acceptance_accepted": (
                    acceptance_decision.accepted
                ),
            },
        )

        if not acceptance_decision.accepted:
            return GoalEvaluationAcceptanceResult(
                decision=enriched,
                acceptance_result=None,
                metadata={
                    **dict(enriched.metadata),
                    "stage": "acceptance",
                    "accepted": False,
                },
            )

        acceptance_result = integration.accept(
            decision.candidate
        )

        return GoalEvaluationAcceptanceResult(
            decision=enriched,
            acceptance_result=acceptance_result,
            metadata={
                **dict(enriched.metadata),
                "stage": "applied",
                "accepted": bool(
                    acceptance_result.accepted
                ),
            },
        )

    def _passes_evaluation_gate(
        self,
        *,
        score: float,
        recommendation: str,
    ) -> bool:

        if (
            self.policy.minimum_evaluation_score
            is not None
        ):
            if (
                score
                < self.policy.minimum_evaluation_score
            ):
                return False

        if self.policy.require_recommendation:

            normalized = (
                str(recommendation)
                .strip()
                .lower()
            )

            if normalized not in {
                "prioritize",
                "accept",
                "recommended",
                "recommend",
                "high",
                "strong",
            }:
                return False

        return True

    @staticmethod
    def _ranking_key(
        decision: GoalEvaluationAcceptanceDecision,
    ) -> tuple[float, float, float, str]:

        candidate = decision.candidate

        return (
            1.0 if decision.eligible else 0.0,
            float(decision.evaluation_score),
            float(
                getattr(
                    candidate,
                    "proposal_score",
                    0.0,
                )
            ),
            str(
                getattr(
                    candidate,
                    "candidate_id",
                    "",
                )
            ),
        )

    @staticmethod
    def _validate_candidate(
        candidate: GoalCandidate,
    ) -> None:

        if candidate is None:
            raise GoalEvaluationAcceptanceInputError(
                "candidate cannot be None"
            )

        if not isinstance(
            candidate,
            GoalCandidate,
        ):
            raise GoalEvaluationAcceptanceInputError(
                "candidate must be a GoalCandidate"
            )


# ============================================================================
# HELPERS
# ============================================================================


def _evaluation_score(
    evaluation: GoalEvaluationResult,
) -> float:
    """
    Resolve the evaluation score from the current
    GoalEvaluationResult implementation.

    The current project uses evaluation_score.
    A couple of fallbacks are kept so this integration
    remains compatible with earlier evaluation result
    implementations.
    """

    for name in (
        "evaluation_score",
        "combined_score",
        "confidence_adjusted_utility",
        "score",
    ):
        value = getattr(
            evaluation,
            name,
            None,
        )

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return 0.0


def _evaluation_recommendation(
    evaluation: GoalEvaluationResult,
) -> str:
    """
    Resolve recommendation from the current evaluation result.
    """

    for name in (
        "evaluation_recommendation",
        "recommendation",
    ):
        value = getattr(
            evaluation,
            name,
            None,
        )

        if value is not None:
            return str(value)

    return ""


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "GoalEvaluationAcceptanceError",
    "GoalEvaluationAcceptanceConfigurationError",
    "GoalEvaluationAcceptanceInputError",
    "GoalEvaluationAcceptancePolicy",
    "GoalEvaluationAcceptanceDecision",
    "GoalEvaluationAcceptanceResult",
    "GoalEvaluationAcceptanceBatch",
    "GoalEvaluationAcceptanceEngine",
]