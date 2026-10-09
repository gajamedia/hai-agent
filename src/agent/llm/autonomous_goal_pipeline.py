from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .goal_acceptance import (
    AgentGoalAcceptanceIntegrator,
    GoalAcceptanceResult,
)
from .goal_evaluation_acceptance import (
    GoalEvaluationAcceptanceDecision,
    GoalEvaluationAcceptanceEngine,
)
from .goal_generation_pipeline import (
    GoalGenerationPipeline,
    GoalGenerationPipelineResult,
)
from .goal_proposal import GoalCandidate


class AutonomousGoalPipelineError(Exception):
    """Base exception for the autonomous goal pipeline."""


class AutonomousGoalPipelineConfigurationError(
    AutonomousGoalPipelineError
):
    """Raised when the pipeline configuration is invalid."""


class AutonomousGoalPipelineInputError(
    AutonomousGoalPipelineError
):
    """Raised when an invalid input is provided."""


@dataclass(frozen=True)
class AutonomousGoalPipelineResult:
    """Immutable summary of one end-to-end pipeline run."""

    generation: GoalGenerationPipelineResult
    evaluation: tuple[GoalEvaluationAcceptanceDecision, ...] = ()
    acceptance: tuple[GoalAcceptanceResult, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.generation, GoalGenerationPipelineResult):
            raise AutonomousGoalPipelineInputError(
                "generation must be a GoalGenerationPipelineResult."
            )

        object.__setattr__(self, "evaluation", tuple(self.evaluation))
        object.__setattr__(self, "acceptance", tuple(self.acceptance))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def candidates(self) -> tuple[GoalCandidate, ...]:
        return tuple(self.generation.candidates)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def evaluation_count(self) -> int:
        return len(self.evaluation)

    @property
    def acceptance_count(self) -> int:
        return len(self.acceptance)

    @property
    def accepted_count(self) -> int:
        return sum(1 for result in self.acceptance if result.accepted)

    @property
    def rejected_count(self) -> int:
        return sum(1 for result in self.acceptance if not result.accepted)

    @property
    def accepted_goals(self) -> tuple[Any, ...]:
        return tuple(
            result.goal
            for result in self.acceptance
            if result.accepted and result.goal is not None
        )

    @property
    def best_candidate(self) -> Optional[GoalCandidate]:
        return self.generation.best_candidate

    @property
    def best_accepted_goal(self) -> Optional[Any]:
        for result in self.acceptance:
            if result.accepted and result.goal is not None:
                return result.goal
        return None

    @property
    def has_candidates(self) -> bool:
        return self.candidate_count > 0

    @property
    def has_accepted_goal(self) -> bool:
        return self.accepted_count > 0


class AutonomousGoalPipeline:
    """
    Orchestrates the existing goal-generation architecture.

    Flow:
        Memory gap
            -> GoalGenerationPipeline
            -> GoalCandidate
            -> GoalEvaluationAcceptanceEngine
            -> AgentGoalAcceptanceIntegrator
            -> GoalLifecycleManager

    This class coordinates existing modules. It does not replace their
    generation, evaluation, or acceptance logic.
    """

    def __init__(
        self,
        *,
        generation_pipeline: Optional[GoalGenerationPipeline] = None,
        evaluation_acceptance_engine: Optional[
            GoalEvaluationAcceptanceEngine
        ] = None,
        acceptance_integrator: Optional[
            AgentGoalAcceptanceIntegrator
        ] = None,
    ) -> None:
        self.generation_pipeline = (
            generation_pipeline
            if generation_pipeline is not None
            else GoalGenerationPipeline()
        )
        self.evaluation_acceptance_engine = (
            evaluation_acceptance_engine
            if evaluation_acceptance_engine is not None
            else GoalEvaluationAcceptanceEngine()
        )
        self.acceptance_integrator = acceptance_integrator

        if not isinstance(self.generation_pipeline, GoalGenerationPipeline):
            raise AutonomousGoalPipelineConfigurationError(
                "generation_pipeline must be a GoalGenerationPipeline."
            )

        if not isinstance(
            self.evaluation_acceptance_engine,
            GoalEvaluationAcceptanceEngine,
        ):
            raise AutonomousGoalPipelineConfigurationError(
                "evaluation_acceptance_engine must be a "
                "GoalEvaluationAcceptanceEngine."
            )

        if (
            self.acceptance_integrator is not None
            and not isinstance(
                self.acceptance_integrator,
                AgentGoalAcceptanceIntegrator,
            )
        ):
            raise AutonomousGoalPipelineConfigurationError(
                "acceptance_integrator must be an "
                "AgentGoalAcceptanceIntegrator or None."
            )

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        *,
        generation_pipeline: Optional[GoalGenerationPipeline] = None,
        evaluation_acceptance_engine: Optional[
            GoalEvaluationAcceptanceEngine
        ] = None,
        acceptance_integrator: Optional[
            AgentGoalAcceptanceIntegrator
        ] = None,
        auto_select: bool = False,
    ) -> "AutonomousGoalPipeline":
        if agent is None:
            raise AutonomousGoalPipelineConfigurationError(
                "agent must not be None."
            )

        if acceptance_integrator is None:
            acceptance_integrator = AgentGoalAcceptanceIntegrator.for_agent(
                agent,
                auto_select=auto_select,
            )
        elif acceptance_integrator.agent is not agent:
            raise AutonomousGoalPipelineConfigurationError(
                "acceptance_integrator must belong to the supplied agent."
            )

        return cls(
            generation_pipeline=generation_pipeline,
            evaluation_acceptance_engine=evaluation_acceptance_engine,
            acceptance_integrator=acceptance_integrator,
        )

    def generate(
        self,
        gaps: Iterable[Any],
        *,
        max_signals: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> GoalGenerationPipelineResult:
        """Run generation only. No goal is accepted or inserted here."""
        gap_items = self._normalize_gaps(gaps)
        return self.generation_pipeline.generate(
            gap_items,
            max_signals=max_signals,
            max_candidates=max_candidates,
        )

    def evaluate(
        self,
        candidates: Iterable[GoalCandidate],
    ) -> tuple[GoalEvaluationAcceptanceDecision, ...]:
        """Evaluate candidates without accepting them."""
        candidate_items = self._normalize_candidates(candidates)
        if not candidate_items:
            return ()

        return tuple(
            self.evaluation_acceptance_engine.evaluate_many(candidate_items)
        )

    def accept(
        self,
        candidates: Iterable[GoalCandidate],
        *,
        atomic: bool = False,
    ) -> tuple[GoalAcceptanceResult, ...]:
        """
        Evaluate and apply candidates through the existing evaluation/
        acceptance engine and agent integrator.

        GoalEvaluationAcceptanceEngine.accept_many() returns a
        GoalEvaluationAcceptanceBatch. Its individual results are stored
        in batch.results; the batch object itself is not iterable.
        """
        if self.acceptance_integrator is None:
            raise AutonomousGoalPipelineConfigurationError(
                "An acceptance_integrator is required. "
                "Use AutonomousGoalPipeline.for_agent(...)."
            )

        candidate_items = self._normalize_candidates(candidates)
        if not candidate_items:
            return ()

        batch = self.evaluation_acceptance_engine.accept_many(
            candidate_items,
            self.acceptance_integrator,
            atomic=atomic,
        )

        converted_results: list[GoalAcceptanceResult] = []

        for batch_result in batch.results:
            # Normal success, or a rejection produced by the underlying
            # goal acceptance engine: preserve its original result object.
            acceptance_result = batch_result.acceptance_result
            if acceptance_result is not None:
                converted_results.append(acceptance_result)
                continue

            # A candidate can be rejected by the evaluation gate before
            # the acceptance integrator is invoked. Preserve that outcome
            # as a GoalAcceptanceResult so callers still receive one
            # consistently typed result for the reported batch item.
            candidate = batch_result.decision.evaluation.candidate
            acceptance_decision = (
                batch_result.decision.acceptance_decision
            )
            if acceptance_decision is None:
                acceptance_decision = self.acceptance_integrator.evaluate(
                    candidate
                )

            metadata = dict(batch_result.metadata or {})
            metadata.setdefault("applied", False)
            metadata.setdefault("stage", "evaluation")
            metadata["evaluation_eligible"] = bool(
                batch_result.decision.eligible
            )

            converted_results.append(
                GoalAcceptanceResult(
                    decision=acceptance_decision,
                    goal=None,
                    current_goal_id=self.acceptance_integrator.manager.current_goal_id,
                    projected_to_agent=(
                        self.acceptance_integrator.goal_integrator.project_to_agent
                    ),
                    metadata=metadata,
                )
            )

        return tuple(converted_results)

    def run(
        self,
        gaps: Iterable[Any],
        *,
        max_signals: Optional[int] = None,
        max_candidates: Optional[int] = None,
        atomic: bool = False,
        accept_goals: bool = True,
    ) -> AutonomousGoalPipelineResult:
        """
        Run the end-to-end pipeline.

        accept_goals=True: Generate -> Evaluate -> Accept
        accept_goals=False: Generate -> Evaluate

        The input iterable is materialized once so generators are not
        accidentally consumed twice.
        """
        if not isinstance(accept_goals, bool):
            raise AutonomousGoalPipelineInputError(
                "accept_goals must be a bool."
            )
        if not isinstance(atomic, bool):
            raise AutonomousGoalPipelineInputError("atomic must be a bool.")

        gap_items = self._normalize_gaps(gaps)
        generation = self.generation_pipeline.generate(
            gap_items,
            max_signals=max_signals,
            max_candidates=max_candidates,
        )

        candidates = tuple(generation.candidates)
        evaluation = self.evaluate(candidates)
        acceptance: tuple[GoalAcceptanceResult, ...] = ()

        if accept_goals and candidates:
            acceptance = self.accept(candidates, atomic=atomic)

        metadata = {
            "input_gap_count": len(gap_items),
            "candidate_count": len(candidates),
            "evaluation_count": len(evaluation),
            "acceptance_count": len(acceptance),
            "accepted_count": sum(1 for item in acceptance if item.accepted),
            "accept_goals": accept_goals,
            "atomic": atomic,
        }

        return AutonomousGoalPipelineResult(
            generation=generation,
            evaluation=evaluation,
            acceptance=acceptance,
            metadata=metadata,
        )

    def run_one(
        self,
        gap: Any,
        *,
        atomic: bool = False,
        accept_goals: bool = True,
    ) -> AutonomousGoalPipelineResult:
        """Convenience method for processing one memory gap."""
        return self.run(
            (gap,),
            atomic=atomic,
            accept_goals=accept_goals,
        )

    @property
    def manager(self):
        if self.acceptance_integrator is None:
            return None
        return self.acceptance_integrator.manager

    @staticmethod
    def _normalize_gaps(gaps: Iterable[Any]) -> tuple[Any, ...]:
        if gaps is None:
            raise AutonomousGoalPipelineInputError("gaps must not be None.")

        if isinstance(gaps, (str, bytes, Mapping)):
            raise AutonomousGoalPipelineInputError(
                "gaps must be an iterable of gap objects, "
                "not a string, bytes, or mapping."
            )

        try:
            gap_items = tuple(gaps)
        except TypeError as exc:
            raise AutonomousGoalPipelineInputError(
                "gaps must be iterable."
            ) from exc

        for index, gap in enumerate(gap_items):
            if gap is None:
                raise AutonomousGoalPipelineInputError(
                    f"gaps[{index}] must not be None."
                )

        return gap_items

    @staticmethod
    def _normalize_candidates(
        candidates: Iterable[GoalCandidate],
    ) -> tuple[GoalCandidate, ...]:
        if candidates is None:
            raise AutonomousGoalPipelineInputError(
                "candidates must not be None."
            )

        if isinstance(candidates, (str, bytes, Mapping)):
            raise AutonomousGoalPipelineInputError(
                "candidates must be an iterable of GoalCandidate objects."
            )

        try:
            candidate_items = tuple(candidates)
        except TypeError as exc:
            raise AutonomousGoalPipelineInputError(
                "candidates must be iterable."
            ) from exc

        for index, candidate in enumerate(candidate_items):
            if not isinstance(candidate, GoalCandidate):
                raise AutonomousGoalPipelineInputError(
                    f"candidates[{index}] must be a GoalCandidate."
                )

        return candidate_items


__all__ = [
    "AutonomousGoalPipeline",
    "AutonomousGoalPipelineError",
    "AutonomousGoalPipelineConfigurationError",
    "AutonomousGoalPipelineInputError",
    "AutonomousGoalPipelineResult",
]
