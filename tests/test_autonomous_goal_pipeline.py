from __future__ import annotations

import pytest

from src.agent.llm.autonomous_goal_pipeline import (
    AutonomousGoalPipeline,
    AutonomousGoalPipelineConfigurationError,
)
from src.agent.llm.goal_acceptance import (
    AgentGoalAcceptanceIntegrator,
)
from src.agent.llm.goal_proposal import (
    GoalCandidate,
)
from src.agent.llm.memory_goal_generation import (
    MemoryGap,
    MemoryGapType,
)


class TestAgent:
    pass


def make_agent():
    return TestAgent()


def make_gap(
    gap_id: str = "gap-1",
    *,
    importance: float = 0.9,
    urgency: float = 0.9,
    uncertainty: float = 0.8,
    novelty: float = 0.8,
    feasibility: float = 0.9,
    evidence_quality: float = 0.9,
):
    return MemoryGap(
        gap_id=gap_id,
        gap_type=MemoryGapType.MISSING_KNOWLEDGE,
        description="Learn an unknown fact",
        importance=importance,
        urgency=urgency,
        uncertainty=uncertainty,
        novelty=novelty,
        feasibility=feasibility,
        evidence_quality=evidence_quality,
        expected_value=0.9,
        target="unknown_fact",
        source="test",
    )


class TestConfiguration:

    def test_pipeline_can_be_created(self):
        pipeline = AutonomousGoalPipeline()

        assert pipeline.generation_pipeline is not None
        assert pipeline.evaluation_acceptance_engine is not None
        assert pipeline.acceptance_integrator is None

    def test_for_agent_creates_integrator(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        assert pipeline.acceptance_integrator is not None
        assert agent.goal_manager is pipeline.manager

    def test_none_agent_is_rejected(self):
        with pytest.raises(
            AutonomousGoalPipelineConfigurationError
        ):
            AutonomousGoalPipeline.for_agent(None)


class TestGeneration:

    def test_generate_produces_candidates(self):
        pipeline = AutonomousGoalPipeline()

        result = pipeline.generate(
            [make_gap()]
        )

        assert result is not None
        assert result.candidates
        assert result.best_candidate is not None

    def test_generate_does_not_require_agent(self):
        pipeline = AutonomousGoalPipeline()

        result = pipeline.generate(
            [make_gap()]
        )

        assert result.candidates
        assert pipeline.acceptance_integrator is None

    def test_generate_multiple_gaps(self):
        pipeline = AutonomousGoalPipeline()

        result = pipeline.generate(
            [
                make_gap("gap-1"),
                make_gap("gap-2"),
                make_gap("gap-3"),
            ]
        )

        assert result.candidates
        assert len(result.candidates) <= 10


class TestEvaluation:

    def test_evaluation_is_side_effect_free(self):
        pipeline = AutonomousGoalPipeline()

        generation = pipeline.generate(
            [make_gap()]
        )

        decisions = pipeline.evaluate(
            generation.candidates
        )

        assert decisions
        assert len(decisions) == len(
            generation.candidates
        )

    def test_empty_evaluation(self):
        pipeline = AutonomousGoalPipeline()

        assert pipeline.evaluate([]) == ()

    def test_candidates_remain_immutable(self):
        pipeline = AutonomousGoalPipeline()

        generation = pipeline.generate(
            [make_gap()]
        )

        candidate = generation.best_candidate

        assert candidate is not None

        assert isinstance(
            candidate,
            GoalCandidate,
        )


class TestAcceptance:

    def test_accept_requires_integrator(self):
        pipeline = AutonomousGoalPipeline()

        generation = pipeline.generate(
            [make_gap()]
        )

        with pytest.raises(
            AutonomousGoalPipelineConfigurationError
        ):
            pipeline.accept(
                generation.candidates
            )

    def test_accepts_generated_goal(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        generation = pipeline.generate(
            [make_gap()]
        )

        results = pipeline.accept(
            generation.candidates
        )

        assert results

        accepted = [
            result
            for result in results
            if result.accepted
        ]

        assert accepted

        assert (
            accepted[0].goal is not None
        )


class TestEndToEnd:

    def test_run_generates_evaluates_and_accepts(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        result = pipeline.run(
            [make_gap()]
        )

        assert result.candidate_count > 0
        assert result.evaluation_count > 0
        assert result.acceptance_count > 0
        assert result.accepted_count > 0

        assert result.has_candidates
        assert result.has_accepted_goal

        assert (
            result.best_candidate is not None
        )

        assert (
            result.best_accepted_goal is not None
        )

    def test_run_without_acceptance(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        before = pipeline.manager.snapshot()

        result = pipeline.run(
            [make_gap()],
            accept_goals=False,
        )

        after = pipeline.manager.snapshot()

        assert result.candidate_count > 0
        assert result.evaluation_count > 0
        assert result.acceptance_count == 0

        assert before == after

    def test_run_empty_input(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        result = pipeline.run([])

        assert result.candidate_count == 0
        assert result.evaluation_count == 0
        assert result.acceptance_count == 0
        assert result.accepted_count == 0

    def test_run_one(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        result = pipeline.run_one(
            make_gap()
        )

        assert result.candidate_count > 0
        assert result.evaluation_count > 0
        assert result.acceptance_count > 0

    def test_manager_contains_accepted_goal(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        result = pipeline.run(
            [make_gap()]
        )

        assert result.accepted_count > 0

        for accepted in result.acceptance:
            if accepted.accepted:
                assert accepted.goal is not None

                stored = pipeline.manager.get_goal(
                    accepted.goal.goal_id
                )

                assert (
                    stored.goal_id
                    == accepted.goal.goal_id
                )

    def test_pipeline_does_not_bypass_acceptance(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        result = pipeline.run(
            [make_gap()]
        )

        assert result.acceptance

        for item in result.acceptance:
            assert item.decision is not None

    def test_result_is_immutable_structure(self):
        agent = make_agent()

        pipeline = (
            AutonomousGoalPipeline.for_agent(
                agent
            )
        )

        result = pipeline.run(
            [make_gap()]
        )

        assert isinstance(
            result.candidates,
            tuple,
        )

        assert isinstance(
            result.evaluation,
            tuple,
        )

        assert isinstance(
            result.acceptance,
            tuple,
        )