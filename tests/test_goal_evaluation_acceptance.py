from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent.llm.goal_acceptance import (
    AgentGoalAcceptanceIntegrator,
    GoalAcceptanceEngine,
    GoalAcceptancePolicy,
)
from agent.llm.goal_evaluation_acceptance import (
    GoalEvaluationAcceptanceBatch,
    GoalEvaluationAcceptanceDecision,
    GoalEvaluationAcceptanceEngine,
    GoalEvaluationAcceptancePolicy,
    GoalEvaluationAcceptanceResult,
)
from agent.llm.goal_proposal import (
    GoalCandidate,
    GoalOrigin,
)


# ============================================================================
# HELPERS
# ============================================================================


def candidate(
    candidate_id: str = "goal-1",
    *,
    description: str | None = None,
    proposal_score: float = 0.80,
    feasibility: float = 0.80,
    priority: float = 0.80,
    urgency: float = 0.70,
    expected_value: float = 0.80,
    novelty: float = 0.70,
    evidence_quality: float = 0.80,
    payload=None,
    metadata=None,
) -> GoalCandidate:

    return GoalCandidate(
        candidate_id=candidate_id,
        description=(
            description
            if description is not None
            else f"Goal {candidate_id}"
        ),
        origin=GoalOrigin.MEMORY,
        priority=priority,
        proposal_score=proposal_score,
        urgency=urgency,
        expected_value=expected_value,
        feasibility=feasibility,
        novelty=novelty,
        evidence_quality=evidence_quality,
        payload=payload or {},
        metadata=metadata or {},
    )


def make_integration():
    """
    Create a minimal Agent instance and its goal acceptance integration.

    IMPORTANT:
    The helper name intentionally differs from the local variable
    'integration' used by the tests.
    """

    agent = type(
        "Agent",
        (),
        {},
    )()

    integration = AgentGoalAcceptanceIntegrator.for_agent(
        agent
    )

    return agent, integration


def manager_goal_ids(integration):
    manager = integration.manager

    if hasattr(manager, "list_goals"):
        goals = manager.list_goals()

        return {
            getattr(goal, "goal_id", None)
            for goal in goals
        }

    if hasattr(manager, "goals"):
        goals = manager.goals

        if isinstance(goals, dict):
            return set(goals.keys())

        return {
            getattr(goal, "goal_id", None)
            for goal in goals
        }

    return set()


# ============================================================================
# POLICY
# ============================================================================


class TestPolicy:

    def test_default_policy(self):
        policy = GoalEvaluationAcceptancePolicy()

        assert policy.minimum_evaluation_score is None
        assert policy.require_recommendation is False
        assert policy.prioritize_by_evaluation is True
        assert policy.max_candidates is None

    def test_valid_score(self):
        policy = GoalEvaluationAcceptancePolicy(
            minimum_evaluation_score=0.75
        )

        assert policy.minimum_evaluation_score == 0.75

    def test_invalid_score(self):
        with pytest.raises(Exception):
            GoalEvaluationAcceptancePolicy(
                minimum_evaluation_score=1.5
            )

    def test_invalid_max_candidates(self):
        with pytest.raises(Exception):
            GoalEvaluationAcceptancePolicy(
                max_candidates=0
            )


# ============================================================================
# ENGINE EVALUATION
# ============================================================================


class TestEngine:

    def test_evaluate_returns_decision(self):
        engine = GoalEvaluationAcceptanceEngine()

        item = candidate()

        result = engine.evaluate(item)

        assert isinstance(
            result,
            GoalEvaluationAcceptanceDecision,
        )

    def test_result_candidate_identity(self):
        engine = GoalEvaluationAcceptanceEngine()

        item = candidate()

        result = engine.evaluate(item)

        assert result.candidate is item

    def test_evaluate_does_not_mutate_lifecycle(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        before = manager_goal_ids(integration)

        item = candidate()

        result = engine.evaluate(item)

        after = manager_goal_ids(integration)

        assert result.eligible is True
        assert before == after

    def test_default_candidate_is_eligible(self):
        engine = GoalEvaluationAcceptanceEngine()

        result = engine.evaluate(
            candidate()
        )

        assert result.eligible is True

    def test_evaluation_score_is_available(self):
        engine = GoalEvaluationAcceptanceEngine()

        result = engine.evaluate(
            candidate()
        )

        assert isinstance(
            result.evaluation_score,
            float,
        )

    def test_evaluation_recommendation_is_available(self):
        engine = GoalEvaluationAcceptanceEngine()

        result = engine.evaluate(
            candidate()
        )

        assert isinstance(
            result.evaluation_recommendation,
            str,
        )

    def test_none_candidate_rejected(self):
        engine = GoalEvaluationAcceptanceEngine()

        with pytest.raises(Exception):
            engine.evaluate(None)

    def test_invalid_candidate_rejected(self):
        engine = GoalEvaluationAcceptanceEngine()

        with pytest.raises(Exception):
            engine.evaluate(
                object()
            )


# ============================================================================
# ACCEPTANCE
# ============================================================================


class TestAcceptance:

    def test_accept_delegates_to_integrator(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        item = candidate()

        result = engine.accept(
            item,
            integration,
        )

        assert isinstance(
            result,
            GoalEvaluationAcceptanceResult,
        )

        assert result.accepted is True

    def test_acceptance_decision_is_preserved(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        item = candidate()

        result = engine.accept(
            item,
            integration,
        )

        assert result.decision.acceptance_decision is not None

        assert (
            result.decision.acceptance_decision.accepted
            is True
        )

    def test_existing_goal_is_rejected(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        item = candidate(
            candidate_id="same-goal"
        )

        first = engine.accept(
            item,
            integration,
        )

        assert first.accepted is True

        second = engine.accept(
            item,
            integration,
        )

        assert second.accepted is False

        assert (
            second.decision.acceptance_decision
            is not None
        )

    def test_accept_rejects_evaluation_gate_without_mutation(self):
        engine = GoalEvaluationAcceptanceEngine(
            policy=GoalEvaluationAcceptancePolicy(
                minimum_evaluation_score=0.99
            )
        )

        agent, integration = make_integration()

        item = candidate()

        before = manager_goal_ids(integration)

        result = engine.accept(
            item,
            integration,
        )

        after = manager_goal_ids(integration)

        assert result.accepted is False
        assert result.acceptance_result is None
        assert before == after

    def test_accept_rejects_acceptance_gate(self):
        engine = GoalEvaluationAcceptanceEngine(
            acceptance_engine=GoalAcceptanceEngine(
                policy=GoalAcceptancePolicy(
                    minimum_proposal_score=0.95
                )
            )
        )

        agent, integration = make_integration()

        item = candidate(
            proposal_score=0.50
        )

        result = engine.accept(
            item,
            integration,
        )

        assert result.accepted is False
        assert result.acceptance_result is None

        assert (
            result.decision.acceptance_decision
            is not None
        )

        assert (
            result.decision.acceptance_decision.accepted
            is False
        )


# ============================================================================
# BATCH EVALUATION
# ============================================================================


class TestBatch:

    def test_empty_batch(self):
        engine = GoalEvaluationAcceptanceEngine()

        decisions = engine.evaluate_many(
            []
        )

        assert decisions == ()

    def test_batch_returns_tuple(self):
        engine = GoalEvaluationAcceptanceEngine()

        decisions = engine.evaluate_many(
            [
                candidate("a"),
                candidate("b"),
            ]
        )

        assert isinstance(
            decisions,
            tuple,
        )

    def test_batch_count(self):
        engine = GoalEvaluationAcceptanceEngine()

        decisions = engine.evaluate_many(
            [
                candidate("a"),
                candidate("b"),
                candidate("c"),
            ]
        )

        assert len(decisions) == 3

    def test_batch_max_candidates(self):
        engine = GoalEvaluationAcceptanceEngine(
            policy=GoalEvaluationAcceptancePolicy(
                max_candidates=2
            )
        )

        decisions = engine.evaluate_many(
            [
                candidate("a"),
                candidate("b"),
                candidate("c"),
            ]
        )

        assert len(decisions) == 2

    def test_batch_results_are_ranked(self):
        engine = GoalEvaluationAcceptanceEngine()

        low = candidate(
            "low",
            proposal_score=0.40,
            expected_value=0.40,
            priority=0.40,
        )

        high = candidate(
            "high",
            proposal_score=0.95,
            expected_value=0.95,
            priority=0.95,
        )

        decisions = engine.evaluate_many(
            [
                low,
                high,
            ]
        )

        assert len(decisions) == 2

        assert (
            decisions[0].evaluation_score
            >= decisions[1].evaluation_score
        )

    def test_batch_evaluation_gate(self):
        engine = GoalEvaluationAcceptanceEngine(
            policy=GoalEvaluationAcceptancePolicy(
                minimum_evaluation_score=0.99
            )
        )

        decisions = engine.evaluate_many(
            [
                candidate("a"),
                candidate("b"),
            ]
        )

        assert len(decisions) == 2

        assert all(
            decision.eligible is False
            for decision in decisions
        )

    def test_batch_evaluation_does_not_mutate(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        before = manager_goal_ids(integration)

        engine.evaluate_many(
            [
                candidate("a"),
                candidate("b"),
                candidate("c"),
            ]
        )

        after = manager_goal_ids(integration)

        assert before == after


# ============================================================================
# BATCH ACCEPTANCE
# ============================================================================


class TestBatchAcceptance:

    def test_batch_accepts_goals(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        results = engine.accept_many(
            [
                candidate("a"),
                candidate("b"),
            ],
            integration,
        )

        assert isinstance(
            results,
            GoalEvaluationAcceptanceBatch,
        )

        assert results.count == 2

        assert results.accepted_count == 2

    def test_batch_result_properties(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        batch = engine.accept_many(
            [
                candidate("a"),
                candidate("b"),
            ],
            integration,
        )

        assert batch.count == 2

        assert len(batch.results) == 2

        assert len(batch.accepted) == 2

        assert len(batch.rejected) == 0

        assert batch.best is not None

    def test_batch_metadata(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        batch = engine.accept_many(
            [
                candidate("a"),
            ],
            integration,
        )

        result = batch.results[0]

        assert (
            "evaluation_score"
            in result.metadata
        )

        assert (
            "evaluation_recommendation"
            in result.metadata
        )

    def test_batch_empty(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        batch = engine.accept_many(
            [],
            integration,
        )

        assert batch.count == 0
        assert batch.accepted_count == 0
        assert batch.rejected_count == 0

    def test_batch_evaluation_gate(self):
        engine = GoalEvaluationAcceptanceEngine(
            policy=GoalEvaluationAcceptancePolicy(
                minimum_evaluation_score=0.99
            )
        )

        agent, integration = make_integration()

        before = manager_goal_ids(integration)

        batch = engine.accept_many(
            [
                candidate("a"),
                candidate("b"),
            ],
            integration,
        )

        after = manager_goal_ids(integration)

        assert batch.count == 2

        assert batch.accepted_count == 0

        assert before == after


# ============================================================================
# ATOMIC
# ============================================================================


class TestAtomic:

    def test_atomic_evaluation_gate_does_not_mutate(self):
        engine = GoalEvaluationAcceptanceEngine(
            policy=GoalEvaluationAcceptancePolicy(
                minimum_evaluation_score=0.99
            )
        )

        agent, integration = make_integration()

        before = manager_goal_ids(integration)

        batch = engine.accept_many(
            [
                candidate("a"),
                candidate("b"),
            ],
            integration,
            atomic=True,
        )

        after = manager_goal_ids(integration)

        assert before == after

        assert batch.accepted_count == 0

    def test_atomic_acceptance_gate_does_not_mutate(self):
        engine = GoalEvaluationAcceptanceEngine(
            acceptance_engine=GoalAcceptanceEngine(
                policy=GoalAcceptancePolicy(
                    minimum_proposal_score=0.95
                )
            )
        )

        agent, integration = make_integration()

        before = manager_goal_ids(integration)

        batch = engine.accept_many(
            [
                candidate(
                    "a",
                    proposal_score=0.50,
                ),
                candidate(
                    "b",
                    proposal_score=0.50,
                ),
            ],
            integration,
            atomic=True,
        )

        after = manager_goal_ids(integration)

        assert before == after

        assert batch.accepted_count == 0


# ============================================================================
# PAYLOAD / METADATA
# ============================================================================


class TestPreservation:

    def test_payload_survives_acceptance(self):
        payload = {
            "source": "memory",
            "target": "database",
        }

        item = candidate(
            metadata={
                "source": "memory",
            }
        )

        item = GoalCandidate(
            candidate_id=item.candidate_id,
            description=item.description,
            origin=item.origin,
            priority=item.priority,
            proposal_score=item.proposal_score,
            urgency=item.urgency,
            expected_value=item.expected_value,
            feasibility=item.feasibility,
            novelty=item.novelty,
            evidence_quality=item.evidence_quality,
            parent_goal_id=item.parent_goal_id,
            payload=payload,
            metadata=item.metadata,
        )

        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        result = engine.accept(
            item,
            integration,
        )

        assert result.accepted is True

        assert (
            result.candidate.payload
            == payload
        )

    def test_metadata_survives_acceptance(self):
        metadata = {
            "source": "memory",
            "gap_type": "missing_knowledge",
        }

        item = candidate(
            metadata=metadata
        )

        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        result = engine.accept(
            item,
            integration,
        )

        assert result.accepted is True

        assert (
            result.candidate.metadata
            == metadata
        )


# ============================================================================
# IMMUTABILITY
# ============================================================================


class TestImmutability:

    def test_decision_is_immutable(self):
        engine = GoalEvaluationAcceptanceEngine()

        decision = engine.evaluate(
            candidate()
        )

        with pytest.raises(
            FrozenInstanceError
        ):
            decision.eligible = False

    def test_result_is_immutable(self):
        engine = GoalEvaluationAcceptanceEngine()

        agent, integration = make_integration()

        result = engine.accept(
            candidate(),
            integration,
        )

        with pytest.raises(
            FrozenInstanceError
        ):
            result.acceptance_result = None


# ============================================================================
# MAX CANDIDATES
# ============================================================================


class TestMaxCandidates:

    def test_max_candidates_one(self):
        engine = GoalEvaluationAcceptanceEngine(
            policy=GoalEvaluationAcceptancePolicy(
                max_candidates=1
            )
        )

        decisions = engine.evaluate_many(
            [
                candidate("a"),
                candidate("b"),
                candidate("c"),
            ]
        )

        assert len(decisions) == 1

    def test_max_candidates_two(self):
        engine = GoalEvaluationAcceptanceEngine(
            policy=GoalEvaluationAcceptancePolicy(
                max_candidates=2
            )
        )

        decisions = engine.evaluate_many(
            [
                candidate("a"),
                candidate("b"),
                candidate("c"),
                candidate("d"),
            ]
        )

        assert len(decisions) == 2

    def test_max_candidates_larger_than_input(self):
        engine = GoalEvaluationAcceptanceEngine(
            policy=GoalEvaluationAcceptancePolicy(
                max_candidates=10
            )
        )

        decisions = engine.evaluate_many(
            [
                candidate("a"),
                candidate("b"),
            ]
        )

        assert len(decisions) == 2
