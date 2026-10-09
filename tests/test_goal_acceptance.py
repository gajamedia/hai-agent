from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.goal_acceptance import (
    AgentGoalAcceptanceIntegrator,
    GoalAcceptanceApplicationError,
    GoalAcceptanceConfigurationError,
    GoalAcceptanceDecision,
    GoalAcceptanceEngine,
    GoalAcceptanceInputError,
    GoalAcceptancePolicy,
    GoalAcceptanceReason,
    GoalAcceptanceResult,
)
from agent.llm.goal_management import GoalLifecycleManager, GoalStatus
from agent.llm.goal_proposal import GoalCandidate, GoalOrigin


def candidate(
    cid: str = "c1",
    description: str = "Ship feature",
    *,
    proposal_score: float = 0.8,
    feasibility: float = 0.8,
    parent_goal_id: str | None = None,
) -> GoalCandidate:
    return GoalCandidate(
        candidate_id=cid,
        description=description,
        origin=GoalOrigin.EXTERNAL,
        priority=proposal_score,
        proposal_score=proposal_score,
        urgency=0.8,
        expected_value=0.8,
        feasibility=feasibility,
        novelty=0.5,
        evidence_quality=0.8,
        parent_goal_id=parent_goal_id,
        metadata={"source": "test"},
    )


class TestPolicy:
    def test_defaults(self):
        policy = GoalAcceptancePolicy()
        assert policy.minimum_proposal_score == 0.35
        assert policy.minimum_feasibility == 0.25

    def test_bounds_score(self):
        with pytest.raises(ValueError):
            GoalAcceptancePolicy(minimum_proposal_score=1.1)

    def test_bounds_feasibility(self):
        with pytest.raises(ValueError):
            GoalAcceptancePolicy(minimum_feasibility=-0.1)

    def test_booleans_are_normalized(self):
        policy = GoalAcceptancePolicy(allow_duplicate_descriptions=1)
        assert policy.allow_duplicate_descriptions is True


class TestDecision:
    def test_accepted_decision(self):
        decision = GoalAcceptanceDecision(candidate("c1"), True, GoalAcceptanceReason.ACCEPTED)
        assert decision.should_accept is True

    def test_rejected_decision(self):
        decision = GoalAcceptanceDecision(
            candidate("c1"), False, GoalAcceptanceReason.BELOW_FEASIBILITY
        )
        assert decision.should_accept is False

    def test_invalid_accepted_reason(self):
        with pytest.raises(ValueError):
            GoalAcceptanceDecision(candidate("c1"), True, GoalAcceptanceReason.BELOW_FEASIBILITY)

    def test_invalid_rejected_reason(self):
        with pytest.raises(ValueError):
            GoalAcceptanceDecision(candidate("c1"), False, GoalAcceptanceReason.ACCEPTED)

    def test_collections_are_immutable_views(self):
        decision = GoalAcceptanceDecision(
            candidate("c1"), False, GoalAcceptanceReason.BELOW_FEASIBILITY,
            conflicts=["x"], metadata={"a": 1},
        )
        assert decision.conflicts == ("x",)
        assert decision.metadata == {"a": 1}


class TestEngine:
    def test_accepts_valid_candidate(self):
        manager = GoalLifecycleManager()
        decision = GoalAcceptanceEngine().evaluate(candidate(), manager)
        assert decision.accepted is True
        assert decision.reason is GoalAcceptanceReason.ACCEPTED

    def test_below_score_rejected(self):
        manager = GoalLifecycleManager()
        engine = GoalAcceptanceEngine(policy=GoalAcceptancePolicy(minimum_proposal_score=0.9))
        decision = engine.evaluate(candidate(proposal_score=0.8), manager)
        assert decision.reason is GoalAcceptanceReason.BELOW_PROPOSAL_SCORE

    def test_below_feasibility_rejected(self):
        manager = GoalLifecycleManager()
        engine = GoalAcceptanceEngine(policy=GoalAcceptancePolicy(minimum_feasibility=0.9))
        decision = engine.evaluate(candidate(feasibility=0.8), manager)
        assert decision.reason is GoalAcceptanceReason.BELOW_FEASIBILITY

    def test_duplicate_id_rejected(self):
        manager = GoalLifecycleManager()
        manager.add_goal("c1", "Existing")
        decision = GoalAcceptanceEngine().evaluate(candidate("c1"), manager)
        assert decision.reason is GoalAcceptanceReason.DUPLICATE_GOAL_ID
        assert "duplicate_goal_id" in decision.conflicts

    def test_duplicate_description_is_case_insensitive(self):
        manager = GoalLifecycleManager()
        manager.add_goal("existing", " ship   FEATURE ")
        decision = GoalAcceptanceEngine().evaluate(candidate(description="Ship feature"), manager)
        assert decision.reason is GoalAcceptanceReason.DUPLICATE_DESCRIPTION

    def test_duplicate_description_can_be_allowed(self):
        manager = GoalLifecycleManager()
        manager.add_goal("existing", "Ship feature")
        engine = GoalAcceptanceEngine(policy=GoalAcceptancePolicy(allow_duplicate_descriptions=True))
        decision = engine.evaluate(candidate(), manager)
        assert decision.accepted is True

    def test_duplicate_id_can_never_be_allowed(self):
        manager = GoalLifecycleManager()
        manager.add_goal("c1", "Existing")
        decision = GoalAcceptanceEngine().evaluate(candidate("c1"), manager)
        assert decision.accepted is False
        assert decision.reason is GoalAcceptanceReason.DUPLICATE_GOAL_ID

    def test_parent_not_found_rejected(self):
        manager = GoalLifecycleManager()
        decision = GoalAcceptanceEngine().evaluate(candidate(parent_goal_id="p1"), manager)
        assert decision.reason is GoalAcceptanceReason.PARENT_NOT_FOUND

    def test_cancelled_parent_rejected(self):
        manager = GoalLifecycleManager()
        manager.add_goal("p1", "Parent")
        manager.cancel("p1")
        decision = GoalAcceptanceEngine().evaluate(candidate(parent_goal_id="p1"), manager)
        assert decision.reason is GoalAcceptanceReason.PARENT_CANCELLED

    def test_completed_parent_is_allowed_by_existing_manager_contract(self):
        manager = GoalLifecycleManager()
        manager.add_goal("p1", "Parent")
        manager.activate("p1")
        manager.complete("p1")
        decision = GoalAcceptanceEngine().evaluate(candidate(parent_goal_id="p1"), manager)
        assert decision.accepted is True

    def test_no_mutation_during_evaluation(self):
        manager = GoalLifecycleManager()
        before = manager.snapshot()
        GoalAcceptanceEngine().evaluate(candidate(), manager)
        after = manager.snapshot()
        assert after == before

    def test_invalid_candidate_type(self):
        manager = GoalLifecycleManager()
        with pytest.raises(GoalAcceptanceInputError):
            GoalAcceptanceEngine().evaluate(object(), manager)  # type: ignore[arg-type]

    def test_invalid_manager_type(self):
        with pytest.raises(GoalAcceptanceInputError):
            GoalAcceptanceEngine().evaluate(candidate(), object())  # type: ignore[arg-type]

    def test_evaluate_many_preserves_order(self):
        manager = GoalLifecycleManager()
        decisions = GoalAcceptanceEngine().evaluate_many(
            [candidate("a", "A"), candidate("b", "B", proposal_score=0.1)], manager
        )
        assert [d.candidate.candidate_id for d in decisions] == ["a", "b"]
        assert decisions[1].accepted is False

    def test_threshold_equality_is_accepted(self):
        manager = GoalLifecycleManager()
        engine = GoalAcceptanceEngine(policy=GoalAcceptancePolicy(minimum_proposal_score=0.8))
        decision = engine.evaluate(candidate(proposal_score=0.8), manager)
        assert decision.accepted is True

    def test_feasibility_threshold_equality_is_accepted(self):
        manager = GoalLifecycleManager()
        engine = GoalAcceptanceEngine(policy=GoalAcceptancePolicy(minimum_feasibility=0.8))
        decision = engine.evaluate(candidate(feasibility=0.8), manager)
        assert decision.accepted is True


class TestIntegrator:
    def test_for_agent_attaches_manager(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        assert agent.goal_manager is integration.manager

    def test_accept_inserts_goal(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        result = integration.accept(candidate())
        assert result.accepted is True
        assert result.goal is not None
        assert integration.manager.get_goal("c1").description == "Ship feature"

    def test_accept_projects_current_goal_fields(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        result = integration.accept(candidate())
        assert agent.current_goal_id is None
        assert result.goal is not None

    def test_auto_select_activates_goal(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent, auto_select=True)
        result = integration.accept(candidate())
        assert result.current_goal_id == "c1"
        assert integration.manager.get_goal("c1").status is GoalStatus.ACTIVE
        assert agent.current_goal_id == "c1"

    def test_rejected_candidate_does_not_mutate_manager(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(
            agent,
            acceptance_engine=GoalAcceptanceEngine(
                policy=GoalAcceptancePolicy(minimum_proposal_score=0.9)
            ),
        )
        result = integration.accept(candidate(proposal_score=0.8))
        assert result.accepted is False
        assert integration.manager.list_goals() == ()

    def test_last_result_tracks_application(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        result = integration.accept(candidate())
        assert integration.last_result == result

    def test_snapshot_available(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        integration.accept(candidate())
        snapshot = integration.snapshot()
        assert snapshot.manager_sequence == 1

    def test_current_goal_property(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent, auto_select=True)
        integration.accept(candidate())
        assert integration.current_goal is not None
        assert integration.current_goal.goal_id == "c1"

    def test_existing_manager_can_be_reused(self):
        agent = type("Agent", (), {})()
        manager = GoalLifecycleManager()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent, manager=manager)
        assert integration.manager is manager

    def test_existing_goal_integrator_can_be_reused(self):
        agent = type("Agent", (), {})()
        from agent.llm.agent_goal_management_integration import AgentGoalManagementIntegrator
        goal_integrator = AgentGoalManagementIntegrator.for_agent(agent)
        integration = AgentGoalAcceptanceIntegrator(agent, goal_integrator=goal_integrator)
        assert integration.goal_integrator is goal_integrator

    def test_conflicting_manager_and_integrator_rejected(self):
        agent = type("Agent", (), {})()
        from agent.llm.agent_goal_management_integration import AgentGoalManagementIntegrator
        gi = AgentGoalManagementIntegrator.for_agent(agent)
        other = GoalLifecycleManager()
        with pytest.raises(GoalAcceptanceConfigurationError):
            AgentGoalAcceptanceIntegrator(agent, goal_integrator=gi, manager=other)

    def test_accept_many_non_atomic(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        results = integration.accept_many([candidate("a", "A"), candidate("b", "B")])
        assert len(results) == 2
        assert all(result.accepted for result in results)
        assert len(integration.manager.list_goals()) == 2

    def test_accept_many_handles_rejection(self):
        agent = type("Agent", (), {})()
        engine = GoalAcceptanceEngine(policy=GoalAcceptancePolicy(minimum_proposal_score=0.9))
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent, acceptance_engine=engine)
        results = integration.accept_many([candidate("a", "A"), candidate("b", "B", proposal_score=0.1)])
        assert results[0].accepted is False
        assert len(integration.manager.list_goals()) == 0

    def test_accept_many_atomic_prechecks_before_mutation(self):
        agent = type("Agent", (), {})()
        engine = GoalAcceptanceEngine(policy=GoalAcceptancePolicy(minimum_proposal_score=0.9))
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent, acceptance_engine=engine)
        results = integration.accept_many(
            [candidate("a", "A", proposal_score=0.95), candidate("b", "B", proposal_score=0.1)],
            atomic=True,
        )
        assert all(result.accepted is False for result in results)
        assert integration.manager.list_goals() == ()

    def test_manager_metadata_is_preserved(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        goal = integration.accept(candidate()).goal
        assert goal is not None
        assert goal.metadata["source"] == "test"

    def test_parent_goal_is_preserved(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        integration.accept(candidate("parent", "Parent"))
        child = integration.accept(candidate("child", "Child", parent_goal_id="parent")).goal
        assert child is not None
        assert child.parent_id == "parent"

    def test_terminal_duplicate_id_is_still_rejected(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        integration.accept(candidate("c1", "First"))
        integration.manager.activate("c1")
        integration.manager.complete("c1")
        result = integration.accept(candidate("c1", "Replacement"))
        assert result.accepted is False
        assert result.decision.reason is GoalAcceptanceReason.DUPLICATE_GOAL_ID

    def test_projection_can_be_disabled(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent, project_to_agent=False)
        result = integration.accept(candidate())
        assert result.projected_to_agent is False
        assert not hasattr(agent, "current_goal")

    def test_agent_goal_manager_is_attached_even_projection_disabled(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent, project_to_agent=False)
        assert agent.goal_manager is integration.manager

    def test_acceptance_result_metadata(self):
        agent = type("Agent", (), {})()
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent)
        result = integration.accept(candidate())
        assert result.metadata["applied"] is True

    def test_rejection_result_metadata(self):
        agent = type("Agent", (), {})()
        engine = GoalAcceptanceEngine(policy=GoalAcceptancePolicy(minimum_proposal_score=0.9))
        integration = AgentGoalAcceptanceIntegrator.for_agent(agent, acceptance_engine=engine)
        result = integration.accept(candidate(proposal_score=0.8))
        assert result.metadata["applied"] is False


class TestResult:
    def test_accepted_property_requires_goal(self):
        decision = GoalAcceptanceDecision(candidate(), True, GoalAcceptanceReason.ACCEPTED)
        result = GoalAcceptanceResult(decision, None, None, False)
        assert result.accepted is False

    def test_metadata_copy(self):
        decision = GoalAcceptanceDecision(candidate(), True, GoalAcceptanceReason.ACCEPTED)
        metadata = {"x": 1}
        result = GoalAcceptanceResult(decision, None, None, False, metadata)
        metadata["y"] = 2
        assert result.metadata == {"x": 1}
