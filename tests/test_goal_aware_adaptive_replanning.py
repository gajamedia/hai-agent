from __future__ import annotations

from dataclasses import replace

import pytest

from agent.llm.context import AgentContext
from agent.llm.goal_aware_adaptive_replanning import (
    GoalAwareAdaptiveReplanningConfigurationError,
    GoalAwareAdaptiveReplanningConsistencyError,
    GoalAwareAdaptiveReplanningController,
    GoalAwareAdaptiveReplanningDecision,
    GoalAwareAdaptiveReplanningError,
    GoalAwareAdaptiveReplanningInputError,
    GoalAwareAdaptiveReplanningProvider,
)
from agent.llm.goal_aware_llm_planning import GoalAwarePlanningContext
from agent.llm.goal_conflict_resolution import (
    GoalConflictRelation,
    GoalConflictResolver,
)
from agent.llm.goal_management import GoalLifecycleManager, GoalStatus, ManagedGoal
from agent.llm.llm_plan_adaptive_replanning import LLMPlanAdaptiveReplanner
from agent.llm.llm_plan_runtime_learning_feedback import LLMPlanRuntimeLearningFeedback
from agent.llm.multi_goal_arbitration import (
    MultiGoalArbitrationCandidate,
    MultiGoalArbitrationDecision,
)
from agent.llm.planning import LLMPlan, LLMPlanStep
from agent.llm.reasoning import LLMReasoning


def goal(gid="g1", description="Reach target", priority=0.8, progress=0.1, status=GoalStatus.PENDING, sequence=1):
    return ManagedGoal(
        goal_id=gid,
        description=description,
        priority=priority,
        progress=progress,
        status=status,
        sequence=sequence,
    )


def plan(goal_text="Reach target", indices=(1, 2, 3), risks=(0.2, 0.3, 0.4)):
    return LLMPlan(
        goal=goal_text,
        rationale="test",
        steps=tuple(
            LLMPlanStep(index=i, action=f"action_{i}", purpose=f"purpose_{i}", risk=r)
            for i, r in zip(indices, risks)
        ),
        confidence=0.9,
    )


def feedback(step=2, *, change_required=False, continue_recommended=True, success=True, progress_delta=0.2, risk_level="low"):
    return LLMPlanRuntimeLearningFeedback(
        step_index=step,
        action=f"action_{step}",
        success=success,
        progress_delta=progress_delta,
        evidence_quality=0.8,
        confidence=0.8,
        diagnosis="ok",
        lesson="keep going",
        next_step="continue",
        change_required=change_required,
        continue_recommended=continue_recommended,
        exploration_recommended=False,
        risk_level=risk_level,
        risk_score=0.25 if risk_level == "low" else 0.5,
        information_gain=0.2,
    )


def arbitration_for(selected="g1", previous="g0", switched=True, preempted=True):
    c = MultiGoalArbitrationCandidate(
        goal_id=selected,
        description="Reach target",
        status=GoalStatus.ACTIVE,
        sequence=1,
        priority=0.9,
        urgency=0.8,
        progress=0.1,
        progress_need=0.9,
        achievement_score=0.2,
        achievement_gap=0.8,
        score=0.9,
        eligible=True,
    )
    return MultiGoalArbitrationDecision(
        selected_goal_id=selected,
        previous_goal_id=previous,
        selected_score=0.9,
        previous_score=0.5,
        score_margin=0.4,
        switched=switched,
        preempted=preempted,
        reason="higher_arbitration_score_preempts_current" if preempted else "goal_switched",
        ranked_goal_ids=(selected,),
        eligible_goal_ids=(selected,),
        candidates=(c,),
    )


class FakeGoalAwarePlanner:
    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def plan_for_goal(self, **kwargs):
        self.calls.append(kwargs)
        return self.output or plan(kwargs["selected_goal"].description, (1, 2), ("low", "medium"))


def make_provider(output=None):
    fake = FakeGoalAwarePlanner(output)
    provider = GoalAwareAdaptiveReplanningProvider(fake)
    return provider, fake


def make_controller(provider, max_replans=3):
    adaptive = LLMPlanAdaptiveReplanner(provider, max_replans=max_replans)
    return GoalAwareAdaptiveReplanningController(adaptive, provider), adaptive


# 1-10: provider contract and state

def test_provider_requires_plan_for_goal():
    with pytest.raises(GoalAwareAdaptiveReplanningConfigurationError):
        GoalAwareAdaptiveReplanningProvider(object())


def test_provider_rejects_noncallable_context_factory():
    with pytest.raises(GoalAwareAdaptiveReplanningConfigurationError):
        GoalAwareAdaptiveReplanningProvider(FakeGoalAwarePlanner(), context_factory=object())


def test_provider_rejects_noncallable_reasoning_factory():
    with pytest.raises(GoalAwareAdaptiveReplanningConfigurationError):
        GoalAwareAdaptiveReplanningProvider(FakeGoalAwarePlanner(), reasoning_factory=object())


def test_provider_rejects_noncallable_risk_resolver():
    with pytest.raises(GoalAwareAdaptiveReplanningConfigurationError):
        GoalAwareAdaptiveReplanningProvider(FakeGoalAwarePlanner(), risk_resolver=object())


def test_provider_requires_context_before_call():
    provider, _ = make_provider()
    adaptive = LLMPlanAdaptiveReplanner(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningInputError):
        adaptive.replan(plan(), feedback=feedback(change_required=True), successful_step_indices=(1,), goal="Reach target")


def test_provider_set_context_and_property():
    provider, _ = make_provider()
    planning = GoalAwarePlanningContext(selected_goal=goal())
    provider.set_planning_context(planning)
    assert provider.planning_context == planning


def test_provider_set_coordination_preserves_multiple_goals():
    provider, fake = make_provider()
    g1 = goal()
    g2 = goal("g2", "Other target", sequence=2)
    planning = GoalAwarePlanningContext(selected_goal=g1, available_goal_ids=("g1", "g2"))
    provider.set_coordination_context(planning, available_goals=(g1, g2))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    replacement = plan("Reach target", (4, 5), ("low", "medium"))
    provider.last_plan = None
    provider.goal_planner.output = replacement
    result = adaptive.replan(plan(), feedback=feedback(change_required=True), successful_step_indices=(1,), goal=g1.description, next_step_index=2)
    assert result.replacement_plan is not None
    assert tuple(g.goal_id for g in fake.calls[-1]["available_goals"]) == ("g1", "g2")


def test_provider_resets_state():
    provider, _ = make_provider()
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    provider.reset()
    assert provider.planning_context is None
    assert provider.last_plan is None


def test_provider_returns_llm_plan_and_trace():
    provider, fake = make_provider(plan("Reach target", (1,), (0.2,)))
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    result = adaptive.replan(plan(), feedback=feedback(change_required=True), goal="Reach target", next_step_index=1)
    assert result.replacement_plan is not None
    assert provider.last_request is not None
    assert provider.last_context is not None
    assert provider.last_reasoning is not None
    assert provider.last_plan.goal == result.replacement_plan.goal
    assert tuple(step.action for step in provider.last_plan.steps) == tuple(step.action for step in result.replacement_plan.steps)
    assert len(fake.calls) == 1


def test_provider_normalizes_text_risk():
    provider, _ = make_provider(plan("Reach target", (1,), ("high",)))
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    result = adaptive.replan(plan(), feedback=feedback(change_required=True), goal="Reach target", next_step_index=1)
    assert result.replacement_plan.steps[0].risk == 0.75


def test_provider_rejects_goal_mismatch():
    provider, _ = make_provider()
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningConsistencyError):
        adaptive.replan(plan(), feedback=feedback(change_required=True), goal="Wrong goal", next_step_index=1)


# 11-20: controller evaluation

def test_controller_requires_replanner_api():
    provider, _ = make_provider()
    with pytest.raises(GoalAwareAdaptiveReplanningConfigurationError):
        GoalAwareAdaptiveReplanningController(object(), provider)


def test_controller_requires_provider_type():
    with pytest.raises(GoalAwareAdaptiveReplanningConfigurationError):
        GoalAwareAdaptiveReplanningController(LLMPlanAdaptiveReplanner(lambda r: plan()), object())


def test_evaluate_no_trigger():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    decision = controller.evaluate(plan(), goal(), feedback=feedback())
    assert decision.should_replan is False
    assert decision.reason == "no_replan_trigger"


def test_evaluate_runtime_trigger():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    decision = controller.evaluate(plan(), goal(), feedback=feedback(change_required=True, continue_recommended=False, success=False, progress_delta=-0.1, risk_level="high"))
    assert decision.should_replan is True
    assert decision.runtime_replan_requested is True
    assert "runtime_feedback_trigger" in decision.reason


def test_evaluate_goal_description_change():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    decision = controller.evaluate(plan("Old target"), goal(description="New target"), original_goal_id="g1")
    assert decision.should_replan is True
    assert decision.goal_changed is True
    assert "goal_changed" in decision.reason


def test_evaluate_goal_id_change_same_description():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    decision = controller.evaluate(plan("Reach target"), goal("g2", "Reach target"), original_goal_id="g1")
    assert decision.goal_changed is True


def test_evaluate_no_change_when_id_and_description_match():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    decision = controller.evaluate(plan(), goal("g1", "Reach target"), original_goal_id="g1")
    assert decision.goal_changed is False


def test_evaluate_arbitration_preemption():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    decision = controller.evaluate(plan(), goal(), arbitration=arbitration_for())
    assert decision.should_replan is True
    assert decision.arbitration_preempted is True


def test_evaluate_arbitration_switch_without_preemption():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    decision = controller.evaluate(plan(), goal(), arbitration=arbitration_for(switched=True, preempted=False))
    assert decision.should_replan is True
    assert decision.arbitration_switched is True
    assert decision.arbitration_preempted is False


def test_evaluate_invalid_arbitration_type():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningInputError):
        controller.evaluate(plan(), goal(), arbitration=object())


def test_evaluate_invalid_conflict_type():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningInputError):
        controller.evaluate(plan(), goal(), conflict_resolution=object())


# 21-30: conflict/feedback decisions and goal normalization

def test_terminal_selected_goal_rejected():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningConsistencyError):
        controller.evaluate(plan(), goal(status=GoalStatus.COMPLETED))


def test_conflict_resolution_requires_replan_when_selected_is_blocked():
    manager = GoalLifecycleManager()
    g1 = manager.add_goal("g1", "Reach target", priority=0.8)
    g2 = manager.add_goal("g2", "Other target", priority=0.7)
    resolver = GoalConflictResolver()
    conflict = resolver.resolve((g1, g2), relations=(GoalConflictRelation("g1", "g2"),), blocked_goal_ids=("g1",))
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningConsistencyError):
        controller.evaluate(plan(), g1, conflict_resolution=conflict)


def test_conflict_resolution_for_nonwinner_requires_replan():
    manager = GoalLifecycleManager()
    g1 = manager.add_goal("g1", "Reach target", priority=0.1)
    g2 = manager.add_goal("g2", "Other target", priority=0.9)
    resolver = GoalConflictResolver()
    conflict = resolver.resolve((g1, g2), relations=(GoalConflictRelation("g1", "g2"),))
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    decision = controller.evaluate(plan(), g1, conflict_resolution=conflict)
    assert decision.conflict_requires_replanning is True
    assert decision.should_replan is True


def test_no_feedback_goal_change_is_replannable():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan("Old target"), goal("g2", "New target"), original_goal_id="g1", next_step_index=1)
    assert result.replanned is True
    assert result.effective_feedback is not None
    assert result.effective_feedback.metadata["synthetic"] is True


def test_runtime_feedback_is_not_replaced_when_it_triggers():
    fb = feedback(change_required=True, continue_recommended=False, success=False, progress_delta=-0.1)
    provider, _ = make_provider(plan("Reach target", (4, 5), (0.2, 0.3)))
    controller, _ = make_controller(provider)
    result = controller.replan(plan(), goal(), feedback=fb, successful_step_indices=(1,), next_step_index=2)
    assert result.effective_feedback is fb


def test_no_trigger_returns_no_adaptive_result():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan(), goal(), feedback=feedback(), original_goal_id="g1")
    assert result.replanned is False
    assert result.adaptive_result is None


def test_result_properties_follow_adaptive_result():
    provider, _ = make_provider(plan("Reach target", (1,), (0.2,)))
    controller, _ = make_controller(provider)
    result = controller.replan(plan(), goal(), feedback=feedback(change_required=True), next_step_index=1)
    assert result.replanned is True
    assert result.replacement_plan is result.adaptive_result.replacement_plan
    assert result.validation is result.adaptive_result.validation
    assert result.llm_plan_requested is True


def test_decision_is_frozen_like():
    decision = GoalAwareAdaptiveReplanningDecision(
        should_replan=False,
        reason="none",
        active_goal_id="g1",
        original_goal="Reach target",
        goal_changed=False,
        arbitration_switched=False,
        arbitration_preempted=False,
        conflict_requires_replanning=False,
        runtime_replan_requested=False,
        unresolved_conflict_count=0,
    )
    with pytest.raises((AttributeError, TypeError)):
        decision.reason = "changed"


def test_decision_rejects_empty_goal_id():
    with pytest.raises(GoalAwareAdaptiveReplanningInputError):
        GoalAwareAdaptiveReplanningDecision(False, "none", "", "goal", False, False, False, False, False, 0)


def test_decision_rejects_negative_unresolved_conflict_count():
    with pytest.raises(GoalAwareAdaptiveReplanningInputError):
        GoalAwareAdaptiveReplanningDecision(False, "none", "g1", "goal", False, False, False, False, False, -1)


def test_result_tracks_planning_context():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan("Old target"), goal("g2", "New target"), original_goal_id="g1", available_goals=(goal("g2", "New target"),))
    assert result.planning.selected_goal.goal_id == "g2"
    assert result.planning.available_goal_ids == ("g2",)


# 31-40: end-to-end provider/controller semantics

def test_successful_steps_are_preserved():
    provider, _ = make_provider(plan("Reach target", (10, 11), (0.2, 0.3)))
    controller, _ = make_controller(provider)
    result = controller.replan(plan("Old target", (1, 2, 3), (0.2, 0.3, 0.4)), goal(), original_goal_id="g1", successful_step_indices=(1, 2), feedback=feedback(change_required=True), next_step_index=4)
    assert result.adaptive_result.preserved_step_indices == (1, 2)


def test_replacement_steps_are_rebased():
    provider, _ = make_provider(plan("Reach target", (99, 100), (0.2, 0.3)))
    controller, _ = make_controller(provider)
    result = controller.replan(plan(), goal(), feedback=feedback(change_required=True), successful_step_indices=(1,), next_step_index=5)
    assert result.replacement_plan.actions == ("action_99", "action_100")
    assert tuple(step.index for step in result.replacement_plan.steps) == (5, 6)


def test_multiple_goals_flow_to_planner():
    provider, fake = make_provider()
    controller, _ = make_controller(provider)
    g1 = goal("g1", "Reach target")
    g2 = goal("g2", "Other target", sequence=2)
    controller.replan(plan(), g1, available_goals=(g1, g2), feedback=feedback(change_required=True), next_step_index=1)
    assert tuple(g.goal_id for g in fake.calls[-1]["available_goals"]) == ("g1", "g2")


def test_arbitration_flows_to_planner():
    provider, fake = make_provider()
    controller, _ = make_controller(provider)
    arb = arbitration_for(switched=True, preempted=True)
    controller.replan(plan(), goal(), arbitration=arb, feedback=feedback(change_required=True), next_step_index=1)
    assert fake.calls[-1]["arbitration"] is arb


def test_conflict_flows_to_planner():
    manager = GoalLifecycleManager()
    g1 = manager.add_goal("g1", "Reach target", priority=0.9)
    g2 = manager.add_goal("g2", "Other target", priority=0.1)
    resolver = GoalConflictResolver()
    conflict = resolver.resolve((g1, g2), relations=(GoalConflictRelation("g1", "g2"),))
    provider, fake = make_provider()
    controller, _ = make_controller(provider)
    controller.replan(plan(), g1, conflict_resolution=conflict, feedback=feedback(change_required=True), next_step_index=1)
    assert fake.calls[-1]["conflict_resolution"] is conflict


def test_custom_context_factory_is_used():
    seen = {}
    def context_factory(request, planning):
        seen["request"] = request
        return AgentContext(goal={"custom": planning.selected_goal.goal_id})
    provider, _ = make_provider()
    provider.context_factory = context_factory
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    adaptive.replan(plan(), feedback=feedback(change_required=True), goal="Reach target", next_step_index=1)
    assert seen["request"].goal == "Reach target"
    assert provider.last_context.goal == {"custom": "g1"}


def test_custom_reasoning_factory_is_used():
    def reasoning_factory(request, planning):
        return LLMReasoning(summary=f"custom:{planning.selected_goal.goal_id}")
    provider, _ = make_provider()
    provider.reasoning_factory = reasoning_factory
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    adaptive.replan(plan(), feedback=feedback(change_required=True), goal="Reach target", next_step_index=1)
    assert provider.last_reasoning.summary == "custom:g1"


def test_context_factory_wrong_type_rejected():
    def context_factory(request, planning):
        return object()
    provider, _ = make_provider()
    provider.context_factory = context_factory
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningInputError):
        adaptive.replan(plan(), feedback=feedback(change_required=True), goal="Reach target", next_step_index=1)


def test_reasoning_factory_wrong_type_rejected():
    def reasoning_factory(request, planning):
        return object()
    provider, _ = make_provider()
    provider.reasoning_factory = reasoning_factory
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningInputError):
        adaptive.replan(plan(), feedback=feedback(change_required=True), goal="Reach target", next_step_index=1)


def test_planner_output_wrong_type_rejected():
    provider, fake = make_provider()
    fake.output = object()
    provider.set_planning_context(GoalAwarePlanningContext(selected_goal=goal()))
    adaptive = LLMPlanAdaptiveReplanner(provider)
    with pytest.raises(GoalAwareAdaptiveReplanningError):
        adaptive.replan(plan(), feedback=feedback(change_required=True), goal="Reach target", next_step_index=1)


def test_reset_clears_controller_result():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    controller.replan(plan("Old target"), goal("g2", "New target"), original_goal_id="g1", next_step_index=1)
    assert controller.last_result is not None
    controller.reset()
    assert controller.last_result is None

# 41-52: edge cases and metadata

def test_mapping_goal_is_accepted():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan(), {"goal_id": "g1", "description": "Reach target"}, feedback=feedback(change_required=True), next_step_index=1)
    assert result.planning.selected_goal.goal_id == "g1"


def test_invalid_goal_mapping_rejected():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    with pytest.raises((KeyError, GoalAwareAdaptiveReplanningInputError)):
        controller.evaluate(plan(), {"description": "missing id"})


def test_available_goals_auto_include_selected_goal():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan("Old target"), goal("g2", "New target"), available_goals=(goal("g3", "Other", sequence=3),), original_goal_id="g1", next_step_index=1)
    assert "g2" in result.planning.available_goal_ids


def test_effective_feedback_synthetic_step_uses_failure_index():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan("Old target"), goal("g2", "New target"), original_goal_id="g1", failed_step_index=3, next_step_index=4)
    assert result.effective_feedback.step_index == 3


def test_metadata_contains_stage_marker():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan("Old target"), goal("g2", "New target"), original_goal_id="g1", metadata={"ticket": "ABC"}, next_step_index=1)
    assert result.metadata["stage"] == "25.7"


def test_provider_request_metadata_has_stage_marker():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    controller.replan(plan("Old target"), goal("g2", "New target"), original_goal_id="g1", next_step_index=1)
    assert provider.last_request.metadata["integration_stage"] == "25.7"


def test_goal_change_reason_is_stable():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    d1 = controller.evaluate(plan("Old"), goal("g2", "New"), original_goal_id="g1")
    d2 = controller.evaluate(plan("Old"), goal("g2", "New"), original_goal_id="g1")
    assert d1.reason == d2.reason == "goal_changed"


def test_combined_reasons_are_pipe_joined():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    d = controller.evaluate(plan("Old"), goal("g2", "New"), original_goal_id="g1", arbitration=arbitration_for(), feedback=feedback(change_required=True, continue_recommended=False, success=False, progress_delta=-0.1, risk_level="high"))
    assert d.reason == "goal_changed|goal_preempted|runtime_feedback_trigger"


def test_runtime_feedback_can_replan_without_goal_change():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan(), goal(), feedback=feedback(change_required=True, continue_recommended=False, success=False, progress_delta=-0.1, risk_level="high"), next_step_index=1)
    assert result.replanned is True
    assert result.decision.goal_changed is False


def test_positive_feedback_does_not_replan_without_goal_change():
    provider, fake = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan(), goal(), feedback=feedback(), next_step_index=1)
    assert result.adaptive_result is None
    assert not fake.calls


def test_synthetic_feedback_marks_goal_id():
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan("Old"), goal("g2", "New"), original_goal_id="g1", next_step_index=1)
    assert result.effective_feedback.metadata["goal_id"] == "g2"


def test_selected_goal_is_passed_to_planner():
    provider, fake = make_provider()
    controller, _ = make_controller(provider)
    controller.replan(plan("Old"), goal("g2", "New"), original_goal_id="g1", next_step_index=1)
    assert fake.calls[-1]["selected_goal"].goal_id == "g2"


def test_controller_result_stores_original_feedback():
    fb = feedback(change_required=True)
    provider, _ = make_provider()
    controller, _ = make_controller(provider)
    result = controller.replan(plan(), goal(), feedback=fb, next_step_index=1)
    assert result.feedback is fb
