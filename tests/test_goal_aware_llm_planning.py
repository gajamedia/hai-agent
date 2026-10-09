from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from agent.llm.contracts import LLMRequest, LLMResponse
from agent.llm.context import AgentContext
from agent.llm.goal_aware_llm_planning import (
    GoalAwareLLMPlanningConfigurationError,
    GoalAwareLLMPlanningConsistencyError,
    GoalAwareLLMPlanningError,
    GoalAwareLLMPlanningInputError,
    GoalAwareLLMPlanningRequest,
    GoalAwareLLMPlanner,
    GoalAwarePlanningContext,
    GoalAwarePlanningContextBuilder,
)
from agent.llm.goal_conflict_resolution import (
    GoalConflictRelation,
    GoalConflictResolver,
)
from agent.llm.goal_management import GoalStatus, ManagedGoal
from agent.llm.multi_goal_arbitration import MultiGoalArbiter
from agent.llm.planning import LLMPlan, LLMPlanner
from agent.llm.reasoning import LLMReasoning
from agent.llm.service import LLMService


class FakeProvider:
    name = "fake"

    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=self.content,
            provider=self.name,
            model="fake-model",
        )


def plan_json(goal: str, *, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "goal": goal,
            "rationale": "goal-aware test plan",
            "steps": [
                {
                    "index": 1,
                    "action": "inspect",
                    "purpose": "inspect relevant state",
                    "preconditions": [],
                    "expected_observation": "state is known",
                    "success_criteria": "inspection completes",
                    "risk": "low",
                }
            ],
            "assumptions": [],
            "constraints": [],
            "stop_conditions": [],
            "confidence": confidence,
        }
    )


def goal_a(priority: float = 0.8, progress: float = 0.1) -> ManagedGoal:
    return ManagedGoal(
        goal_id="g-a",
        description="finish order",
        priority=priority,
        progress=progress,
        sequence=1,
    )


def goal_b(priority: float = 0.4, progress: float = 0.2) -> ManagedGoal:
    return ManagedGoal(
        goal_id="g-b",
        description="prepare report",
        priority=priority,
        progress=progress,
        sequence=2,
    )


def reasoning() -> LLMReasoning:
    return LLMReasoning(summary="Use the selected goal and current evidence.")


def context() -> AgentContext:
    return AgentContext(
        goal={"legacy": "goal"},
        state={"mode": "ready"},
        world_model={"location": "desk"},
        relevant_memory=("memory-1",),
        strategy={"name": "safe"},
        self_awareness={"confidence": 0.7},
        calibration={"confidence": 0.6},
        metacognition={"lesson": "observe"},
        metadata={"source": "test", "goal_management": {"legacy": True}},
    )


def arbitration_for(selected: ManagedGoal, other: ManagedGoal | None = None):
    goals = [selected] + ([other] if other is not None else [])
    return MultiGoalArbiter().arbitrate(
        goals,
        urgency={g.goal_id: 0.8 for g in goals},
    )


def conflict_for(selected: ManagedGoal, other: ManagedGoal):
    arbitration = arbitration_for(selected, other)
    return GoalConflictResolver().resolve(
        [selected, other],
        relations=[GoalConflictRelation(selected.goal_id, other.goal_id, severity=0.8)],
        arbitration=arbitration,
        urgency={selected.goal_id: 0.8, other.goal_id: 0.4},
    )


def make_adapter(goal: ManagedGoal | None = None):
    selected = goal or goal_a()
    provider = FakeProvider(plan_json(selected.description))
    llm_service = LLMService(provider)
    planner = LLMPlanner(llm_service)
    adapter = GoalAwareLLMPlanner(planner)
    return adapter, provider, llm_service, planner


def test_base_error_hierarchy():
    assert issubclass(GoalAwareLLMPlanningConfigurationError, GoalAwareLLMPlanningError)
    assert issubclass(GoalAwareLLMPlanningInputError, GoalAwareLLMPlanningError)
    assert issubclass(GoalAwareLLMPlanningConsistencyError, GoalAwareLLMPlanningError)


def test_planning_context_requires_managed_goal():
    with pytest.raises(GoalAwareLLMPlanningInputError):
        GoalAwarePlanningContext(selected_goal="g-a")  # type: ignore[arg-type]


def test_planning_context_normalizes_ids_and_score():
    item = GoalAwarePlanningContext(
        selected_goal=goal_a(),
        available_goal_ids=("g-a", "g-a", " g-b "),
        competing_goal_ids=("g-b", "g-b"),
        deferred_goal_ids=("g-c",),
        blocked_goal_ids=("g-d",),
        unresolved_conflict_ids=("c1",),
        arbitration_score=0.75,
    )
    assert item.available_goal_ids == ("g-a", "g-b")
    assert item.competing_goal_ids == ("g-b",)
    assert item.has_conflict is True


def test_planning_context_rejects_out_of_range_score():
    with pytest.raises(GoalAwareLLMPlanningInputError):
        GoalAwarePlanningContext(selected_goal=goal_a(), arbitration_score=1.1)


def test_planning_context_rejects_negative_conflict_count():
    with pytest.raises(GoalAwareLLMPlanningInputError):
        GoalAwarePlanningContext(selected_goal=goal_a(), conflict_count=-1)


def test_planning_context_to_dict_is_json_friendly():
    item = GoalAwarePlanningContext(selected_goal=goal_a(), metadata={"x": 1})
    payload = item.to_dict()
    assert payload["selected_goal"]["goal_id"] == "g-a"
    assert payload["selected_goal"]["status"] == GoalStatus.PENDING.value
    assert payload["metadata"] == {"x": 1}


def test_request_requires_reasoning():
    planning = GoalAwarePlanningContext(selected_goal=goal_a())
    with pytest.raises(GoalAwareLLMPlanningInputError):
        GoalAwareLLMPlanningRequest("bad", context(), planning)  # type: ignore[arg-type]


def test_request_requires_agent_context():
    planning = GoalAwarePlanningContext(selected_goal=goal_a())
    with pytest.raises(GoalAwareLLMPlanningInputError):
        GoalAwareLLMPlanningRequest(reasoning(), {}, planning)  # type: ignore[arg-type]


def test_request_normalizes_metadata():
    planning = GoalAwarePlanningContext(selected_goal=goal_a())
    request = GoalAwareLLMPlanningRequest(reasoning(), context(), planning, {"k": 1})
    assert request.metadata == {"k": 1}


def test_builder_uses_selected_goal_when_available_goals_empty():
    result = GoalAwarePlanningContextBuilder().build(goal_a())
    assert result.selected_goal.goal_id == "g-a"
    assert result.available_goal_ids == ("g-a",)


def test_builder_accepts_goal_mapping():
    result = GoalAwarePlanningContextBuilder().build(
        {"goal_id": "g-a", "description": "finish order", "priority": 0.9}
    )
    assert result.selected_goal.priority == 0.9


def test_builder_accepts_status_string_in_mapping():
    result = GoalAwarePlanningContextBuilder().build(
        {"goal_id": "g-a", "description": "finish order", "status": "active"}
    )
    assert result.selected_goal.status is GoalStatus.ACTIVE


def test_builder_rejects_invalid_goal_mapping():
    with pytest.raises(GoalAwareLLMPlanningInputError):
        GoalAwarePlanningContextBuilder().build({"goal_id": "g-a"})


def test_builder_available_goal_ids_preserve_sequence_order():
    result = GoalAwarePlanningContextBuilder().build(
        goal_a(), available_goals=[goal_b(), goal_a()]
    )
    assert result.available_goal_ids == ("g-b", "g-a")


def test_builder_records_arbitration():
    selected = goal_a()
    other = goal_b()
    arb = arbitration_for(selected, other)
    result = GoalAwarePlanningContextBuilder().build(
        selected, available_goals=[selected, other], arbitration=arb
    )
    assert result.arbitration_reason == arb.reason
    assert result.arbitration_score == arb.selected_score
    assert result.competing_goal_ids == ("g-b",)


def test_builder_rejects_arbitration_selected_goal_mismatch():
    arb = arbitration_for(goal_b(priority=0.95), goal_a(priority=0.2))
    with pytest.raises(GoalAwareLLMPlanningConsistencyError):
        GoalAwarePlanningContextBuilder().build(goal_a(), arbitration=arb)


def test_builder_records_conflict_resolution():
    selected = goal_a(priority=0.9)
    other = goal_b(priority=0.2)
    arb = arbitration_for(selected, other)
    conflict = GoalConflictResolver().resolve(
        [selected, other],
        relations=[GoalConflictRelation("g-a", "g-b", severity=0.8)],
        arbitration=arb,
        urgency={"g-a": 0.8, "g-b": 0.3},
    )
    result = GoalAwarePlanningContextBuilder().build(
        selected,
        available_goals=[selected, other],
        arbitration=arb,
        conflict_resolution=conflict,
    )
    assert result.conflict_count == 1
    assert "g-b" in result.deferred_goal_ids


def test_builder_rejects_blocked_selected_goal():
    selected = goal_a()
    other = goal_b()
    conflict = GoalConflictResolver().resolve(
        [selected, other],
        relations=[GoalConflictRelation("g-a", "g-b")],
        blocked_goal_ids=["g-a"],
    )
    with pytest.raises(GoalAwareLLMPlanningConsistencyError):
        GoalAwarePlanningContextBuilder().build(selected, conflict_resolution=conflict)


def test_builder_rejects_conflicting_loser_selected_goal():
    selected = goal_b(priority=0.2)
    other = goal_a(priority=0.9)
    conflict = conflict_for(other, selected)
    with pytest.raises(GoalAwareLLMPlanningConsistencyError):
        GoalAwarePlanningContextBuilder().build(selected, conflict_resolution=conflict)


def test_builder_merges_competing_and_deferred_without_duplicates():
    selected = goal_a(priority=0.9)
    other = goal_b(priority=0.5)
    arb = arbitration_for(selected, other)
    conflict = conflict_for(selected, other)
    result = GoalAwarePlanningContextBuilder().build(
        selected,
        available_goals=[selected, other],
        arbitration=arb,
        conflict_resolution=conflict,
    )
    assert len(result.competing_goal_ids) == len(set(result.competing_goal_ids))
    assert "g-b" in result.competing_goal_ids


def test_enrich_context_replaces_goal_with_selected_goal_payload():
    planning = GoalAwarePlanningContextBuilder().build(goal_a())
    enriched = GoalAwarePlanningContextBuilder().enrich_context(context(), planning)
    assert enriched.goal["goal_id"] == "g-a"
    assert enriched.goal["description"] == "finish order"


def test_enrich_context_preserves_non_goal_context():
    original = context()
    planning = GoalAwarePlanningContextBuilder().build(goal_a())
    enriched = GoalAwarePlanningContextBuilder().enrich_context(original, planning)
    assert enriched.state == original.state
    assert enriched.world_model == original.world_model
    assert enriched.relevant_memory == original.relevant_memory
    assert enriched.strategy == original.strategy
    assert enriched.self_awareness == original.self_awareness


def test_enrich_context_adds_goal_management_metadata():
    planning = GoalAwarePlanningContextBuilder().build(goal_a(), metadata={"custom": "x"})
    enriched = GoalAwarePlanningContextBuilder().enrich_context(context(), planning)
    management = enriched.metadata["goal_management"]
    assert management["selected_goal"]["goal_id"] == "g-a"
    assert enriched.metadata["source"] == "test"


def test_enrich_context_overwrites_only_goal_management_section():
    original = context()
    planning = GoalAwarePlanningContextBuilder().build(goal_a())
    enriched = GoalAwarePlanningContextBuilder().enrich_context(original, planning)
    assert enriched.metadata["source"] == original.metadata["source"]
    assert enriched.metadata["goal_management"]["selected_goal"]["goal_id"] == "g-a"


def test_build_request_returns_goal_aware_request():
    adapter, *_ = make_adapter()
    request = adapter.build_request(
        reasoning=reasoning(), context=context(), selected_goal=goal_a()
    )
    assert isinstance(request, GoalAwareLLMPlanningRequest)
    assert request.planning.selected_goal.goal_id == "g-a"


def test_build_request_passes_arbitration_and_conflicts():
    adapter, *_ = make_adapter()
    selected = goal_a(priority=0.9)
    other = goal_b(priority=0.2)
    arb = arbitration_for(selected, other)
    conflict = conflict_for(selected, other)
    request = adapter.build_request(
        reasoning=reasoning(),
        context=context(),
        selected_goal=selected,
        available_goals=[selected, other],
        arbitration=arb,
        conflict_resolution=conflict,
    )
    assert request.planning.arbitration_score == arb.selected_score
    assert request.planning.conflict_count == 1


def test_planner_rejects_non_planner():
    with pytest.raises(GoalAwareLLMPlanningConfigurationError):
        GoalAwareLLMPlanner(object())  # type: ignore[arg-type]


def test_plan_calls_underlying_canonical_planner_once():
    adapter, provider, *_ = make_adapter()
    request = adapter.build_request(
        reasoning=reasoning(), context=context(), selected_goal=goal_a()
    )
    result = adapter.plan(request)
    assert len(provider.requests) == 1
    assert isinstance(result, LLMPlan)


def test_plan_prompt_contains_selected_goal():
    adapter, provider, *_ = make_adapter()
    request = adapter.build_request(
        reasoning=reasoning(), context=context(), selected_goal=goal_a()
    )
    adapter.plan(request)
    prompt = provider.requests[-1].messages[-1].content
    assert "finish order" in prompt
    assert "g-a" in prompt


def test_plan_metadata_contains_goal_management():
    adapter, provider, *_ = make_adapter()
    request = adapter.build_request(
        reasoning=reasoning(), context=context(), selected_goal=goal_a()
    )
    adapter.plan(request)
    metadata = provider.requests[-1].metadata
    assert metadata["operation"] == "goal_aware_planning"
    assert metadata["goal_id"] == "g-a"
    assert metadata["goal_progress"] == 0.1


def test_plan_goal_mismatch_is_rejected():
    selected = goal_a()
    provider = FakeProvider(plan_json("wrong goal"))
    adapter = GoalAwareLLMPlanner(LLMPlanner(LLMService(provider)))
    request = adapter.build_request(
        reasoning=reasoning(), context=context(), selected_goal=selected
    )
    with pytest.raises(GoalAwareLLMPlanningConsistencyError):
        adapter.plan(request)


def test_plan_goal_match_is_trimmed_consistently():
    selected = ManagedGoal("g-a", "finish order")
    provider = FakeProvider(plan_json("finish order"))
    adapter = GoalAwareLLMPlanner(LLMPlanner(LLMService(provider)))
    result = adapter.plan_for_goal(
        reasoning=reasoning(), context=context(), selected_goal=selected
    )
    assert result.goal == "finish order"


def test_plan_for_goal_is_convenience_method():
    adapter, provider, *_ = make_adapter()
    result = adapter.plan_for_goal(
        reasoning=reasoning(), context=context(), selected_goal=goal_a()
    )
    assert result.goal == "finish order"
    assert provider.requests


def test_last_plan_tracks_successful_plan():
    adapter, *_ = make_adapter()
    result = adapter.plan_for_goal(
        reasoning=reasoning(), context=context(), selected_goal=goal_a()
    )
    assert adapter.last_plan is result


def test_last_response_delegates_to_underlying_planner():
    adapter, *_ = make_adapter()
    assert adapter.last_response is None
    adapter.plan_for_goal(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    assert adapter.last_response is not None
    assert adapter.last_response.model == "fake-model"


def test_clear_trace_clears_adapter_and_underlying_planner():
    adapter, *_ = make_adapter()
    adapter.plan_for_goal(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    adapter.clear_trace()
    assert adapter.last_plan is None
    assert adapter.last_response is None


def test_custom_context_builder_is_used():
    class SpyBuilder(GoalAwarePlanningContextBuilder):
        def __init__(self) -> None:
            self.calls = 0

        def build(self, *args: Any, **kwargs: Any):
            self.calls += 1
            return super().build(*args, **kwargs)

    builder = SpyBuilder()
    adapter, *_ = make_adapter()
    adapter.context_builder = builder
    adapter.build_request(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    assert builder.calls == 1


def test_mapping_goal_is_used_for_planning():
    adapter, provider, *_ = make_adapter()
    adapter.plan_for_goal(
        reasoning=reasoning(),
        context=context(),
        selected_goal={"goal_id": "g-a", "description": "finish order", "priority": 0.7},
    )
    assert provider.requests


def test_available_goals_are_exposed_in_prompt():
    adapter, provider, *_ = make_adapter()
    adapter.plan_for_goal(
        reasoning=reasoning(),
        context=context(),
        selected_goal=goal_a(),
        available_goals=[goal_a(), goal_b()],
    )
    prompt = provider.requests[-1].messages[-1].content
    assert "g-b" in prompt
    assert "prepare report" not in prompt or "g-b" in prompt


def test_arbitration_switch_metadata_is_exposed():
    selected = goal_a(priority=0.9)
    other = goal_b(priority=0.2)
    arb = MultiGoalArbiter().arbitrate(
        [selected, other], current_goal_id=other.goal_id, urgency={"g-a": 1.0, "g-b": 0.0}
    )
    adapter, provider, *_ = make_adapter(selected)
    adapter.plan_for_goal(
        reasoning=reasoning(),
        context=context(),
        selected_goal=selected,
        available_goals=[selected, other],
        arbitration=arb,
    )
    metadata = provider.requests[-1].metadata["goal_management"]
    assert metadata["arbitration_switched"] is True


def test_conflict_count_is_exposed_to_planner():
    selected = goal_a(priority=0.9)
    other = goal_b(priority=0.2)
    conflict = conflict_for(selected, other)
    adapter, provider, *_ = make_adapter(selected)
    adapter.plan_for_goal(
        reasoning=reasoning(),
        context=context(),
        selected_goal=selected,
        available_goals=[selected, other],
        conflict_resolution=conflict,
    )
    prompt = provider.requests[-1].messages[-1].content
    assert "conflict_count" in prompt


def test_blocked_candidate_is_exposed_to_planner():
    selected = goal_a(priority=0.9)
    other = goal_b(priority=0.2)
    conflict = GoalConflictResolver().resolve(
        [selected, other],
        relations=[GoalConflictRelation("g-a", "g-b")],
        blocked_goal_ids=["g-b"],
    )
    adapter, provider, *_ = make_adapter(selected)
    adapter.plan_for_goal(
        reasoning=reasoning(),
        context=context(),
        selected_goal=selected,
        available_goals=[selected, other],
        conflict_resolution=conflict,
    )
    management = provider.requests[-1].metadata["goal_management"]
    assert "g-b" in management["blocked_goal_ids"]


def test_no_conflict_has_zero_conflict_count():
    planning = GoalAwarePlanningContextBuilder().build(goal_a())
    assert planning.conflict_count == 0
    assert planning.has_conflict is False


def test_enriched_context_is_new_immutable_snapshot():
    original = context()
    planning = GoalAwarePlanningContextBuilder().build(goal_a())
    enriched = GoalAwarePlanningContextBuilder().enrich_context(original, planning)
    assert enriched is not original
    assert original.goal == {"legacy": "goal"}


def test_planning_context_metadata_is_copied():
    source = {"nested": {"value": 1}}
    planning = GoalAwarePlanningContextBuilder().build(goal_a(), metadata=source)
    assert planning.metadata == source
    assert planning.metadata is not source


def test_request_metadata_is_copied():
    source = {"x": 1}
    request = GoalAwareLLMPlanningRequest(
        reasoning(), context(), GoalAwarePlanningContext(selected_goal=goal_a()), source
    )
    assert request.metadata == source
    assert request.metadata is not source


def test_plan_preserves_underlying_planner_configuration():
    provider = FakeProvider(plan_json("finish order"))
    planner = LLMPlanner(LLMService(provider), temperature=0.2, max_tokens=321, max_steps=4)
    adapter = GoalAwareLLMPlanner(planner)
    adapter.plan_for_goal(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    request = provider.requests[-1]
    assert request.temperature == 0.2
    assert request.max_tokens == 321


def test_planner_does_not_execute_actions():
    adapter, provider, *_ = make_adapter()
    adapter.plan_for_goal(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    content = provider.requests[-1].messages[-1].content.lower()
    assert "execute" in content
    assert provider.name == "fake"


def test_status_terminal_is_preserved_in_payload():
    selected = replace(goal_a(), status=GoalStatus.PAUSED)
    planning = GoalAwarePlanningContextBuilder().build(selected)
    assert planning.to_dict()["selected_goal"]["status"] == "paused"


def test_goal_metadata_is_preserved():
    selected = ManagedGoal("g-a", "finish order", metadata={"domain": "sales"})
    planning = GoalAwarePlanningContextBuilder().build(selected)
    assert planning.to_dict()["selected_goal"]["metadata"] == {"domain": "sales"}


def test_unresolved_conflict_sets_has_conflict():
    planning = GoalAwarePlanningContext(
        selected_goal=goal_a(), unresolved_conflict_ids=("c1",)
    )
    assert planning.has_conflict is True


def test_arbitration_without_competing_goal_is_valid():
    selected = goal_a()
    arb = arbitration_for(selected)
    planning = GoalAwarePlanningContextBuilder().build(selected, arbitration=arb)
    assert planning.competing_goal_ids == ()


def test_planning_request_holds_enriched_context():
    adapter, *_ = make_adapter()
    request = adapter.build_request(
        reasoning=reasoning(), context=context(), selected_goal=goal_a()
    )
    assert request.context.goal["management"]["conflict_count"] == 0


def test_metadata_override_can_add_request_level_fields():
    adapter, provider, *_ = make_adapter()
    adapter.plan_for_goal(
        reasoning=reasoning(), context=context(), selected_goal=goal_a(), metadata={"trace_id": "abc"}
    )
    assert provider.requests[-1].metadata["trace_id"] == "abc"


def test_goal_management_metadata_request_level_is_present():
    adapter, provider, *_ = make_adapter()
    adapter.plan_for_goal(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    assert "goal_management" in provider.requests[-1].metadata


def test_last_plan_not_set_when_goal_consistency_fails():
    provider = FakeProvider(plan_json("wrong goal"))
    adapter = GoalAwareLLMPlanner(LLMPlanner(LLMService(provider)))
    with pytest.raises(GoalAwareLLMPlanningConsistencyError):
        adapter.plan_for_goal(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    assert adapter.last_plan is None


def test_invalid_planning_request_type_is_rejected():
    adapter, *_ = make_adapter()
    with pytest.raises(GoalAwareLLMPlanningInputError):
        adapter.plan(object())  # type: ignore[arg-type]


def test_empty_ids_are_removed():
    item = GoalAwarePlanningContext(
        selected_goal=goal_a(), competing_goal_ids=("", " ", "g-b")
    )
    assert item.competing_goal_ids == ("g-b",)


def test_selected_goal_management_contains_switch_flags():
    selected = goal_a()
    arb = arbitration_for(selected)
    planning = GoalAwarePlanningContextBuilder().build(selected, arbitration=arb)
    payload = planning.to_dict()
    assert "arbitration_switched" in payload
    assert "arbitration_preempted" in payload


def test_enrich_context_keeps_custom_goal_management_fields():
    original = context()
    planning = GoalAwarePlanningContextBuilder().build(goal_a())
    enriched = GoalAwarePlanningContextBuilder().enrich_context(original, planning)
    assert enriched.metadata["goal_management"]["legacy"] is True
    assert enriched.metadata["goal_management"]["selected_goal"]["goal_id"] == "g-a"


def test_plan_response_provider_and_model_flow_through():
    adapter, *_ = make_adapter()
    plan = adapter.plan_for_goal(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    assert plan.provider == "fake"
    assert plan.model == "fake-model"


def test_plan_clear_trace_also_clears_underlying_response():
    adapter, _, _, planner = make_adapter()
    adapter.plan_for_goal(reasoning=reasoning(), context=context(), selected_goal=goal_a())
    assert planner.last_plan is not None
    adapter.clear_trace()
    assert planner.last_plan is None
    assert planner.last_response is None


def test_goal_aware_adapter_does_not_mutate_selected_goal():
    selected = goal_a()
    before = selected.to_dict() if hasattr(selected, "to_dict") else (selected.goal_id, selected.progress)
    adapter, *_ = make_adapter()
    adapter.build_request(reasoning=reasoning(), context=context(), selected_goal=selected)
    after = (selected.goal_id, selected.progress)
    assert before == after


def test_builder_does_not_mutate_available_goals():
    goals = [goal_a(), goal_b()]
    before = [(g.goal_id, g.priority, g.progress) for g in goals]
    GoalAwarePlanningContextBuilder().build(goal_a(), available_goals=goals)
    after = [(g.goal_id, g.priority, g.progress) for g in goals]
    assert before == after
