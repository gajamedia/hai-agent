from __future__ import annotations

import json

import pytest

from agent.llm.context import AgentContext, AgentContextBuilder
from agent.llm.contracts import LLMResponse
from agent.llm.llm_plan_adaptive_replanning import (
    LLMPlanAdaptiveReplanner,
    LLMPlanReplanningRequest,
)
from agent.llm.llm_planner_adaptive_replanning import (
    LLMPlannerAdaptiveReplanningBuildError,
    LLMPlannerAdaptiveReplanningConfigurationError,
    LLMPlannerAdaptiveReplanningProvider,
    default_risk_resolver,
    LLMPlannerAdaptiveReplanningRequestError,
    build_llm_adaptive_replanner,
)
from agent.llm.planning import LLMPlan, LLMPlanStep, LLMPlanner
from agent.llm.reasoning import LLMReasoning
from agent.llm.service import LLMService
from agent.llm.llm_plan_runtime_learning_feedback import LLMPlanRuntimeLearningFeedback


class FakeProvider:
    name = "fake"

    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            content=self.content,
            provider=self.name,
            model="fake-model",
        )


def step(index: int, action: str, purpose: str) -> LLMPlanStep:
    return LLMPlanStep(
        index=index,
        action=action,
        purpose=purpose,
        preconditions=("ready",),
        expected_observation="observed",
        success_criteria="completed",
        risk="low",
    )


def plan(*steps_: LLMPlanStep, goal: str = "reach target") -> LLMPlan:
    return LLMPlan(
        goal=goal,
        rationale="original",
        steps=tuple(steps_),
        assumptions=("world is observable",),
        constraints=("stay safe",),
        stop_conditions=("stop on danger",),
        confidence=0.8,
    )


def feedback(
    *,
    step_index: int = 1,
    action: str = "move",
    success: bool | None = False,
    progress_delta: float | None = 0.0,
    change_required: bool = True,
    continue_recommended: bool = False,
    exploration_recommended: bool = True,
    risk_level: str = "high",
) -> LLMPlanRuntimeLearningFeedback:
    return LLMPlanRuntimeLearningFeedback(
        step_index=step_index,
        action=action,
        success=success,
        progress_delta=progress_delta,
        evidence_quality=0.9,
        confidence=0.8,
        diagnosis="previous route blocked",
        lesson="avoid the blocked route",
        next_step="reassess the route",
        change_required=change_required,
        continue_recommended=continue_recommended,
        exploration_recommended=exploration_recommended,
        risk_level=risk_level,
        risk_score=0.75,
        information_gain=0.0,
        metadata={"origin": "test"},
    )


def valid_payload(goal: str = "reach target") -> str:
    return json.dumps(
        {
            "goal": goal,
            "rationale": "Use a safer alternate route.",
            "steps": [
                {
                    "index": 1,
                    "action": "inspect alternate route",
                    "purpose": "Gather evidence about the safe route.",
                    "preconditions": ["current state is observable"],
                    "expected_observation": "alternate route is visible",
                    "success_criteria": "route is identified",
                    "risk": "unknown route conditions",
                },
                {
                    "index": 2,
                    "action": "move via alternate route",
                    "purpose": "Continue toward the goal without repeating the failed move.",
                    "preconditions": ["alternate route is identified"],
                    "expected_observation": "position changes toward target",
                    "success_criteria": "target is reached or progress is measured",
                    "risk": "route may still be blocked",
                },
            ],
            "assumptions": ["alternate route exists"],
            "constraints": ["preserve completed successful work"],
            "stop_conditions": ["stop if immediate danger is observed"],
            "confidence": 0.86,
        }
    )


def make_request() -> LLMPlanReplanningRequest:
    original = plan(
        step(1, "move", "move directly"),
        step(2, "inspect", "inspect target"),
        step(3, "finish", "finish task"),
    )
    fb = feedback()
    return LLMPlanReplanningRequest(
        original_plan=original,
        goal=original.goal,
        world_state={"position": "start", "blocked": True},
        feedback=fb,
        preserved_steps=(),
        remaining_steps=original.steps,
        failed_step_index=1,
        next_step_index=2,
        generation=2,
        metadata={"episode_id": "ep-1"},
    )


def make_provider(content: str | None = None):
    fake = FakeProvider(content or valid_payload())
    service = LLMService(fake)
    planner = LLMPlanner(service, max_steps=4)
    adapter = LLMPlannerAdaptiveReplanningProvider(planner)
    return fake, service, planner, adapter


def test_requires_llm_planner():
    with pytest.raises(LLMPlannerAdaptiveReplanningConfigurationError):
        LLMPlannerAdaptiveReplanningProvider(object())


def test_accepts_callable_context_builder():
    _, _, planner, _ = make_provider()
    provider = LLMPlannerAdaptiveReplanningProvider(
        planner,
        context_builder=lambda request: AgentContext(goal=request.goal),
    )
    assert provider.context_builder is not None


def test_rejects_invalid_context_builder():
    _, _, planner, _ = make_provider()
    with pytest.raises(LLMPlannerAdaptiveReplanningConfigurationError):
        LLMPlannerAdaptiveReplanningProvider(planner, context_builder=123)


def test_rejects_invalid_reasoning_builder():
    _, _, planner, _ = make_provider()
    with pytest.raises(LLMPlannerAdaptiveReplanningConfigurationError):
        LLMPlannerAdaptiveReplanningProvider(planner, reasoning_builder=123)


def test_default_context_contains_goal_world_and_remaining_work():
    fake, _, planner, adapter = make_provider()
    result = adapter.replan(make_request())
    assert result.goal == "reach target"
    assert adapter.last_context is not None
    rendered = adapter.last_context.to_prompt()
    assert "reach target" in rendered
    assert '"blocked": true' in rendered
    assert "move directly" in rendered
    assert "finish task" in rendered
    request = fake.requests[0]
    assert "adaptive_replanning" in request.messages[1].content


def test_default_context_carries_feedback_and_runtime_position():
    _, _, _, adapter = make_provider()
    adapter.replan(make_request())
    context = adapter.last_context
    assert context is not None
    prompt = context.to_prompt()
    assert "previous route blocked" in prompt
    assert '"failed_step_index": 1' in prompt
    assert '"generation": 2' in prompt


def test_default_reasoning_is_high_level_not_action_list():
    _, _, _, adapter = make_provider()
    adapter.replan(make_request())
    reasoning = adapter.last_reasoning
    assert reasoning is not None
    assert "failed" in reasoning.summary.lower()
    assert reasoning.recommended_direction is not None
    assert "preserve completed successful work" in reasoning.recommended_direction


def test_custom_context_builder_is_used():
    _, _, planner, _ = make_provider()
    calls = []

    def build(request):
        calls.append(request)
        return AgentContext(goal="custom goal", world_model={"custom": True})

    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        context_builder=build,
    )
    adapter.replan(make_request())
    assert len(calls) == 1
    assert adapter.last_context.goal == "custom goal"


def test_custom_reasoning_builder_is_used():
    _, _, planner, _ = make_provider()
    calls = []
    custom = LLMReasoning(
        summary="custom reasoning",
        conclusions=("custom",),
        confidence=0.77,
    )

    def build(request):
        calls.append(request)
        return custom

    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        reasoning_builder=build,
    )
    adapter.replan(make_request())
    assert calls
    assert adapter.last_reasoning is custom
    assert "custom reasoning" in planner.llm_service.last_request.messages[1].content


def test_custom_context_builder_must_return_context():
    _, _, planner, _ = make_provider()
    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        context_builder=lambda request: object(),
    )
    with pytest.raises(LLMPlannerAdaptiveReplanningBuildError):
        adapter.replan(make_request())


def test_custom_reasoning_builder_must_return_reasoning():
    _, _, planner, _ = make_provider()
    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        reasoning_builder=lambda request: object(),
    )
    with pytest.raises(LLMPlannerAdaptiveReplanningBuildError):
        adapter.replan(make_request())


def test_context_builder_error_is_wrapped():
    _, _, planner, _ = make_provider()

    def broken(request):
        raise RuntimeError("boom")

    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        context_builder=broken,
    )
    with pytest.raises(LLMPlannerAdaptiveReplanningBuildError, match="boom"):
        adapter.replan(make_request())


def test_reasoning_builder_error_is_wrapped():
    _, _, planner, _ = make_provider()

    def broken(request):
        raise RuntimeError("reasoning boom")

    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        reasoning_builder=broken,
    )
    with pytest.raises(LLMPlannerAdaptiveReplanningBuildError, match="reasoning boom"):
        adapter.replan(make_request())


def test_requires_replanning_request():
    _, _, _, adapter = make_provider()
    with pytest.raises(LLMPlannerAdaptiveReplanningRequestError):
        adapter.replan(object())


def test_requires_nonempty_goal():
    _, _, planner, _ = make_provider()
    request = make_request()
    bad = LLMPlanReplanningRequest(
        **{**request.__dict__, "goal": ""}
    )
    adapter = LLMPlannerAdaptiveReplanningProvider(planner)
    with pytest.raises(LLMPlannerAdaptiveReplanningRequestError):
        adapter.replan(bad)


def test_rejects_invalid_next_index():
    _, _, planner, _ = make_provider()
    request = make_request()
    bad = LLMPlanReplanningRequest(
        **{**request.__dict__, "next_step_index": 0}
    )
    adapter = LLMPlannerAdaptiveReplanningProvider(planner)
    with pytest.raises(LLMPlannerAdaptiveReplanningRequestError):
        adapter.replan(bad)


def test_rejects_invalid_generation():
    _, _, planner, _ = make_provider()
    request = make_request()
    bad = LLMPlanReplanningRequest(
        **{**request.__dict__, "generation": 0}
    )
    adapter = LLMPlannerAdaptiveReplanningProvider(planner)
    with pytest.raises(LLMPlannerAdaptiveReplanningRequestError):
        adapter.replan(bad)


def test_callable_contract_is_supported():
    _, _, _, adapter = make_provider()
    result = adapter(make_request())
    assert isinstance(result, LLMPlan)


def test_provider_uses_canonical_llm_planner():
    fake, _, planner, adapter = make_provider()
    result = adapter.replan(make_request())
    assert result is not None
    assert adapter.last_plan is not None
    assert planner.last_plan is not None
    assert len(fake.requests) == 1
    assert fake.requests[0].metadata["operation"] == "adaptive_replanning"
    assert fake.requests[0].metadata["stage"] == "24.15"


def test_metadata_includes_generation_and_step_indices():
    fake, _, _, adapter = make_provider()
    adapter.replan(make_request())
    metadata = fake.requests[0].metadata
    assert metadata["generation"] == 2
    assert metadata["feedback_step_index"] == 1
    assert metadata["failed_step_index"] == 1
    assert metadata["next_step_index"] == 2
    assert metadata["remaining_original_step_indices"] == (1, 2, 3)


def test_custom_adapter_metadata_is_merged():
    fake, _, planner, _ = make_provider()
    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        metadata={"trace_id": "abc"},
    )
    adapter.replan(make_request())
    assert fake.requests[0].metadata["trace_id"] == "abc"


def test_llm_plan_provider_and_model_are_preserved():
    _, _, _, adapter = make_provider()
    result = adapter.replan(make_request())
    assert result.provider == "fake"
    assert result.model == "fake-model"
    assert all(isinstance(step.risk, (int, float)) or step.risk is None for step in result.steps)


def test_invalid_llm_json_propagates_planning_error():
    _, _, _, adapter = make_provider("not json")
    with pytest.raises(Exception, match="valid planning JSON"):
        adapter.replan(make_request())


def test_planner_trace_is_available_after_replan():
    _, _, planner, adapter = make_provider()
    adapter.replan(make_request())
    assert planner.last_response is not None
    assert planner.last_plan is not None
    assert adapter.last_plan is not None
    assert planner.last_plan is not None


def test_clear_trace_clears_adapter_and_planner():
    _, _, planner, adapter = make_provider()
    adapter.replan(make_request())
    adapter.clear_trace()
    assert adapter.last_request is None
    assert adapter.last_context is None
    assert adapter.last_reasoning is None
    assert adapter.last_plan is None
    assert planner.last_response is None
    assert planner.last_plan is None


def test_factory_returns_stage_24_13_replanner_and_adapter():
    _, _, planner, _ = make_provider()
    replanner, adapter = build_llm_adaptive_replanner(planner, max_replans=2)
    assert isinstance(replanner, LLMPlanAdaptiveReplanner)
    assert isinstance(adapter, LLMPlannerAdaptiveReplanningProvider)
    assert replanner.max_replans == 2


def test_factory_provider_is_the_replan_callback():
    _, _, planner, _ = make_provider()
    replanner, adapter = build_llm_adaptive_replanner(planner)
    assert replanner.replanner is adapter


def test_no_llm_call_when_trigger_does_not_require_replanning():
    fake, _, planner, adapter = make_provider()
    replanner = LLMPlanAdaptiveReplanner(adapter, max_replans=3)
    original = plan(step(1, "move", "move"), step(2, "inspect", "inspect"))
    positive = feedback(
        success=True,
        progress_delta=0.4,
        change_required=False,
        continue_recommended=True,
        exploration_recommended=False,
        risk_level="low",
    )
    result = replanner.replan(original, feedback=positive, next_step_index=2)
    assert not result.replanned
    assert len(fake.requests) == 0
    assert planner.last_plan is None


def test_adaptive_replanner_calls_llm_only_after_trigger():
    fake, _, _, adapter = make_provider()
    replanner = LLMPlanAdaptiveReplanner(adapter, max_replans=3)
    original = plan(step(1, "move", "move"), step(2, "inspect", "inspect"))
    result = replanner.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(),
        next_step_index=2,
        available_actions={"inspect alternate route": object(), "move via alternate route": object()},
    )
    assert result.replanned
    assert len(fake.requests) == 1
    assert result.replacement_plan is not None
    assert result.replacement_plan.goal == original.goal


def test_adaptive_replanner_replacement_is_rebased_by_stage_24_13():
    _, _, _, adapter = make_provider()
    replanner = LLMPlanAdaptiveReplanner(adapter, max_replans=3)
    original = plan(step(1, "move", "move"), step(2, "inspect", "inspect"))
    result = replanner.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=4,
        available_actions={"inspect alternate route": object(), "move via alternate route": object()},
    )
    assert result.replacement_step_indices == (4, 5)


def test_adapter_does_not_execute_actions():
    _, _, _, adapter = make_provider()
    request = make_request()
    result = adapter.replan(request)
    assert isinstance(result, LLMPlan)
    assert not hasattr(result, "execute")


def test_adapter_does_not_require_agent_instance():
    _, _, _, adapter = make_provider()
    assert not hasattr(adapter, "agent")


def test_result_wrapper_records_llm_trace():
    _, _, _, adapter = make_provider()
    replanner = LLMPlanAdaptiveReplanner(adapter)
    original = plan(step(1, "move", "move"))
    replanning = replanner.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=2,
        available_actions={"inspect alternate route": object(), "move via alternate route": object()},
    )
    from agent.llm.llm_planner_adaptive_replanning import LLMPlannerAdaptiveReplanningResult
    wrapped = LLMPlannerAdaptiveReplanningResult(
        replanning=replanning,
        planner=adapter,
        llm_provider=adapter.last_plan.provider,
        llm_model=adapter.last_plan.model,
    )
    assert wrapped.llm_called
    assert wrapped.replanned
    assert wrapped.replacement_plan is replanning.replacement_plan


def test_context_builder_snapshot_helper_round_trip():
    from agent.llm.llm_planner_adaptive_replanning import LLMPlannerAdaptiveReplanningContext
    request = make_request()
    snapshot = LLMPlannerAdaptiveReplanningContext(
        goal=request.goal,
        world_state=request.world_state,
        preserved_steps=request.preserved_steps,
        remaining_steps=request.remaining_steps,
        failed_step_index=request.failed_step_index,
        next_step_index=request.next_step_index,
        generation=request.generation,
        feedback_step_index=request.feedback.step_index,
        feedback_action=request.feedback.action,
        metadata={"x": 1},
    )
    data = snapshot.to_dict()
    assert data["goal"] == request.goal
    assert data["failed_step_index"] == 1
    assert data["remaining_steps"][0]["action"] == "move"


def test_default_risk_resolver_is_conservative_for_textual_risk():
    assert default_risk_resolver(None) == 0.0
    assert default_risk_resolver("low risk") == 0.25
    assert default_risk_resolver("unknown conditions") == 0.5
    assert default_risk_resolver("immediate danger") == 0.75


def test_custom_risk_resolver_is_used():
    _, _, planner, _ = make_provider()
    calls = []

    def resolver(value):
        calls.append(value)
        return 0.1

    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        risk_resolver=resolver,
    )
    result = adapter.replan(make_request())
    assert calls
    assert all(step.risk == 0.1 for step in result.steps)


def test_replan_integrates_with_stage_24_13_without_risk_schema_failure():
    _, _, planner, adapter = make_provider()
    replanner = LLMPlanAdaptiveReplanner(adapter, max_replans=1)
    original = plan(step(1, "move", "move"), step(2, "inspect", "inspect"))
    result = replanner.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=3,
        available_actions={
            "inspect alternate route": object(),
            "move via alternate route": object(),
        },
    )
    assert result.replanned
    assert result.validation is not None and result.validation.valid
    assert result.replacement_plan is not None
    assert tuple(step_.index for step_ in result.replacement_plan.steps) == (3, 4)


def test_original_textual_risk_remains_available_in_raw_content():
    _, _, _, adapter = make_provider()
    result = adapter.replan(make_request())
    assert result.raw_content is not None
    assert "unknown route conditions" in result.raw_content


def test_invalid_risk_resolver_result_is_wrapped():
    _, _, planner, _ = make_provider()
    adapter = LLMPlannerAdaptiveReplanningProvider(
        planner,
        risk_resolver=lambda _: "not numeric",
    )
    with pytest.raises(LLMPlannerAdaptiveReplanningBuildError):
        adapter.replan(make_request())


@pytest.mark.parametrize("bad_goal", [None, 123])
def test_empty_or_invalid_goal_is_rejected(bad_goal):
    _, _, planner, _ = make_provider()
    request = make_request()
    bad = LLMPlanReplanningRequest(
        **{**request.__dict__, "goal": bad_goal}
    )
    adapter = LLMPlannerAdaptiveReplanningProvider(planner)
    with pytest.raises(LLMPlannerAdaptiveReplanningRequestError):
        adapter.replan(bad)
