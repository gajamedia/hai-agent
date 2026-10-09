from __future__ import annotations

import copy
import json

import pytest

from agent.llm import (
    AgentContext,
    CognitiveCycleResult,
    CognitiveOrchestrator,
    LLMInterpretation,
    LLMInterpretationParser,
    LLMInterpreter,
    LLMMetacognitionEngine,
    LLMPlan,
    LLMPlanStep,
    LLMPlanner,
    LLMReasoner,
    LLMReasoning,
    LLMResponse,
    LLMService,
    LLMOrchestrationStageError,
)
from agent.learning.metacognition import MetacognitiveReflection


INTERPRETATION = {
    "summary": "A switch is closed.",
    "facts": ["switch=closed"],
    "entities": {"switch": {"type": "object", "state": "closed"}},
    "uncertainties": [],
    "implications": ["Opening may change the room state."],
    "confidence": 0.9,
}

REASONING = {
    "summary": "Opening the switch is a plausible way to change the state.",
    "conclusions": ["The switch state matters to the goal."],
    "alternatives": ["Another mechanism may change the same state."],
    "risks": ["The switch may already be physically blocked."],
    "information_gaps": [],
    "recommended_direction": "Verify the switch and then change it if appropriate.",
    "confidence": 0.8,
}

PLAN = {
    "goal": "Open the switch",
    "rationale": "The switch state blocks the goal.",
    "steps": [
        {
            "index": 1,
            "action": "inspect_switch",
            "purpose": "Verify the switch is accessible.",
            "preconditions": ["switch exists"],
            "expected_observation": "switch can be inspected",
            "success_criteria": "inspection succeeds",
            "risk": None,
        },
        {
            "index": 2,
            "action": "open_switch",
            "purpose": "Change the switch to open.",
            "preconditions": ["inspection succeeded"],
            "expected_observation": "switch becomes open",
            "success_criteria": "switch=open",
            "risk": "switch may be blocked",
        },
    ],
    "assumptions": [],
    "constraints": ["do not execute during planning"],
    "stop_conditions": ["stop if the switch cannot be accessed"],
    "confidence": 0.85,
}

META = {
    "diagnosis": "The previous action produced partial progress.",
    "lesson": "Verify the switch state before attempting the next step.",
    "next_step": "Inspect the switch condition.",
    "evidence_for": ["progress increased"],
    "evidence_against": [],
    "alternative_explanations": ["another mechanism could explain the progress"],
    "uncertainties": ["the blocking cause is not fully established"],
    "change_required": True,
    "confidence": 0.74,
    "alignment": "refines",
}


class FakeProvider:
    name = "fake"

    def __init__(self, *, fail_operation: str | None = None) -> None:
        self.fail_operation = fail_operation
        self.operations: list[str] = []
        self.requests = []

    def complete(self, request):
        operation = request.metadata.get("operation")
        self.operations.append(operation)
        self.requests.append(request)
        if operation == self.fail_operation:
            raise RuntimeError(f"boom:{operation}")

        payload = {
            "interpretation": INTERPRETATION,
            "reasoning": REASONING,
            "planning": PLAN,
            "metacognition": META,
        }[operation]
        return LLMResponse(
            content=json.dumps(payload),
            provider=self.name,
            model="fake-model",
            finish_reason="stop",
        )


def make_reflection() -> MetacognitiveReflection:
    return MetacognitiveReflection(
        episode_id="ep-1",
        goal="Open the switch",
        action="inspect_switch",
        success=False,
        progress_before=0.2,
        progress_after=0.4,
        progress_delta=0.2,
        diagnosis="failure_with_partial_progress",
        lesson="Partial progress was observed.",
        next_step="Inspect the remaining constraint.",
        evidence_quality=1.0,
        confidence=1.0,
        change_required=True,
    )


def make_orchestrator(*, provider=None, with_meta=False):
    provider = provider or FakeProvider()
    service = LLMService(provider)
    orchestrator = CognitiveOrchestrator(
        service,
        metacognition_engine=(LLMMetacognitionEngine(service) if with_meta else None),
    )
    return orchestrator, provider, service


def make_context() -> AgentContext:
    return AgentContext(
        goal={"description": "Open the switch"},
        state={"cycle": 3},
        world_model={"switch": "closed"},
        relevant_memory=("switch was inspected",),
        strategy={"name": "verify_then_act"},
        self_awareness={"confidence": 0.7},
        calibration={"calibrated_confidence": 0.65},
        metacognition={"lesson": "verify before acting"},
    )


def test_requires_service():
    with pytest.raises(ValueError, match="llm_service is required"):
        CognitiveOrchestrator(None)


def test_default_components_share_service():
    orchestrator, _, service = make_orchestrator()
    assert orchestrator.interpreter.llm_service is service
    assert orchestrator.reasoner.llm_service is service
    assert orchestrator.planner.llm_service is service


def test_incompatible_injected_component_is_rejected():
    p1 = FakeProvider()
    p2 = FakeProvider()
    service1 = LLMService(p1)
    service2 = LLMService(p2)
    with pytest.raises(ValueError, match="same LLMService instance"):
        CognitiveOrchestrator(
            service1,
            interpreter=LLMInterpreter(service2),
        )


def test_custom_components_are_accepted_when_they_share_service():
    _, _, service = make_orchestrator()
    orchestrator = CognitiveOrchestrator(
        service,
        interpreter=LLMInterpreter(service),
        reasoner=LLMReasoner(service),
        planner=LLMPlanner(service),
        metacognition_engine=LLMMetacognitionEngine(service),
    )
    assert isinstance(orchestrator, CognitiveOrchestrator)


def test_run_requires_agent_context():
    orchestrator, _, _ = make_orchestrator()
    with pytest.raises(TypeError, match="AgentContext"):
        orchestrator.run("obs", {})


def test_run_completes_three_main_stages():
    orchestrator, provider, _ = make_orchestrator()
    result = orchestrator.run("switch observed", make_context())
    assert isinstance(result, CognitiveCycleResult)
    assert provider.operations == ["interpretation", "reasoning", "planning"]
    assert result.interpretation.summary == INTERPRETATION["summary"]
    assert result.reasoning.confidence == REASONING["confidence"]
    assert result.plan.actions == ("inspect_switch", "open_switch")
    assert result.metacognition is None


def test_run_stores_last_result():
    orchestrator, _, _ = make_orchestrator()
    result = orchestrator.run("obs", make_context())
    assert orchestrator.last_result is result
    assert orchestrator.last_failed_stage is None


def test_run_preserves_context_identity():
    orchestrator, _, _ = make_orchestrator()
    context = make_context()
    result = orchestrator.run("obs", context)
    assert result.context is context


def test_run_does_not_mutate_observation():
    orchestrator, _, _ = make_orchestrator()
    observation = {"switch": {"state": "closed"}, "items": [1, 2]}
    before = copy.deepcopy(observation)
    orchestrator.run(observation, make_context())
    assert observation == before


def test_build_context_delegates_to_context_builder():
    orchestrator, _, _ = make_orchestrator()
    context = orchestrator.build_context(
        goal="goal",
        state={"x": 1},
        world_model={"room": "A"},
        relevant_memory=["m1", "m2"],
        strategy={"name": "s"},
        self_awareness={"confidence": 0.8},
        calibration={"confidence": 0.7},
        metacognition={"lesson": "l"},
        metadata={"source": "test"},
    )
    assert context.goal == "goal"
    assert context.state == {"x": 1}
    assert context.relevant_memory == ("m1", "m2")
    assert context.metadata == {"source": "test"}


def test_build_context_can_feed_run():
    orchestrator, provider, _ = make_orchestrator()
    context = orchestrator.build_context(goal="goal", relevant_memory=["m"])
    result = orchestrator.run("observation", context)
    assert result.context.goal == "goal"
    assert provider.operations[-1] == "planning"


def test_metadata_is_propagated_to_each_stage():
    orchestrator, provider, _ = make_orchestrator()
    orchestrator.run("obs", make_context(), metadata={"trace_id": "abc"})
    assert [r.metadata["trace_id"] for r in provider.requests] == ["abc", "abc", "abc"]
    assert [r.metadata["stage"] for r in provider.requests] == [
        "interpretation",
        "reasoning",
        "planning",
    ]
    assert all(r.metadata["orchestration"] == "cognitive_cycle" for r in provider.requests)


def test_planning_receives_prior_reasoning_output():
    orchestrator, provider, _ = make_orchestrator()
    orchestrator.run("obs", make_context())
    planning_request = provider.requests[2]
    assert REASONING["summary"] in planning_request.messages[-1].content


def test_reasoning_receives_prior_interpretation_output():
    orchestrator, provider, _ = make_orchestrator()
    orchestrator.run("obs", make_context())
    reasoning_request = provider.requests[1]
    assert INTERPRETATION["summary"] in reasoning_request.messages[-1].content


def test_each_stage_uses_same_service_instance():
    orchestrator, _, service = make_orchestrator()
    assert orchestrator.interpreter.llm_service is service
    assert orchestrator.reasoner.llm_service is service
    assert orchestrator.planner.llm_service is service


def test_interpretation_failure_is_wrapped_with_stage():
    orchestrator, _, _ = make_orchestrator(provider=FakeProvider(fail_operation="interpretation"))
    with pytest.raises(LLMOrchestrationStageError, match="interpretation") as exc:
        orchestrator.run("obs", make_context())
    assert exc.value.stage == "interpretation"
    assert isinstance(exc.value.cause, RuntimeError)
    assert orchestrator.last_failed_stage == "interpretation"
    assert orchestrator.last_result is None


def test_reasoning_failure_stops_before_planning():
    provider = FakeProvider(fail_operation="reasoning")
    orchestrator, provider, _ = make_orchestrator(provider=provider)
    with pytest.raises(LLMOrchestrationStageError, match="reasoning"):
        orchestrator.run("obs", make_context())
    assert provider.operations == ["interpretation", "reasoning"]
    assert orchestrator.last_failed_stage == "reasoning"


def test_planning_failure_stops_cycle():
    provider = FakeProvider(fail_operation="planning")
    orchestrator, provider, _ = make_orchestrator(provider=provider)
    with pytest.raises(LLMOrchestrationStageError, match="planning"):
        orchestrator.run("obs", make_context())
    assert provider.operations == ["interpretation", "reasoning", "planning"]
    assert orchestrator.last_failed_stage == "planning"


def test_metacognition_requires_engine_when_requested():
    orchestrator, _, _ = make_orchestrator(with_meta=False)
    with pytest.raises(ValueError, match="metacognition_engine is required"):
        orchestrator.run(
            "obs",
            make_context(),
            episode={"success": False},
            deterministic_reflection=make_reflection(),
            include_metacognition=True,
        )


def test_metacognition_requires_episode_when_requested():
    orchestrator, _, _ = make_orchestrator(with_meta=True)
    with pytest.raises(ValueError, match="episode is required"):
        orchestrator.run(
            "obs",
            make_context(),
            deterministic_reflection=make_reflection(),
            include_metacognition=True,
        )


def test_metacognition_requires_deterministic_reflection():
    orchestrator, _, _ = make_orchestrator(with_meta=True)
    with pytest.raises(TypeError, match="MetacognitiveReflection"):
        orchestrator.run(
            "obs",
            make_context(),
            episode={"success": False},
            deterministic_reflection={},
            include_metacognition=True,
        )


def test_optional_metacognition_appends_fourth_stage():
    orchestrator, provider, _ = make_orchestrator(with_meta=True)
    result = orchestrator.run(
        "obs",
        make_context(),
        episode={"episode_id": "ep-1", "success": False},
        deterministic_reflection=make_reflection(),
        include_metacognition=True,
    )
    assert provider.operations == [
        "interpretation",
        "reasoning",
        "planning",
        "metacognition",
    ]
    assert result.metacognition is not None
    assert result.metacognition.alignment == "refines"


def test_metacognition_failure_is_wrapped_with_stage():
    provider = FakeProvider(fail_operation="metacognition")
    orchestrator, provider, _ = make_orchestrator(provider=provider, with_meta=True)
    with pytest.raises(LLMOrchestrationStageError, match="metacognition"):
        orchestrator.run(
            "obs",
            make_context(),
            episode={"success": False},
            deterministic_reflection=make_reflection(),
            include_metacognition=True,
        )
    assert provider.operations == [
        "interpretation",
        "reasoning",
        "planning",
        "metacognition",
    ]


def test_result_to_dict_contains_all_main_stages():
    orchestrator, _, _ = make_orchestrator()
    result = orchestrator.run("obs", make_context())
    payload = result.to_dict()
    assert set(payload) == {
        "context",
        "interpretation",
        "reasoning",
        "plan",
        "metacognition",
    }
    assert payload["metacognition"] is None
    assert payload["plan"]["steps"][0]["index"] == 1


def test_result_to_dict_contains_metacognition_when_present():
    orchestrator, _, _ = make_orchestrator(with_meta=True)
    result = orchestrator.run(
        "obs",
        make_context(),
        episode={"success": False},
        deterministic_reflection=make_reflection(),
        include_metacognition=True,
    )
    payload = result.to_dict()
    assert payload["metacognition"]["llm_reflection"]["alignment"] == "refines"


def test_clear_trace_clears_orchestrator_and_components():
    orchestrator, _, _ = make_orchestrator(with_meta=True)
    orchestrator.run(
        "obs",
        make_context(),
        episode={"success": False},
        deterministic_reflection=make_reflection(),
        include_metacognition=True,
    )
    orchestrator.clear_trace()
    assert orchestrator.last_result is None
    assert orchestrator.last_failed_stage is None
    assert orchestrator.interpreter.last_interpretation is None
    assert orchestrator.reasoner.last_reasoning is None
    assert orchestrator.planner.last_plan is None
    assert orchestrator.metacognition_engine.last_result is None


def test_failed_stage_does_not_publish_partial_result():
    orchestrator, _, _ = make_orchestrator(provider=FakeProvider(fail_operation="planning"))
    with pytest.raises(LLMOrchestrationStageError):
        orchestrator.run("obs", make_context())
    assert orchestrator.last_result is None


def test_success_after_failure_clears_previous_failure_marker():
    provider = FakeProvider(fail_operation="planning")
    orchestrator, _, _ = make_orchestrator(provider=provider)
    with pytest.raises(LLMOrchestrationStageError):
        orchestrator.run("obs", make_context())
    provider.fail_operation = None
    result = orchestrator.run("obs", make_context())
    assert result is orchestrator.last_result
    assert orchestrator.last_failed_stage is None


def test_orchestrator_is_side_effect_free_against_context_data():
    orchestrator, _, _ = make_orchestrator()
    context = make_context()
    original = copy.deepcopy(context.to_dict())
    orchestrator.run("obs", context)
    assert context.to_dict() == original


def test_result_models_remain_typed_objects():
    orchestrator, _, _ = make_orchestrator()
    result = orchestrator.run("obs", make_context())
    assert isinstance(result.interpretation, LLMInterpretation)
    assert isinstance(result.reasoning, LLMReasoning)
    assert isinstance(result.plan, LLMPlan)
    assert isinstance(result.plan.steps[0], LLMPlanStep)
