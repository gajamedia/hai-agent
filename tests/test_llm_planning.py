from __future__ import annotations

import json

import pytest

from agent.llm.context import AgentContextBuilder
from agent.llm.contracts import LLMResponse
from agent.llm.planning import (
    LLMPlan,
    LLMPlanParser,
    LLMPlanStep,
    LLMPlanner,
    LLMPlanningParseError,
    LLMPlanningSchemaError,
)
from agent.llm.reasoning import LLMReasoning
from agent.llm.service import LLMService


class FakeProvider:
    name = "fake"

    def __init__(self, content: str | list[str] = "") -> None:
        self.content = content
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if isinstance(self.content, list):
            content = self.content.pop(0) if self.content else ""
        else:
            content = self.content
        return LLMResponse(content=content, provider=self.name, model="fake-model")


def make_service(content: str) -> tuple[LLMService, FakeProvider]:
    provider = FakeProvider(content)
    return LLMService(provider), provider


def valid_payload() -> str:
    return json.dumps(
        {
            "goal": "inspect box",
            "rationale": "The box is already open and more evidence is needed.",
            "steps": [
                {
                    "index": 1,
                    "action": "inspect box contents",
                    "purpose": "Collect evidence about the contents.",
                    "preconditions": ["box is open"],
                    "expected_observation": "contents become observable",
                    "success_criteria": "contents are identified or uncertainty is recorded",
                    "risk": "contents may be unsafe",
                },
                {
                    "index": 2,
                    "action": "record inspection result",
                    "purpose": "Preserve the observation for later reasoning.",
                    "preconditions": ["inspection has completed"],
                    "expected_observation": "result is available",
                    "success_criteria": "inspection result is captured",
                    "risk": None,
                },
            ],
            "assumptions": ["the box remains accessible"],
            "constraints": ["do not open unknown sealed containers"],
            "stop_conditions": ["stop if the contents present immediate danger"],
            "confidence": 0.88,
        },
        ensure_ascii=False,
    )


def make_reasoning() -> LLMReasoning:
    return LLMReasoning(
        summary="The box is open and should be inspected cautiously.",
        conclusions=("The box is open",),
        alternatives=("The observation could be stale",),
        risks=("The contents are unknown",),
        information_gaps=("The box contents are not identified",),
        recommended_direction="Gather evidence before committing to a specific outcome.",
        confidence=0.81,
        provider="fake",
        model="fake-model",
    )


def make_context():
    return AgentContextBuilder().build(
        goal={"target": "inspect box"},
        state={"position": "room"},
        world_model={"box": {"state": "open"}},
        relevant_memory=[{"episode": 1, "result": "opened box"}],
        strategy={"name": "inspect_first"},
        self_awareness={"confidence": 0.7},
        calibration={"confidence": 0.65},
        metacognition={"next_step": "observe"},
    )


def test_parser_builds_plan():
    result = LLMPlanParser.parse(valid_payload(), provider="opencode", model="test-model")
    assert isinstance(result, LLMPlan)
    assert result.goal == "inspect box"
    assert len(result.steps) == 2
    assert result.actions == ("inspect box contents", "record inspection result")
    assert result.steps[0].index == 1
    assert result.steps[0].preconditions == ("box is open",)
    assert result.steps[1].risk is None
    assert result.confidence == 0.88
    assert result.provider == "opencode"
    assert result.model == "test-model"


def test_parser_accepts_json_fence():
    result = LLMPlanParser.parse("```json\n" + valid_payload() + "\n```")
    assert result.goal == "inspect box"


def test_parser_extracts_json_surrounded_by_text():
    result = LLMPlanParser.parse("Plan:\n" + valid_payload() + "\nDone.")
    assert len(result.steps) == 2


def test_parser_rejects_empty_content():
    with pytest.raises(LLMPlanningParseError, match="content is empty"):
        LLMPlanParser.parse(" ")


def test_parser_rejects_invalid_json():
    with pytest.raises(LLMPlanningParseError, match="valid planning JSON"):
        LLMPlanParser.parse("not json")


def test_parser_rejects_missing_root_field():
    payload = json.loads(valid_payload())
    payload.pop("constraints")
    with pytest.raises(LLMPlanningSchemaError, match="missing required fields"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_requires_at_least_one_step():
    payload = json.loads(valid_payload())
    payload["steps"] = []
    with pytest.raises(LLMPlanningSchemaError, match="at least one step"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_rejects_step_without_required_field():
    payload = json.loads(valid_payload())
    payload["steps"][0].pop("purpose")
    with pytest.raises(LLMPlanningSchemaError, match="missing required fields"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_rejects_non_object_step():
    payload = json.loads(valid_payload())
    payload["steps"][0] = "inspect box"
    with pytest.raises(LLMPlanningSchemaError, match="must be an object"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_rejects_duplicate_indices():
    payload = json.loads(valid_payload())
    payload["steps"][1]["index"] = 1
    with pytest.raises(LLMPlanningSchemaError, match="indices must be unique"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_requires_sequential_indices():
    payload = json.loads(valid_payload())
    payload["steps"][1]["index"] = 3
    with pytest.raises(LLMPlanningSchemaError, match="start at 1 and be sequential"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_rejects_non_string_preconditions():
    payload = json.loads(valid_payload())
    payload["steps"][0]["preconditions"] = [1]
    with pytest.raises(LLMPlanningSchemaError, match="preconditions"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_accepts_nullable_step_fields():
    payload = json.loads(valid_payload())
    payload["steps"][0]["expected_observation"] = None
    payload["steps"][0]["success_criteria"] = None
    payload["steps"][0]["risk"] = None
    result = LLMPlanParser.parse(json.dumps(payload))
    assert result.steps[0].expected_observation is None
    assert result.steps[0].success_criteria is None
    assert result.steps[0].risk is None


def test_parser_rejects_non_string_optional_step_field():
    payload = json.loads(valid_payload())
    payload["steps"][0]["risk"] = 123
    with pytest.raises(LLMPlanningSchemaError, match="risk must be a string or null"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_rejects_non_array_assumptions():
    payload = json.loads(valid_payload())
    payload["assumptions"] = "not-array"
    with pytest.raises(LLMPlanningSchemaError, match="assumptions"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_rejects_confidence_out_of_range():
    payload = json.loads(valid_payload())
    payload["confidence"] = 1.2
    with pytest.raises(LLMPlanningSchemaError, match="0.0 to 1.0"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_rejects_boolean_confidence():
    payload = json.loads(valid_payload())
    payload["confidence"] = True
    with pytest.raises(LLMPlanningSchemaError, match="0.0 to 1.0"):
        LLMPlanParser.parse(json.dumps(payload))


def test_parser_enforces_max_steps():
    payload = json.loads(valid_payload())
    payload["steps"] = payload["steps"] * 2
    payload["steps"][2]["index"] = 3
    payload["steps"][3]["index"] = 4
    with pytest.raises(LLMPlanningSchemaError, match="more than 3 steps"):
        LLMPlanParser.parse(json.dumps(payload), max_steps=3)


def test_parser_rejects_invalid_max_steps():
    with pytest.raises(ValueError, match="max_steps"):
        LLMPlanParser.parse(valid_payload(), max_steps=0)


def test_planner_sends_reasoning_and_context():
    service, provider = make_service(valid_payload())
    planner = LLMPlanner(service, max_steps=4)
    result = planner.plan(make_reasoning(), make_context(), metadata={"episode_id": "ep-1"})

    assert result.goal == "inspect box"
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    content = request.messages[1].content
    assert "### REASONING" in content
    assert "The box is open" in content
    assert "### AGENT CONTEXT" in content
    assert "## GOAL" in content
    assert request.metadata["operation"] == "planning"
    assert request.metadata["max_steps"] == 4
    assert request.metadata["episode_id"] == "ep-1"


def test_planner_tracks_trace_and_can_clear():
    service, _ = make_service(valid_payload())
    planner = LLMPlanner(service)
    result = planner.plan(make_reasoning(), make_context())
    assert planner.last_response is not None
    assert planner.last_plan is result
    planner.clear_trace()
    assert planner.last_response is None
    assert planner.last_plan is None


def test_planner_requires_reasoning_type():
    service, _ = make_service(valid_payload())
    with pytest.raises(TypeError, match="LLMReasoning"):
        LLMPlanner(service).plan(object(), make_context())


def test_planner_requires_context_type():
    service, _ = make_service(valid_payload())
    with pytest.raises(TypeError, match="AgentContext"):
        LLMPlanner(service).plan(make_reasoning(), object())


def test_planner_requires_nonempty_system_prompt():
    service, _ = make_service(valid_payload())
    with pytest.raises(ValueError, match="system_prompt"):
        LLMPlanner(service, system_prompt=" ")


def test_planner_requires_positive_max_steps():
    service, _ = make_service(valid_payload())
    with pytest.raises(ValueError, match="max_steps"):
        LLMPlanner(service, max_steps=0)


def test_plan_to_dict_returns_plain_data():
    result = LLMPlanParser.parse(valid_payload())
    data = result.to_dict()
    assert data["steps"][0]["index"] == 1
    assert data["assumptions"] == ["the box remains accessible"]
    assert data["provider"] is None


def test_plan_step_to_dict_returns_plain_data():
    step = LLMPlanStep(
        index=1,
        action="inspect box",
        purpose="gather evidence",
        preconditions=("box open",),
        expected_observation="contents visible",
        success_criteria="observation recorded",
        risk="unknown contents",
    )
    assert step.to_dict()["preconditions"] == ["box open"]




def test_planner_retries_once_when_schema_is_invalid():
    invalid_payload = json.loads(valid_payload())
    invalid_payload["steps"][1]["preconditions"] = {
        "inspection": "completed"
    }
    provider = FakeProvider([json.dumps(invalid_payload), valid_payload()])
    planner = LLMPlanner(LLMService(provider), max_steps=4)

    result = planner.plan(make_reasoning(), make_context())

    assert result.goal == "inspect box"
    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["planning_retry_reason"] == "schema_validation"
    correction_prompt = provider.requests[1].messages[1].content
    assert "array of strings" in correction_prompt
    assert "preconditions" in correction_prompt


def test_planner_stops_after_one_schema_correction_attempt():
    invalid_payload = json.loads(valid_payload())
    invalid_payload["steps"][1]["preconditions"] = {
        "inspection": "completed"
    }
    provider = FakeProvider([json.dumps(invalid_payload), json.dumps(invalid_payload)])
    planner = LLMPlanner(LLMService(provider), max_steps=4)

    with pytest.raises(LLMPlanningSchemaError, match="preconditions"):
        planner.plan(make_reasoning(), make_context())

    assert len(provider.requests) == 2
    assert planner.last_plan is None


def test_planner_retries_once_when_initial_json_is_malformed():
    malformed = valid_payload()[:-1]  # Missing final closing brace.
    provider = FakeProvider([malformed, valid_payload()])
    planner = LLMPlanner(LLMService(provider), max_steps=4)

    result = planner.plan(make_reasoning(), make_context())

    assert result.goal == "inspect box"
    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["planning_retry_reason"] == "json_parse"
    retry_prompt = provider.requests[1].messages[1].content
    assert "syntactically valid JSON object" in retry_prompt
    assert "meta-steps" in retry_prompt


def test_planner_stops_after_one_malformed_json_retry():
    malformed = valid_payload()[:-1]
    provider = FakeProvider([malformed, malformed])
    planner = LLMPlanner(LLMService(provider), max_steps=4)

    with pytest.raises(LLMPlanningParseError, match="valid planning JSON"):
        planner.plan(make_reasoning(), make_context())

    assert len(provider.requests) == 2
    assert planner.last_plan is None
