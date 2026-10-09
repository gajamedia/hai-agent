from __future__ import annotations

import pytest

from agent.llm.context import AgentContextBuilder
from agent.llm.contracts import LLMResponse
from agent.llm.interpretation import LLMInterpretation
from agent.llm.reasoning import (
    LLMReasoning,
    LLMReasoningParseError,
    LLMReasoningParser,
    LLMReasoningSchemaError,
    LLMReasoner,
)
from agent.llm.service import LLMService


class FakeProvider:
    name = "fake"

    def __init__(self, content: str = "") -> None:
        self.content = content
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            content=self.content,
            provider=self.name,
            model="fake-model",
        )


def make_service(content: str) -> tuple[LLMService, FakeProvider]:
    provider = FakeProvider(content)
    return LLMService(provider), provider


def valid_payload() -> str:
    return (
        '{"summary":"The box is likely safe to inspect.",'
        '"conclusions":["The box is open"],'
        '"alternatives":["The observation may be misleading"],'
        '"risks":["Contents are unknown"],'
        '"information_gaps":["No confirmation of the box contents"],'
        '"recommended_direction":"Gather more evidence before committing to a specific action.",'
        '"confidence":0.82}'
    )


def make_interpretation() -> LLMInterpretation:
    return LLMInterpretation(
        summary="A box is open.",
        facts=("box is open",),
        entities={"box": {"state": "open"}},
        uncertainties=("contents are unknown",),
        implications=("the box can be inspected",),
        confidence=0.85,
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


def test_parser_builds_reasoning():
    result = LLMReasoningParser.parse(
        valid_payload(),
        provider="opencode",
        model="test-model",
    )

    assert isinstance(result, LLMReasoning)
    assert result.summary == "The box is likely safe to inspect."
    assert result.conclusions == ("The box is open",)
    assert result.alternatives == ("The observation may be misleading",)
    assert result.risks == ("Contents are unknown",)
    assert result.information_gaps == ("No confirmation of the box contents",)
    assert result.recommended_direction.startswith("Gather more evidence")
    assert result.confidence == 0.82
    assert result.provider == "opencode"
    assert result.model == "test-model"


def test_parser_accepts_json_fence():
    result = LLMReasoningParser.parse("```json\n" + valid_payload() + "\n```")
    assert result.confidence == 0.82


def test_parser_extracts_json_surrounded_by_text():
    result = LLMReasoningParser.parse("Result:\n" + valid_payload() + "\nDone.")
    assert result.summary == "The box is likely safe to inspect."


def test_parser_rejects_empty_content():
    with pytest.raises(LLMReasoningParseError, match="content is empty"):
        LLMReasoningParser.parse(" ")


def test_parser_rejects_invalid_json():
    with pytest.raises(LLMReasoningParseError, match="valid reasoning JSON"):
        LLMReasoningParser.parse("not json")


def test_parser_rejects_missing_field():
    with pytest.raises(LLMReasoningSchemaError, match="missing required fields"):
        LLMReasoningParser.parse('{"summary":"x"}')


def test_parser_rejects_empty_summary():
    content = valid_payload().replace('"The box is likely safe to inspect."', '""')
    with pytest.raises(LLMReasoningSchemaError, match="summary"):
        LLMReasoningParser.parse(content)


def test_parser_rejects_non_string_conclusions():
    content = valid_payload().replace('["The box is open"]', '[1]')
    with pytest.raises(LLMReasoningSchemaError, match="conclusions"):
        LLMReasoningParser.parse(content)


def test_parser_rejects_non_string_alternatives():
    content = valid_payload().replace(
        '["The observation may be misleading"]',
        '[1]',
    )
    with pytest.raises(LLMReasoningSchemaError, match="alternatives"):
        LLMReasoningParser.parse(content)


def test_parser_rejects_non_string_risks():
    content = valid_payload().replace('["Contents are unknown"]', '[1]')
    with pytest.raises(LLMReasoningSchemaError, match="risks"):
        LLMReasoningParser.parse(content)


def test_parser_rejects_non_string_information_gaps():
    content = valid_payload().replace(
        '["No confirmation of the box contents"]',
        '[1]',
    )
    with pytest.raises(LLMReasoningSchemaError, match="information_gaps"):
        LLMReasoningParser.parse(content)


def test_parser_accepts_null_recommended_direction():
    content = valid_payload().replace(
        '"recommended_direction":"Gather more evidence before committing to a specific action."',
        '"recommended_direction":null',
    )
    result = LLMReasoningParser.parse(content)
    assert result.recommended_direction is None


def test_parser_rejects_non_string_recommended_direction():
    content = valid_payload().replace(
        '"recommended_direction":"Gather more evidence before committing to a specific action."',
        '"recommended_direction":123',
    )
    with pytest.raises(LLMReasoningSchemaError, match="recommended_direction"):
        LLMReasoningParser.parse(content)


def test_parser_rejects_confidence_out_of_range():
    content = valid_payload().replace('"confidence":0.82', '"confidence":1.5')
    with pytest.raises(LLMReasoningSchemaError, match="0.0 to 1.0"):
        LLMReasoningParser.parse(content)


def test_parser_rejects_boolean_confidence():
    content = valid_payload().replace('"confidence":0.82', '"confidence":true')
    with pytest.raises(LLMReasoningSchemaError, match="0.0 to 1.0"):
        LLMReasoningParser.parse(content)


def test_parser_ignores_blank_items():
    content = valid_payload().replace(
        '["The box is open"]',
        '["The box is open", ""]',
    )
    result = LLMReasoningParser.parse(content)
    assert result.conclusions == ("The box is open",)


def test_reasoner_sends_interpretation_observation_and_context():
    service, provider = make_service(valid_payload())
    reasoner = LLMReasoner(service)

    result = reasoner.reason(
        make_interpretation(),
        make_context(),
        observation={"visual": "box appears open"},
        metadata={"episode_id": "ep-1"},
    )

    assert result.summary == "The box is likely safe to inspect."
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    content = request.messages[1].content
    assert "### INTERPRETATION" in content
    assert "box is open" in content
    assert "### OBSERVATION" in content
    assert "box appears open" in content
    assert "### AGENT CONTEXT" in content
    assert "## GOAL" in content
    assert "## METACOGNITION" in content
    assert request.metadata["operation"] == "reasoning"
    assert request.metadata["episode_id"] == "ep-1"


def test_reasoner_tracks_last_trace_and_can_clear_it():
    service, _ = make_service(valid_payload())
    reasoner = LLMReasoner(service)
    result = reasoner.reason(make_interpretation(), make_context())

    assert reasoner.last_response is not None
    assert reasoner.last_response.content == valid_payload()
    assert reasoner.last_reasoning is result

    reasoner.clear_trace()
    assert reasoner.last_response is None
    assert reasoner.last_reasoning is None


def test_reasoner_requires_interpretation_type():
    service, _ = make_service(valid_payload())
    reasoner = LLMReasoner(service)

    with pytest.raises(TypeError, match="LLMInterpretation"):
        reasoner.reason(object(), make_context())


def test_reasoner_requires_agent_context_type():
    service, _ = make_service(valid_payload())
    reasoner = LLMReasoner(service)

    with pytest.raises(TypeError, match="AgentContext"):
        reasoner.reason(make_interpretation(), object())


def test_reasoner_requires_nonempty_system_prompt():
    service, _ = make_service(valid_payload())
    with pytest.raises(ValueError, match="system_prompt"):
        LLMReasoner(service, system_prompt=" ")


def test_reasoning_to_dict_returns_plain_data():
    result = LLMReasoningParser.parse(valid_payload())
    payload = result.to_dict()

    assert payload["conclusions"] == ["The box is open"]
    assert payload["information_gaps"] == ["No confirmation of the box contents"]
    assert payload["provider"] is None


def test_reasoner_renders_missing_observation_explicitly():
    service, provider = make_service(valid_payload())
    reasoner = LLMReasoner(service)
    reasoner.reason(make_interpretation(), make_context())

    prompt = provider.requests[0].messages[1].content
    assert "<not provided>" in prompt
