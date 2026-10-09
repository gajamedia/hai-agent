from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.context import AgentContextBuilder
from agent.llm.contracts import LLMResponse
from agent.llm.interpretation import (
    LLMInterpretation,
    LLMInterpretationParseError,
    LLMInterpretationParser,
    LLMInterpretationSchemaError,
    LLMInterpreter,
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
        '{"summary":"A box is open.",'
        '"facts":["box is open"],'
        '"entities":{"box":{"state":"open"}},'
        '"uncertainties":["contents are unknown"],'
        '"implications":["the box can be inspected"],'
        '"confidence":0.85}'
    )


def test_parser_builds_interpretation():
    result = LLMInterpretationParser.parse(
        valid_payload(),
        provider="opencode",
        model="test-model",
    )

    assert isinstance(result, LLMInterpretation)
    assert result.summary == "A box is open."
    assert result.facts == ("box is open",)
    assert result.entities["box"]["state"] == "open"
    assert result.uncertainties == ("contents are unknown",)
    assert result.implications == ("the box can be inspected",)
    assert result.confidence == 0.85
    assert result.provider == "opencode"
    assert result.model == "test-model"


def test_parser_accepts_json_fence():
    result = LLMInterpretationParser.parse(
        "```json\n" + valid_payload() + "\n```"
    )
    assert result.summary == "A box is open."


def test_parser_extracts_json_surrounded_by_text():
    result = LLMInterpretationParser.parse(
        "Here is the result:\n" + valid_payload() + "\nEnd."
    )
    assert result.confidence == 0.85


def test_parser_rejects_empty_content():
    with pytest.raises(LLMInterpretationParseError, match="content is empty"):
        LLMInterpretationParser.parse("   ")


def test_parser_rejects_invalid_json():
    with pytest.raises(LLMInterpretationParseError, match="valid interpretation JSON"):
        LLMInterpretationParser.parse("not json")


def test_parser_rejects_missing_field():
    with pytest.raises(LLMInterpretationSchemaError, match="missing required fields"):
        LLMInterpretationParser.parse('{"summary":"x"}')


def test_parser_rejects_empty_summary():
    content = valid_payload().replace('"A box is open."', '""')
    with pytest.raises(LLMInterpretationSchemaError, match="summary"):
        LLMInterpretationParser.parse(content)


def test_parser_rejects_non_string_facts():
    content = valid_payload().replace('["box is open"]', '[1]')
    with pytest.raises(LLMInterpretationSchemaError, match="facts"):
        LLMInterpretationParser.parse(content)


def test_parser_rejects_non_mapping_entities():
    content = valid_payload().replace('"entities":{"box":{"state":"open"}}', '"entities":["open"]')
    with pytest.raises(LLMInterpretationSchemaError, match="entities"):
        LLMInterpretationParser.parse(content)


def test_parser_rejects_confidence_out_of_range():
    content = valid_payload().replace('"confidence":0.85', '"confidence":1.5')
    with pytest.raises(LLMInterpretationSchemaError, match="0.0 to 1.0"):
        LLMInterpretationParser.parse(content)


def test_parser_ignores_blank_items():
    content = valid_payload().replace('["box is open"]', '["box is open", ""]')
    result = LLMInterpretationParser.parse(content)
    assert result.facts == ("box is open",)


def test_interpreter_sends_context_and_observation_to_service():
    service, provider = make_service(valid_payload())
    interpreter = LLMInterpreter(service)
    context = AgentContextBuilder().build(
        goal={"target": "inspect box"},
        state={"position": "room"},
        world_model={"box": {"state": "closed"}},
        relevant_memory=[{"episode": 1, "result": "opened box"}],
        strategy={"name": "inspect_first"},
        self_awareness={"confidence": 0.7},
        calibration={"confidence": 0.65},
        metacognition={"next_step": "observe"},
    )

    result = interpreter.interpret(
        {"visual": "box appears open"},
        context,
        metadata={"episode_id": "ep-1"},
    )

    assert result.summary == "A box is open."
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert "### OBSERVATION" in request.messages[1].content
    assert "box appears open" in request.messages[1].content
    assert "## GOAL" in request.messages[1].content
    assert "## SELF-AWARENESS" in request.messages[1].content
    assert request.metadata["operation"] == "interpretation"
    assert request.metadata["episode_id"] == "ep-1"


def test_interpreter_tracks_last_response_and_interpretation():
    service, _ = make_service(valid_payload())
    interpreter = LLMInterpreter(service)
    context = AgentContextBuilder().build(goal="test")

    result = interpreter.interpret("hello", context)

    assert interpreter.last_response is not None
    assert interpreter.last_response.content == valid_payload()
    assert interpreter.last_interpretation is result

    interpreter.clear_trace()
    assert interpreter.last_response is None
    assert interpreter.last_interpretation is None


def test_interpreter_requires_agent_context():
    service, _ = make_service(valid_payload())
    interpreter = LLMInterpreter(service)

    with pytest.raises(TypeError, match="AgentContext"):
        interpreter.interpret("hello", object())


def test_interpreter_requires_nonempty_system_prompt():
    service, _ = make_service(valid_payload())
    with pytest.raises(ValueError, match="system_prompt"):
        LLMInterpreter(service, system_prompt=" ")


def test_to_dict_returns_plain_data():
    result = LLMInterpretationParser.parse(valid_payload())
    payload = result.to_dict()

    assert payload["facts"] == ["box is open"]
    assert payload["entities"]["box"]["state"] == "open"
    assert payload["provider"] is None

