from __future__ import annotations

import json

import pytest

from agent.llm.context import AgentContextBuilder
from agent.llm.contracts import LLMResponse
from agent.llm.metacognition import (
    LLMMetacognitiveReflection,
    LLMMetacognitionEngine,
    LLMMetacognitionParseError,
    LLMMetacognitionParser,
    LLMMetacognitionSchemaError,
    LLMMetacognitionResult,
)
from agent.learning.metacognition import MetacognitionEngine


VALID_PAYLOAD = {
    "diagnosis": "The action failed because the required precondition was missing.",
    "lesson": "Verify the precondition before repeating the action.",
    "next_step": "Check whether the prerequisite state is available.",
    "evidence_for": ["The observation reports a missing prerequisite."],
    "evidence_against": [],
    "alternative_explanations": ["The observation may be incomplete."],
    "uncertainties": ["The environment state was only partially observed."],
    "change_required": True,
    "confidence": 0.84,
    "alignment": "refines",
}


def payload_text() -> str:
    return json.dumps(VALID_PAYLOAD)


def make_baseline():
    return MetacognitionEngine().reflect_on_episode(
        {
            "episode_id": "ep-1",
            "goal": "unlock the door",
            "action": "unlock_door",
            "success": False,
            "progress_before": 0.4,
            "progress_after": 0.4,
            "result": {"error": "key unavailable"},
        }
    )


class FakeLLMService:
    provider_name = "fake"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def complete_text(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return LLMResponse(
            content=self.content,
            provider="fake",
            model="fake-model",
        )


def make_context():
    return AgentContextBuilder().build(
        goal="unlock the door",
        state={"door": "locked", "key": "missing"},
        world_model={"door": {"locked": True}},
        relevant_memory=["Previous attempt failed when key was unavailable."],
        strategy={"mode": "explore"},
        self_awareness={"uncertainty": 0.7},
        calibration={"confidence": 0.62},
        metacognition={"diagnosis": "missing prerequisite"},
    )


def test_parser_returns_structured_reflection():
    result = LLMMetacognitionParser.parse(
        payload_text(), provider="fake", model="m1"
    )
    assert isinstance(result, LLMMetacognitiveReflection)
    assert result.diagnosis.startswith("The action failed")
    assert result.change_required is True
    assert result.confidence == 0.84
    assert result.alignment == "refines"
    assert result.provider == "fake"
    assert result.model == "m1"


def test_parser_preserves_lists_as_tuples():
    result = LLMMetacognitionParser.parse(payload_text())
    assert isinstance(result.evidence_for, tuple)
    assert isinstance(result.alternative_explanations, tuple)


def test_parser_accepts_fenced_json():
    result = LLMMetacognitionParser.parse(
        "```json\n" + payload_text() + "\n```"
    )
    assert result.lesson.startswith("Verify")


def test_parser_extracts_json_from_surrounding_text():
    result = LLMMetacognitionParser.parse(
        "Here is the reflection:\n" + payload_text() + "\nDone."
    )
    assert result.next_step.startswith("Check")


def test_parser_rejects_empty_content():
    with pytest.raises(LLMMetacognitionParseError):
        LLMMetacognitionParser.parse("")


def test_parser_rejects_invalid_json():
    with pytest.raises(LLMMetacognitionParseError):
        LLMMetacognitionParser.parse("not json")


def test_parser_rejects_non_object_json():
    with pytest.raises(LLMMetacognitionSchemaError, match="JSON object"):
        LLMMetacognitionParser.parse("[]")


def test_parser_rejects_missing_required_field():
    payload = dict(VALID_PAYLOAD)
    payload.pop("lesson")
    with pytest.raises(LLMMetacognitionSchemaError, match="lesson"):
        LLMMetacognitionParser.parse(json.dumps(payload))


def test_parser_rejects_empty_required_string():
    payload = dict(VALID_PAYLOAD)
    payload["diagnosis"] = "   "
    with pytest.raises(LLMMetacognitionSchemaError, match="diagnosis"):
        LLMMetacognitionParser.parse(json.dumps(payload))


def test_parser_rejects_non_string_list_item():
    payload = dict(VALID_PAYLOAD)
    payload["evidence_for"] = ["ok", 7]
    with pytest.raises(LLMMetacognitionSchemaError, match="evidence_for"):
        LLMMetacognitionParser.parse(json.dumps(payload))


def test_parser_rejects_non_boolean_change_required():
    payload = dict(VALID_PAYLOAD)
    payload["change_required"] = 1
    with pytest.raises(LLMMetacognitionSchemaError, match="boolean"):
        LLMMetacognitionParser.parse(json.dumps(payload))


def test_parser_rejects_boolean_confidence():
    payload = dict(VALID_PAYLOAD)
    payload["confidence"] = True
    with pytest.raises(LLMMetacognitionSchemaError, match="confidence"):
        LLMMetacognitionParser.parse(json.dumps(payload))


def test_parser_rejects_confidence_below_zero():
    payload = dict(VALID_PAYLOAD)
    payload["confidence"] = -0.1
    with pytest.raises(LLMMetacognitionSchemaError, match="confidence"):
        LLMMetacognitionParser.parse(json.dumps(payload))


def test_parser_rejects_confidence_above_one():
    payload = dict(VALID_PAYLOAD)
    payload["confidence"] = 1.1
    with pytest.raises(LLMMetacognitionSchemaError, match="confidence"):
        LLMMetacognitionParser.parse(json.dumps(payload))


def test_parser_rejects_unknown_alignment():
    payload = dict(VALID_PAYLOAD)
    payload["alignment"] = "unknown"
    with pytest.raises(LLMMetacognitionSchemaError, match="alignment"):
        LLMMetacognitionParser.parse(json.dumps(payload))


def test_parser_supports_all_alignments():
    for alignment in (
        "supports",
        "refines",
        "contradicts",
        "insufficient_evidence",
    ):
        payload = dict(VALID_PAYLOAD)
        payload["alignment"] = alignment
        result = LLMMetacognitionParser.parse(json.dumps(payload))
        assert result.alignment == alignment


def test_to_dict_returns_json_friendly_shape():
    result = LLMMetacognitionParser.parse(payload_text())
    data = result.to_dict()
    assert data["evidence_for"] == list(result.evidence_for)
    assert data["change_required"] is True
    assert data["provider"] is None


def test_engine_requires_service():
    with pytest.raises(ValueError, match="llm_service"):
        LLMMetacognitionEngine(None)


def test_engine_rejects_empty_system_prompt():
    with pytest.raises(ValueError, match="system_prompt"):
        LLMMetacognitionEngine(FakeLLMService(payload_text()), system_prompt=" ")


def test_engine_requires_deterministic_reflection():
    engine = LLMMetacognitionEngine(FakeLLMService(payload_text()))
    with pytest.raises(TypeError, match="MetacognitiveReflection"):
        engine.reflect({}, {}, make_context())


def test_engine_requires_context():
    engine = LLMMetacognitionEngine(FakeLLMService(payload_text()))
    with pytest.raises(TypeError, match="AgentContext"):
        engine.reflect({}, make_baseline(), {})


def test_engine_builds_request_and_returns_combined_result():
    fake = FakeLLMService(payload_text())
    engine = LLMMetacognitionEngine(fake)
    baseline = make_baseline()
    result = engine.reflect(
        {"episode_id": "ep-1", "success": False},
        baseline,
        make_context(),
        metadata={"episode_id": "ep-1"},
    )

    assert isinstance(result, LLMMetacognitionResult)
    assert result.deterministic_reflection is baseline
    assert result.llm_reflection.diagnosis.startswith("The action failed")
    assert result.provider == "fake"
    assert result.model == "fake-model"
    assert result.raw_content == payload_text()
    assert len(fake.calls) == 1
    prompt, kwargs = fake.calls[0]
    assert "DETERMINISTIC REFLECTION" in prompt
    assert "AGENT CONTEXT" in prompt
    assert kwargs["metadata"]["operation"] == "metacognition"
    assert kwargs["metadata"]["episode_id"] == "ep-1"


def test_engine_trace_is_updated():
    fake = FakeLLMService(payload_text())
    engine = LLMMetacognitionEngine(fake)
    result = engine.reflect({}, make_baseline(), make_context())
    assert engine.last_result is result
    assert engine.last_response is not None


def test_engine_clear_trace():
    fake = FakeLLMService(payload_text())
    engine = LLMMetacognitionEngine(fake)
    engine.reflect({}, make_baseline(), make_context())
    engine.clear_trace()
    assert engine.last_result is None
    assert engine.last_response is None


def test_result_exposes_llm_confidence_and_alignment():
    fake = FakeLLMService(payload_text())
    result = LLMMetacognitionEngine(fake).reflect(
        {}, make_baseline(), make_context()
    )
    assert result.confidence == 0.84
    assert result.alignment == "refines"


def test_result_to_dict_keeps_deterministic_and_llm_layers_separate():
    fake = FakeLLMService(payload_text())
    result = LLMMetacognitionEngine(fake).reflect(
        {}, make_baseline(), make_context()
    )
    data = result.to_dict()
    assert "deterministic_reflection" in data
    assert "llm_reflection" in data
    assert data["llm_reflection"]["alignment"] == "refines"


def test_engine_does_not_mutate_deterministic_reflection():
    fake = FakeLLMService(payload_text())
    baseline = make_baseline()
    before = baseline.to_dict() if hasattr(baseline, "to_dict") else baseline.__dict__.copy()
    LLMMetacognitionEngine(fake).reflect({}, baseline, make_context())
    after = baseline.to_dict() if hasattr(baseline, "to_dict") else baseline.__dict__.copy()
    assert after == before
