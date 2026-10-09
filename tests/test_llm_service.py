from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from agent.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMService,
)


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.requests = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content="fake response",
            provider=self.name,
            model=request.model or "fake-model",
        )


class InvalidProvider:
    name = "invalid"

    def complete(self, request: LLMRequest):
        return "not an LLMResponse"


def test_service_requires_provider():
    with pytest.raises(ValueError):
        LLMService(None)


def test_complete_builds_request_and_preserves_trace():
    provider = FakeProvider()
    service = LLMService(provider)

    response = service.complete(
        [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello"),
        ],
        model="demo-model",
        temperature=0.2,
        max_tokens=50,
        metadata={"purpose": "test"},
    )

    assert response.content == "fake response"
    assert service.provider_name == "fake"
    assert service.last_response is response
    assert service.last_request is not None
    assert service.last_request.model == "demo-model"
    assert service.last_request.temperature == 0.2
    assert service.last_request.max_tokens == 50
    assert service.last_request.metadata["purpose"] == "test"
    assert len(provider.requests) == 1


def test_complete_text_creates_system_and_user_messages():
    provider = FakeProvider()
    service = LLMService(provider)

    service.complete_text(
        "Explain this task.",
        system_prompt="You are an agent.",
    )

    request = provider.requests[0]

    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[0].content == "You are an agent."
    assert request.messages[1].role == "user"
    assert request.messages[1].content == "Explain this task."


def test_complete_text_without_system_prompt_has_one_user_message():
    provider = FakeProvider()
    service = LLMService(provider)

    service.complete_text("Hello")

    request = provider.requests[0]

    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.messages[0].content == "Hello"


def test_complete_text_requires_string_prompt():
    provider = FakeProvider()
    service = LLMService(provider)

    with pytest.raises(TypeError):
        service.complete_text(123)


def test_service_rejects_invalid_provider_response():
    service = LLMService(InvalidProvider())

    with pytest.raises(TypeError):
        service.complete_text("Hello")


def test_clear_trace_removes_latest_call():
    provider = FakeProvider()
    service = LLMService(provider)

    service.complete_text("Hello")
    assert service.last_request is not None
    assert service.last_response is not None

    service.clear_trace()

    assert service.last_request is None
    assert service.last_response is None
