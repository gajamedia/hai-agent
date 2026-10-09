from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from agent.core.agent import Agent
from agent.llm import (
    LLMMessage,
    LLMResponse,
    LLMService,
)


class FakeProvider:
    name = "fake"

    def complete(self, request):
        return LLMResponse(
            content="hello from llm",
            provider=self.name,
            model=request.model or "fake-model",
        )


def build_agent():
    return Agent(name="llm-agent")


def test_agent_is_llm_unconfigured_by_default():
    agent = build_agent()

    assert agent.has_llm is False
    assert agent.llm_service is None
    assert agent.get_last_llm_response() is None


def test_agent_can_attach_llm_service():
    agent = build_agent()
    service = LLMService(FakeProvider())

    agent.set_llm_service(service)

    assert agent.has_llm is True
    assert agent.llm_service is service


def test_agent_llm_complete_text_delegates_to_service():
    agent = build_agent()
    agent.set_llm_service(LLMService(FakeProvider()))

    response = agent.llm_complete_text(
        "Say hello.",
        system_prompt="You are helpful.",
        model="fake-model",
    )

    assert response.content == "hello from llm"
    assert response.provider == "fake"
    assert agent.get_last_llm_response() is response
    assert agent.last_llm_request is not None
    assert len(agent.last_llm_request.messages) == 2


def test_agent_llm_complete_delegates_messages():
    agent = build_agent()
    agent.set_llm_service(LLMService(FakeProvider()))

    response = agent.llm_complete(
        [
            LLMMessage(role="user", content="Hello"),
        ],
    )

    assert response.content == "hello from llm"
    assert agent.last_llm_response is response


def test_agent_rejects_llm_call_when_unconfigured():
    agent = build_agent()

    with pytest.raises(RuntimeError, match="LLM service is not configured"):
        agent.llm_complete_text("Hello")


def test_replacing_llm_service_clears_previous_trace():
    agent = build_agent()
    first = LLMService(FakeProvider())
    second = LLMService(FakeProvider())

    agent.set_llm_service(first)
    agent.llm_complete_text("First")

    assert agent.last_llm_response is not None

    agent.set_llm_service(second)

    assert agent.last_llm_request is None
    assert agent.last_llm_response is None
    assert second.last_request is None
    assert second.last_response is None


def test_clear_llm_trace_clears_agent_and_service_trace():
    agent = build_agent()
    service = LLMService(FakeProvider())
    agent.set_llm_service(service)

    agent.llm_complete_text("Hello")
    assert agent.last_llm_response is not None

    agent.clear_llm_trace()

    assert agent.last_llm_request is None
    assert agent.last_llm_response is None
    assert service.last_request is None
    assert service.last_response is None
