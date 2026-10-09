from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from agent.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
)


def test_message_accepts_supported_role():
    message = LLMMessage(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"


def test_message_rejects_unknown_role():
    with pytest.raises(ValueError):
        LLMMessage(role="developer", content="hello")


def test_request_requires_messages():
    with pytest.raises(ValueError):
        LLMRequest(messages=())


def test_request_rejects_invalid_temperature():
    with pytest.raises(ValueError):
        LLMRequest(
            messages=(
                LLMMessage(role="user", content="hello"),
            ),
            temperature=-0.1,
        )


def test_request_rejects_invalid_max_tokens():
    with pytest.raises(ValueError):
        LLMRequest(
            messages=(
                LLMMessage(role="user", content="hello"),
            ),
            max_tokens=0,
        )


def test_response_is_provider_neutral():
    response = LLMResponse(
        content="answer",
        provider="test",
        model="test-model",
    )

    assert response.content == "answer"
    assert response.provider == "test"
    assert response.model == "test-model"
