from __future__ import annotations

import json
import subprocess

import pytest

from agent.llm import LLMMessage, LLMRequest
from agent.llm.open_code_provider import (
    OpenCodeConfigurationError,
    OpenCodeOutputParser,
    OpenCodeProcessError,
    OpenCodeProvider,
    OpenCodeResponseParseError,
    OpenCodeTimeoutError,
)


def completed(
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
):
    return subprocess.CompletedProcess(
        args=["opencode"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_provider_name():
    provider = OpenCodeProvider()

    assert provider.name == "opencode"


def test_command_uses_json_format():
    provider = OpenCodeProvider()

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="hello",
            ),
        )
    )

    command = provider.build_command(request)

    assert command[:4] == [
        "opencode",
        "run",
        "--format",
        "json",
    ]


def test_request_model_overrides_provider_model():
    provider = OpenCodeProvider(
        model="provider-model",
    )

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="hello",
            ),
        ),
        model="request-model",
    )

    command = provider.build_command(request)

    assert command[
        command.index("--model") + 1
    ] == "request-model"


def test_provider_model_is_used_when_request_has_none():
    provider = OpenCodeProvider(
        model="provider-model",
    )

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="hello",
            ),
        )
    )

    command = provider.build_command(request)

    assert command[
        command.index("--model") + 1
    ] == "provider-model"


def test_messages_are_rendered_as_role_sections():
    provider = OpenCodeProvider()

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="system",
                content="Be concise",
            ),
            LLMMessage(
                role="user",
                content="Hello",
                name="tester",
            ),
        )
    )

    command = provider.build_command(request)
    prompt = command[-1]

    assert "[SYSTEM]" in prompt
    assert "Be concise" in prompt
    assert "[USER] name=tester" in prompt
    assert "Hello" in prompt


def test_optional_opencode_flags_are_added():
    provider = OpenCodeProvider(
        model="demo-model",
        agent="build",
        variant="high",
        auto_approve=True,
        include_thinking=True,
        files=[
            "a.py",
            "b.py",
        ],
    )

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="Review files",
            ),
        )
    )

    command = provider.build_command(request)

    assert "--agent" in command
    assert "build" in command
    assert "--variant" in command
    assert "high" in command
    assert "--thinking" in command
    assert "--auto" in command
    assert command.count("--file") == 2


def test_json_text_event_is_parsed():
    stdout = json.dumps(
        {
            "type": "text",
            "sessionID": "ses_123",
            "part": {
                "type": "text",
                "text": "Hello agent",
            },
        }
    )

    parsed = OpenCodeOutputParser.parse(stdout)

    assert parsed["content"] == "Hello agent"
    assert parsed["session_id"] == "ses_123"
    assert parsed["event_count"] == 1


def test_multiple_text_events_are_concatenated():
    lines = [
        json.dumps(
            {
                "type": "step_start",
                "sessionID": "ses_1",
                "part": {"type": "step-start"},
            }
        ),
        json.dumps(
            {
                "type": "text",
                "sessionID": "ses_1",
                "part": {
                    "type": "text",
                    "text": "Hello ",
                },
            }
        ),
        json.dumps(
            {
                "type": "text",
                "sessionID": "ses_1",
                "part": {
                    "type": "text",
                    "text": "world",
                },
            }
        ),
    ]

    parsed = OpenCodeOutputParser.parse(
        "\n".join(lines)
    )

    assert parsed["content"] == "Hello world"


def test_step_finish_usage_and_reason_are_extracted():
    lines = [
        json.dumps(
            {
                "type": "text",
                "sessionID": "ses_2",
                "part": {
                    "type": "text",
                    "text": "Done",
                },
            }
        ),
        json.dumps(
            {
                "type": "step_finish",
                "sessionID": "ses_2",
                "part": {
                    "type": "step-finish",
                    "reason": "stop",
                    "tokens": {
                        "input": 12,
                        "output": 4,
                        "total": 16,
                    },
                    "cost": 0,
                },
            }
        ),
    ]

    parsed = OpenCodeOutputParser.parse(
        "\n".join(lines)
    )

    assert parsed["finish_reason"] == "stop"
    assert parsed["usage"]["input"] == 12
    assert parsed["usage"]["output"] == 4
    assert parsed["usage"]["total"] == 16
    assert parsed["usage"]["cost"] == 0


def test_synthetic_compaction_text_is_ignored():
    lines = [
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "Visible response",
                },
            }
        ),
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "Internal summary",
                    "synthetic": True,
                },
            }
        ),
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "Continue internally",
                    "metadata": {
                        "compaction_continue": True,
                    },
                },
            }
        ),
    ]

    parsed = OpenCodeOutputParser.parse(
        "\n".join(lines)
    )

    assert parsed["content"] == "Visible response"


def test_reasoning_events_are_not_returned_as_content():
    lines = [
        json.dumps(
            {
                "type": "reasoning",
                "part": {
                    "type": "reasoning",
                    "text": "private reasoning",
                },
            }
        ),
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "final answer",
                },
            }
        ),
    ]

    parsed = OpenCodeOutputParser.parse(
        "\n".join(lines)
    )

    assert parsed["content"] == "final answer"


def test_no_text_event_raises_parse_error():
    stdout = json.dumps(
        {
            "type": "step_start",
            "part": {
                "type": "step-start",
            },
        }
    )

    with pytest.raises(
        OpenCodeResponseParseError,
        match="no user-visible text event",
    ):
        OpenCodeOutputParser.parse(stdout)


def test_plain_text_fallback_can_be_used():
    parsed = OpenCodeOutputParser.parse(
        "plain text response",
    )

    assert parsed["content"] == "plain text response"


def test_provider_executes_runner_and_returns_response():
    stdout = json.dumps(
        {
            "type": "text",
            "sessionID": "ses_realistic",
            "part": {
                "type": "text",
                "text": "hello from opencode",
            },
        }
    )

    calls = []

    def runner(
        command,
        cwd,
        timeout,
        env,
    ):
        calls.append(
            {
                "command": list(command),
                "cwd": cwd,
                "timeout": timeout,
                "env": env,
            }
        )

        return completed(stdout)

    provider = OpenCodeProvider(
        model="demo-model",
        runner=runner,
    )

    response = provider.complete(
        LLMRequest(
            messages=(
                LLMMessage(
                    role="user",
                    content="Say hello",
                ),
            )
        )
    )

    assert response.content == "hello from opencode"
    assert response.provider == "opencode"
    assert response.model == "demo-model"
    assert response.raw["returncode"] == 0
    assert calls[0]["command"][0:4] == [
        "opencode",
        "run",
        "--format",
        "json",
    ]


def test_nonzero_exit_code_raises_process_error():
    def runner(
        command,
        cwd,
        timeout,
        env,
    ):
        return completed(
            "",
            returncode=2,
            stderr="model unavailable",
        )

    provider = OpenCodeProvider(
        runner=runner,
    )

    with pytest.raises(
        OpenCodeProcessError,
        match="model unavailable",
    ):
        provider.complete(
            LLMRequest(
                messages=(
                    LLMMessage(
                        role="user",
                        content="hello",
                    ),
                )
            )
        )


def test_timeout_is_normalized():
    def runner(
        command,
        cwd,
        timeout,
        env,
    ):
        raise subprocess.TimeoutExpired(
            cmd=list(command),
            timeout=timeout,
        )

    provider = OpenCodeProvider(
        timeout=5,
        runner=runner,
    )

    with pytest.raises(
        OpenCodeTimeoutError,
        match="timed out",
    ):
        provider.complete(
            LLMRequest(
                messages=(
                    LLMMessage(
                        role="user",
                        content="hello",
                    ),
                )
            )
        )


def test_missing_executable_is_normalized():
    def runner(
        command,
        cwd,
        timeout,
        env,
    ):
        raise FileNotFoundError(
            "opencode"
        )

    provider = OpenCodeProvider(
        runner=runner,
    )

    with pytest.raises(
        OpenCodeConfigurationError,
        match="executable was not found",
    ):
        provider.complete(
            LLMRequest(
                messages=(
                    LLMMessage(
                        role="user",
                        content="hello",
                    ),
                )
            )
        )


# def test_empty_messages_are_rejected():
#     provider = OpenCodeProvider()

#     with pytest.raises(
#         OpenCodeResponseParseError,
#         match="At least one LLM message",
#     ):
#         provider.complete(
#             LLMRequest(
#                 messages=(),
#             )
#         )
def test_empty_messages_are_rejected_by_llm_contract():
    with pytest.raises(
        ValueError,
        match="LLM request must contain at least one message",
    ):
        LLMRequest(
            messages=(),
        )
