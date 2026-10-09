from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent.llm.contracts import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
)


DEFAULT_EXECUTABLE = "opencode"
DEFAULT_PROVIDER_NAME = "opencode"


class OpenCodeProviderError(RuntimeError):
    """Base error for the OpenCode LLM provider."""


class OpenCodeConfigurationError(OpenCodeProviderError):
    """Raised when OpenCode cannot be started or is misconfigured."""


class OpenCodeTimeoutError(OpenCodeProviderError):
    """Raised when an OpenCode request exceeds the configured timeout."""


class OpenCodeProcessError(OpenCodeProviderError):
    """Raised when the OpenCode process exits unsuccessfully."""


class OpenCodeResponseParseError(OpenCodeProviderError):
    """Raised when OpenCode output cannot be normalized to a response."""


Runner = Callable[
    [Sequence[str], str | Path | None, float | None, Mapping[str, str] | None],
    subprocess.CompletedProcess[str],
]


class OpenCodeOutputParser:
    """
    Parse the JSONL emitted by ``opencode run --format json``.

    OpenCode documents ``--format json`` as raw JSON events. The
    current CLI emits text events containing a ``part`` object with
    ``type == "text"`` and a ``text`` field. This parser intentionally
    accepts a small family of equivalent shapes so minor event-wrapper
    changes do not leak into the Agent's LLM contract.

    Internal/synthetic compaction text is ignored when the event marks
    it as synthetic or as a compaction continuation.
    """

    TEXT_EVENT_TYPES = {
        "text",
        "text.delta",
        "message.part.updated",
    }

    REASONING_EVENT_TYPES = {
        "reasoning",
        "reasoning.delta",
        "message.reasoning",
    }

    @classmethod
    def parse(
        cls,
        stdout: str,
        *,
        allow_plain_text_fallback: bool = True,
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        event_count = 0
        session_id: str | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] = {}

        for raw_line in stdout.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            event_count += 1

            if session_id is None:
                value = event.get("sessionID")
                if value is None:
                    value = event.get("session_id")
                if value is not None:
                    session_id = str(value)

            event_type = event.get("type")

            if event_type in cls.REASONING_EVENT_TYPES:
                continue

            text_value = cls._extract_text(event)

            if text_value is not None and cls._is_user_visible_text(event):
                text_parts.append(text_value)

            event_part = event.get("part")

            if isinstance(event_part, dict):
                part_type = event_part.get("type")

                if part_type in {"step-finish", "step_finish"}:
                    reason = event_part.get("reason")
                    if reason is not None:
                        finish_reason = str(reason)

                    tokens = event_part.get("tokens")
                    if isinstance(tokens, Mapping):
                        usage.update(dict(tokens))

                    cost = event_part.get("cost")
                    if cost is not None:
                        usage["cost"] = cost

        content = "".join(text_parts)

        if not content.strip() and allow_plain_text_fallback:
            non_json_lines = []

            for raw_line in stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    non_json_lines.append(line)

            if non_json_lines:
                content = "\n".join(non_json_lines)

        if not content.strip():
            raise OpenCodeResponseParseError(
                "OpenCode completed but no user-visible text event was found."
            )

        return {
            "content": content,
            "session_id": session_id,
            "finish_reason": finish_reason,
            "usage": usage,
            "event_count": event_count,
        }

    @classmethod
    def _extract_text(cls, event: Mapping[str, Any]) -> str | None:
        event_type = event.get("type")

        if event_type not in cls.TEXT_EVENT_TYPES:
            part = event.get("part")
            if not (
                isinstance(part, Mapping)
                and part.get("type") == "text"
            ):
                return None

        part = event.get("part")

        if isinstance(part, Mapping):
            value = part.get("text")
            if isinstance(value, str):
                return value

            value = part.get("delta")
            if isinstance(value, str):
                return value

            if isinstance(value, Mapping):
                nested = value.get("text")
                if isinstance(nested, str):
                    return nested

        delta = event.get("delta")

        if isinstance(delta, str):
            return delta

        if isinstance(delta, Mapping):
            value = delta.get("text")
            if isinstance(value, str):
                return value

        value = event.get("text")
        if isinstance(value, str):
            return value

        return None

    @staticmethod
    def _is_user_visible_text(
        event: Mapping[str, Any],
    ) -> bool:
        part = event.get("part")

        if not isinstance(part, Mapping):
            part = {}

        if part.get("synthetic") is True:
            return False

        metadata = part.get("metadata")

        if isinstance(metadata, Mapping):
            if metadata.get("compaction_continue") is True:
                return False

        if event.get("synthetic") is True:
            return False

        return True


class OpenCodeProvider:
    """
    Provider adapter that uses the OpenCode CLI as a temporary LLM backend.

    The Agent only sees the provider-neutral ``LLMProvider`` contract.
    OpenCode itself is launched as a child process:

        LLMRequest
          -> opencode run --format json
          -> JSONL events
          -> OpenCodeOutputParser
          -> LLMResponse

    No OpenAI SDK or OpenAI API is required.
    """

    name = DEFAULT_PROVIDER_NAME

    def __init__(
        self,
        *,
        executable: str | None = None,
        model: str | None = None,
        agent: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = 120.0,
        files: Sequence[str | Path] | None = None,
        auto_approve: bool = False,
        include_thinking: bool = False,
        variant: str | None = None,
        env: Mapping[str, str] | None = None,
        runner: Runner | None = None,
    ) -> None:
        self.executable = (
            executable
            or os.getenv("OPENCODE_EXECUTABLE")
            or DEFAULT_EXECUTABLE
        )

        self.default_model = (
            model
            or os.getenv("OPENCODE_MODEL")
        )

        self.default_agent = (
            agent
            or os.getenv("OPENCODE_AGENT")
        )

        self.cwd = (
            Path(cwd)
            if cwd is not None
            else None
        )

        self.timeout = timeout

        self.files = tuple(
            str(item)
            for item in (files or ())
        )

        self.auto_approve = auto_approve
        self.include_thinking = include_thinking
        self.variant = variant

        self.environment = (
            dict(env)
            if env is not None
            else None
        )

        self._runner = runner

    def complete(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        if not request.messages:
            raise OpenCodeResponseParseError(
                "At least one LLM message is required."
            )

        command = self.build_command(request)

        try:
            result = self._run_command(command)
        except FileNotFoundError as exc:
            raise OpenCodeConfigurationError(
                f"OpenCode executable was not found: {self.executable!r}. "
                "Make sure `opencode` is installed and available on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise OpenCodeTimeoutError(
                f"OpenCode request timed out after {self.timeout} seconds."
            ) from exc
        except OSError as exc:
            raise OpenCodeConfigurationError(
                f"Failed to start OpenCode: {exc}"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            detail = (
                stderr
                if stderr
                else f"process exited with code {result.returncode}"
            )

            raise OpenCodeProcessError(
                f"OpenCode failed: {detail}"
            )

        parsed = OpenCodeOutputParser.parse(
            result.stdout or "",
        )

        resolved_model = (
            request.model
            or self.default_model
        )

        metadata = dict(request.metadata)

        metadata.update(
            {
                "opencode": {
                    "event_count": parsed["event_count"],
                    "session_id": parsed["session_id"],
                    "executable": self.executable,
                }
            }
        )

        return LLMResponse(
            content=parsed["content"],
            provider=self.name,
            model=resolved_model,
            finish_reason=parsed["finish_reason"],
            usage=parsed["usage"],
            raw={
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "returncode": result.returncode,
                "command": list(command),
            },
        )

    def build_command(
        self,
        request: LLMRequest,
    ) -> list[str]:
        prompt = self._render_messages(
            request.messages
        )

        model = (
            request.model
            or self.default_model
        )

        command = [
            self.executable,
            "run",
            "--format",
            "json",
        ]

        if model:
            command.extend(
                [
                    "--model",
                    model,
                ]
            )

        if self.default_agent:
            command.extend(
                [
                    "--agent",
                    self.default_agent,
                ]
            )

        if self.variant:
            command.extend(
                [
                    "--variant",
                    self.variant,
                ]
            )

        if self.include_thinking:
            command.append("--thinking")

        if self.auto_approve:
            command.append("--auto")

        for file_path in self.files:
            command.extend(
                [
                    "--file",
                    file_path,
                ]
            )

        command.append(prompt)

        return command

    def _run_command(
        self,
        command: Sequence[str],
    ) -> subprocess.CompletedProcess[str]:
        if self._runner is not None:
            return self._runner(
                command,
                self.cwd,
                self.timeout,
                self.environment,
            )

        environment = None

        if self.environment is not None:
            environment = dict(os.environ)
            environment.update(self.environment)

        return subprocess.run(
            list(command),
            cwd=self.cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )

    @staticmethod
    def _render_messages(
        messages: Sequence[LLMMessage],
    ) -> str:
        sections: list[str] = []

        for message in messages:
            role = str(message.role).upper()

            header = f"[{role}]"

            if getattr(message, "name", None):
                header += f" name={message.name}"

            sections.append(
                f"{header}\n{message.content}"
            )

        return "\n\n".join(sections)
