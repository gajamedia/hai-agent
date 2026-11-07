from __future__ import annotations

"""LLM provider backed by the Claude Code CLI (``claude -p``).

Uses the user's already-logged-in Claude Code installation, so no API key is
stored or handled by this project. The model is used as a *pure text
completion engine*:

* all tools are disabled (``--tools ""``), so it cannot read files, run
  commands or browse - it only returns text;
* no session is persisted (``--no-session-persistence``);
* it runs from a neutral working directory so project instruction files are
  not picked up;
* the prompt is sent on stdin (no command-line length/quoting problems).

``temperature`` and ``max_tokens`` from the request are not supported by the
CLI and are ignored.
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent.llm.contracts import LLMMessage, LLMRequest, LLMResponse
from agent.llm.process_runner import run_cli, sanitized_environment

DEFAULT_PROVIDER_NAME = "claude-code"

# (command, cwd, timeout, env, stdin_text) -> CompletedProcess
ClaudeRunner = Callable[
    [Sequence[str], "str | Path | None", "float | None", "Mapping[str, str] | None", str],
    "subprocess.CompletedProcess[str]",
]


class ClaudeCodeProviderError(RuntimeError):
    """Base error for the Claude Code provider."""


class ClaudeCodeConfigurationError(ClaudeCodeProviderError):
    """The CLI is missing or cannot be started."""


class ClaudeCodeTimeoutError(ClaudeCodeProviderError):
    """The CLI did not answer within the configured timeout."""


class ClaudeCodeProcessError(ClaudeCodeProviderError):
    """The CLI exited unsuccessfully or reported an error result."""


class ClaudeCodeResponseParseError(ClaudeCodeProviderError):
    """The CLI output could not be parsed."""


def discover_claude_executable() -> str | None:
    """Find the Claude Code CLI: env override, PATH, then the VS Code extension."""
    configured = os.getenv("CLAUDE_CODE_EXECUTABLE")
    if configured and configured.strip():
        return configured.strip()
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    home = Path.home()
    patterns = [
        str(home / ".vscode" / "extensions" / "anthropic.claude-code-*" /
            "resources" / "native-binary" / ("claude.exe" if os.name == "nt" else "claude")),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]  # newest version by name
    return None


class ClaudeCodeProvider:
    name = DEFAULT_PROVIDER_NAME

    def __init__(
        self,
        *,
        executable: str | None = None,
        model: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = 120.0,
        env: Mapping[str, str] | None = None,
        runner: ClaudeRunner | None = None,
    ) -> None:
        resolved = executable or discover_claude_executable()
        if resolved is None and runner is None:
            raise ClaudeCodeConfigurationError(
                "Claude Code CLI was not found. Install Claude Code, add `claude` to "
                "PATH, or set CLAUDE_CODE_EXECUTABLE."
            )
        self.executable = resolved or "claude"
        self.default_model = model or os.getenv("CLAUDE_CODE_MODEL") or None
        self.cwd = Path(cwd) if cwd is not None else Path(tempfile.gettempdir())
        self.timeout = timeout
        self.environment = dict(env) if env is not None else None
        self._runner = runner

    # ------------------------------------------------------------------ API
    def complete(self, request: LLMRequest) -> LLMResponse:
        system_prompt, prompt = self._split_messages(request.messages)
        if not prompt.strip():
            raise ClaudeCodeResponseParseError("At least one non-system message is required.")
        model = request.model or self.default_model
        command = self.build_command(model=model, system_prompt=system_prompt)

        try:
            result = self._run(command, prompt)
        except FileNotFoundError as exc:
            raise ClaudeCodeConfigurationError(
                f"Claude Code executable was not found: {self.executable!r}."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeTimeoutError(
                f"Claude Code request timed out after {self.timeout} seconds."
            ) from exc
        except OSError as exc:
            raise ClaudeCodeConfigurationError(f"Failed to start Claude Code: {exc}") from exc

        parsed = self._parse_output(result.stdout or "")
        if result.returncode != 0 or parsed.get("is_error") is True:
            detail = (
                str(parsed.get("result") or "").strip()
                or (result.stderr or "").strip()
                or f"process exited with code {result.returncode}"
            )
            raise ClaudeCodeProcessError(f"Claude Code failed: {detail}")

        content = parsed.get("result")
        if not isinstance(content, str):
            raise ClaudeCodeResponseParseError("Claude Code output has no text result.")

        usage = dict(parsed.get("usage") or {})
        if "total_cost_usd" in parsed:
            usage["total_cost_usd"] = parsed["total_cost_usd"]
        model_usage = parsed.get("modelUsage")
        resolved_model = (
            next(iter(model_usage), None) if isinstance(model_usage, Mapping) and model_usage
            else model
        )
        return LLMResponse(
            content=content,
            provider=self.name,
            model=resolved_model,
            finish_reason=parsed.get("stop_reason"),
            usage=usage,
            raw={"session_id": parsed.get("session_id"), "returncode": result.returncode,
                 "command": list(command)},
        )

    # ------------------------------------------------------------- internals
    def build_command(self, *, model: str | None, system_prompt: str | None) -> list[str]:
        command = [
            self.executable, "-p",
            "--output-format", "json",
            "--tools", "",
            "--no-session-persistence",
        ]
        if model:
            command.extend(["--model", model])
        if system_prompt:
            command.extend(["--system-prompt", system_prompt])
        return command

    def _run(self, command: Sequence[str], stdin_text: str) -> "subprocess.CompletedProcess[str]":
        if self._runner is not None:
            return self._runner(command, self.cwd, self.timeout, self.environment, stdin_text)
        # No application secrets in the child's environment, no inherited stdin, and the whole
        # process tree is killed on timeout (see process_runner).
        return run_cli(list(command), cwd=self.cwd, timeout=self.timeout, stdin_text=stdin_text,
                       env=sanitized_environment(os.environ, self.environment))

    @staticmethod
    def _split_messages(messages: Sequence[LLMMessage]) -> tuple[str | None, str]:
        system_parts = [m.content for m in messages if m.role == "system" and m.content.strip()]
        sections = []
        for m in messages:
            if m.role == "system":
                continue
            header = f"[{m.role.upper()}]" + (f" name={m.name}" if m.name else "")
            sections.append(f"{header}\n{m.content}")
        return ("\n\n".join(system_parts) or None), "\n\n".join(sections)

    @staticmethod
    def _parse_output(stdout: str) -> dict[str, Any]:
        text = stdout.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Tolerate stray log lines before the JSON object.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise ClaudeCodeResponseParseError("Claude Code output is not valid JSON.") from None
            try:
                payload = json.loads(text[start:end + 1])
            except json.JSONDecodeError as exc:
                raise ClaudeCodeResponseParseError("Claude Code output is not valid JSON.") from exc
        if isinstance(payload, list):  # stream-style array: take the final result event
            payload = next((e for e in reversed(payload)
                            if isinstance(e, dict) and e.get("type") == "result"), {})
        if not isinstance(payload, dict):
            raise ClaudeCodeResponseParseError("Claude Code output has an unexpected shape.")
        return payload


__all__ = [
    "ClaudeCodeProvider", "ClaudeCodeProviderError", "ClaudeCodeConfigurationError",
    "ClaudeCodeTimeoutError", "ClaudeCodeProcessError", "ClaudeCodeResponseParseError",
    "discover_claude_executable",
]
