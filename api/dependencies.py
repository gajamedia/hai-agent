
from __future__ import annotations

import os

from fastapi import Request

from agent.core.agent import Agent
from agent.llm.agent_runtime import AgentCognitiveRuntime
from agent.llm.metacognition import LLMMetacognitionEngine
from agent.llm.claude_code_provider import (
    ClaudeCodeProvider,
    discover_claude_executable,
)
from agent.llm.fallback_provider import FallbackLLMProvider
from agent.llm.open_code_provider import OpenCodeProvider
from agent.llm.orchestrator import CognitiveOrchestrator
from agent.llm.service import LLMService


def _positive_timeout(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number of seconds") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def create_llm_provider():
    """Build the LLM provider selected by HAI_LLM_PROVIDER.

    * ``auto`` (default): OpenCode first; Claude Code CLI as a fallback when it
      is installed. Without the Claude CLI this is just OpenCode.
    * ``opencode``: OpenCode only (previous behaviour).
    * ``claude-code``: Claude Code CLI only.
    """
    mode = os.getenv("HAI_LLM_PROVIDER", "auto").strip().lower()
    if mode not in {"auto", "opencode", "claude-code"}:
        raise RuntimeError("HAI_LLM_PROVIDER must be 'auto', 'opencode' or 'claude-code'")

    def opencode(*, first_of_two: bool = False):
        timeout = _positive_timeout("OPENCODE_TIMEOUT", "120")
        if first_of_two:
            # A provider that hangs only fails when its timeout ends, and only then does the next
            # one get a turn. Cap the first one so a stuck OpenCode costs seconds, not minutes.
            timeout = min(timeout, _positive_timeout("HAI_LLM_FAILOVER_SECONDS", "60"))
        return OpenCodeProvider(
            executable=os.getenv("OPENCODE_EXECUTABLE") or None,
            model=os.getenv("OPENCODE_MODEL") or None,
            agent=os.getenv("OPENCODE_AGENT") or None,
            cwd=os.getenv("OPENCODE_CWD") or None,
            timeout=timeout,
            auto_approve=False,
        )

    def claude():
        return ClaudeCodeProvider(
            executable=discover_claude_executable(),
            timeout=_positive_timeout("CLAUDE_CODE_TIMEOUT", "120"),
        )

    if mode == "opencode":
        return opencode()
    if mode == "claude-code":
        return claude()
    if discover_claude_executable() is None:
        return opencode()
    return FallbackLLMProvider(
        [opencode(first_of_two=True), claude()],
        # After a failure (an account at its usage limit, a stuck model) skip that provider for a
        # while, so only the first request pays for it instead of every request.
        cooldown_seconds=_positive_timeout("HAI_LLM_PROVIDER_COOLDOWN_SECONDS", "300"),
    )


def create_runtime() -> AgentCognitiveRuntime:
    """Construct the Agent and the complete cognitive pipeline."""
    provider = create_llm_provider()

    llm_service = LLMService(provider)

    memory_path = os.getenv("HAI_MEMORY_PATH", "data/memory.db")

    agent = Agent(
        os.getenv("HAI_AGENT_NAME", "hAI"),
        memory_path=memory_path,
        llm_service=llm_service,
    )

    metacognition_engine = LLMMetacognitionEngine(
        llm_service,
    )

    orchestrator = CognitiveOrchestrator(
        llm_service,
        metacognition_engine=metacognition_engine,
    )

    return AgentCognitiveRuntime(
        agent,
        orchestrator,
    )


def get_runtime(request: Request) -> AgentCognitiveRuntime:
    runtime = getattr(
        request.app.state,
        "cognitive_runtime",
        None,
    )

    if runtime is None:
        raise RuntimeError(
            "Cognitive runtime has not been initialized"
        )

    return runtime
