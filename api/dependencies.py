
from __future__ import annotations

import os

from fastapi import Request

from agent.core.agent import Agent
from agent.llm.agent_runtime import AgentCognitiveRuntime
from agent.llm.metacognition import LLMMetacognitionEngine
from agent.llm.open_code_provider import OpenCodeProvider
from agent.llm.orchestrator import CognitiveOrchestrator
from agent.llm.service import LLMService


def create_runtime() -> AgentCognitiveRuntime:
    """Construct the Agent and the complete cognitive pipeline."""
    timeout_value = os.getenv("OPENCODE_TIMEOUT", "120")

    try:
        timeout = float(timeout_value)
    except ValueError as exc:
        raise RuntimeError(
            "OPENCODE_TIMEOUT must be a number of seconds"
        ) from exc

    if timeout <= 0:
        raise RuntimeError(
            "OPENCODE_TIMEOUT must be greater than zero"
        )

    provider = OpenCodeProvider(
        executable=os.getenv("OPENCODE_EXECUTABLE") or None,
        model=os.getenv("OPENCODE_MODEL") or None,
        agent=os.getenv("OPENCODE_AGENT") or None,
        cwd=os.getenv("OPENCODE_CWD") or None,
        timeout=timeout,
        auto_approve=False,
    )

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
