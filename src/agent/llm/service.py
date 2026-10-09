from __future__ import annotations

from typing import Any, Iterable, Mapping

from .contracts import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
)


class LLMService:
    """
    Provider-neutral orchestration boundary for Agent ↔ LLM calls.

    The service owns request normalization and the latest-call trace,
    while the provider owns the actual model/API transport.
    """

    def __init__(self, provider: LLMProvider) -> None:
        if provider is None:
            raise ValueError("provider is required")

        self.provider = provider
        self.last_request: LLMRequest | None = None
        self.last_response: LLMResponse | None = None

    @property
    def provider_name(self) -> str:
        return str(self.provider.name)

    def complete(
        self,
        messages: Iterable[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        request = LLMRequest(
            messages=tuple(messages),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=dict(metadata or {}),
        )

        response = self.provider.complete(request)

        if not isinstance(response, LLMResponse):
            raise TypeError(
                "LLM provider must return an LLMResponse instance"
            )

        self.last_request = request
        self.last_response = response

        return response

    def complete_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        messages: list[LLMMessage] = []

        if system_prompt is not None:
            messages.append(
                LLMMessage(
                    role="system",
                    content=system_prompt,
                )
            )

        messages.append(
            LLMMessage(
                role="user",
                content=prompt,
            )
        )

        return self.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )

    def clear_trace(self) -> None:
        """Clear the latest request/response trace."""

        self.last_request = None
        self.last_response = None
