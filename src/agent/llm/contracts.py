from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class LLMMessage:
    """One normalized message sent to an LLM provider."""

    role: str
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _ALLOWED_ROLES:
            raise ValueError(
                f"Unsupported LLM message role: {self.role!r}"
            )

        if not isinstance(self.content, str):
            raise TypeError("LLM message content must be a string")


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral request contract."""

    messages: tuple[LLMMessage, ...]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("LLM request must contain at least one message")

        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0")

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral response contract."""

    content: str
    provider: str
    model: str | None = None
    finish_reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw: Any = None


class LLMProvider(Protocol):
    """Minimal provider interface required by the Agent."""

    @property
    def name(self) -> str:
        ...

    def complete(self, request: LLMRequest) -> LLMResponse:
        ...
