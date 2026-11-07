from __future__ import annotations

"""Try LLM providers in order; move to the next only on provider-level failure.

Only errors that mean "this provider is unavailable or broke" trigger a
fallback (CLI missing, timeout, non-zero exit, unparseable output). Programming
errors (TypeError, ValueError, ...) propagate immediately so bugs are never
hidden behind a silent switch of provider.

Note: a fallback sends the same prompt to the next provider. Only configure
providers you are willing to send that data to.
"""

import logging
import time
from typing import Sequence

from agent.llm.contracts import LLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class AllProvidersFailedError(RuntimeError):
    """Every configured provider failed; ``failures`` lists (provider, error)."""

    def __init__(self, failures: Sequence[tuple[str, Exception]]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(f"{name}: {exc}" for name, exc in self.failures)
        super().__init__(f"All LLM providers failed: {detail}")


def _default_fallback_errors() -> tuple[type[Exception], ...]:
    from agent.llm.claude_code_provider import ClaudeCodeProviderError
    from agent.llm.open_code_provider import OpenCodeProviderError

    return (OpenCodeProviderError, ClaudeCodeProviderError, TimeoutError, OSError)


class FallbackLLMProvider:
    name = "fallback"

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        fallback_on: tuple[type[Exception], ...] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = tuple(providers)
        self.fallback_on = fallback_on or _default_fallback_errors()
        self.last_used_provider: str | None = None
        self.last_failures: tuple[tuple[str, Exception], ...] = ()

    def complete(self, request: LLMRequest) -> LLMResponse:
        failures: list[tuple[str, Exception]] = []
        for provider in self.providers:
            started = time.monotonic()
            try:
                response = provider.complete(request)
            except self.fallback_on as exc:
                failures.append((str(provider.name), exc))
                logger.warning("LLM provider %s failed after %.0f s (%s: %s); trying next.",
                               provider.name, time.monotonic() - started, type(exc).__name__, exc)
                continue
            self.last_used_provider = str(provider.name)
            self.last_failures = tuple(failures)
            return response
        self.last_used_provider = None
        self.last_failures = tuple(failures)
        raise AllProvidersFailedError(failures)


__all__ = ["FallbackLLMProvider", "AllProvidersFailedError"]
