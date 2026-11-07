from __future__ import annotations

"""Try LLM providers in order; move to the next only on provider-level failure.

Only errors that mean "this provider is unavailable or broke" trigger a
fallback (CLI missing, timeout, non-zero exit, unparseable output). Programming
errors (TypeError, ValueError, ...) propagate immediately so bugs are never
hidden behind a silent switch of provider.

Cooldown: a provider that just failed (for example an account that hit its usage limit) will most
likely fail again, and every attempt costs its whole timeout. With ``cooldown_seconds`` > 0 a failed
provider is skipped for that long, so only the first request pays for the failure. A provider in
cooldown is still tried as a last resort when every other provider fails, and one success clears it.

Note: a fallback sends the same prompt to the next provider. Only configure
providers you are willing to send that data to.
"""

import logging
import threading
import time
from typing import Callable, Sequence

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
        cooldown_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        if isinstance(cooldown_seconds, bool) or cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be zero or positive")
        self.providers = tuple(providers)
        self.fallback_on = fallback_on or _default_fallback_errors()
        self.cooldown_seconds = float(cooldown_seconds)
        self._clock = clock
        self._failed_at: dict[int, float] = {}
        self._lock = threading.Lock()
        self.last_used_provider: str | None = None
        self.last_failures: tuple[tuple[str, Exception], ...] = ()

    def _order(self) -> tuple[list[LLMProvider], list[LLMProvider]]:
        """(providers to try now, providers resting after a recent failure)."""
        if self.cooldown_seconds <= 0:
            return list(self.providers), []
        now = self._clock()
        with self._lock:
            resting = [p for p in self.providers
                       if id(p) in self._failed_at and now - self._failed_at[id(p)] < self.cooldown_seconds]
        active = [p for p in self.providers if p not in resting]
        if not active:                      # everything is resting: do not refuse to try at all
            return list(self.providers), []
        return active, resting

    def _record_failure(self, provider: LLMProvider) -> None:
        if self.cooldown_seconds > 0:
            with self._lock:
                self._failed_at[id(provider)] = self._clock()

    def _record_success(self, provider: LLMProvider) -> None:
        with self._lock:
            self._failed_at.pop(id(provider), None)

    def complete(self, request: LLMRequest) -> LLMResponse:
        failures: list[tuple[str, Exception]] = []
        active, resting = self._order()
        for provider in resting:
            logger.warning("LLM provider %s failed recently; skipping it for %.0f s.",
                           provider.name, self.cooldown_seconds)
        for provider in [*active, *resting]:     # resting providers only as a last resort
            started = time.monotonic()
            try:
                response = provider.complete(request)
            except self.fallback_on as exc:
                failures.append((str(provider.name), exc))
                self._record_failure(provider)
                logger.warning("LLM provider %s failed after %.0f s (%s: %s); trying next.",
                               provider.name, time.monotonic() - started, type(exc).__name__, exc)
                continue
            self._record_success(provider)
            self.last_used_provider = str(provider.name)
            self.last_failures = tuple(failures)
            return response
        self.last_used_provider = None
        self.last_failures = tuple(failures)
        raise AllProvidersFailedError(failures)


__all__ = ["FallbackLLMProvider", "AllProvidersFailedError"]
