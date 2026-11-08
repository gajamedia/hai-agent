from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapabilityExecutionResourceBundle:
    """Transfer explicitly owned resources from a gateway factory to the app.

    ``owned_resources`` must list resources in construction order. The app
    closes them in reverse order. Dependencies not transferred here remain
    borrowed and must not be closed by the app.

    The gateway itself is not implicitly owned; include it in
    ``owned_resources`` only when the factory intentionally transfers its
    ownership to the application.
    """

    gateway: Any
    owned_resources: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.owned_resources, tuple):
            raise TypeError("owned_resources must be a tuple.")
        if any(resource is None for resource in self.owned_resources):
            raise ValueError("owned_resources cannot contain None.")


class BackgroundTaskRegistry:
    """Track app-owned background work and prevent new work during shutdown.

    This registry is process-local. Durable batch ownership remains enforced
    by SQLiteTestResultRepository; the registry only coordinates this app's
    graceful shutdown and never authorizes replay.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active: set[str] = set()
        self._accepting = True

    def register(self, task_id: str) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        with self._condition:
            if not self._accepting:
                raise RuntimeError(
                    "Application is shutting down; background work is disabled"
                )
            if task_id in self._active:
                raise RuntimeError(f"Background task is already registered: {task_id}")
            self._active.add(task_id)

    def complete(self, task_id: str) -> None:
        with self._condition:
            self._active.discard(task_id)
            self._condition.notify_all()

    def stop_accepting(self) -> None:
        with self._condition:
            self._accepting = False
            self._condition.notify_all()

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait until all registered work completes; return False on timeout."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._active:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(sorted(self._active))

    @property
    def accepting(self) -> bool:
        with self._condition:
            return self._accepting


async def _await_completion_despite_cancellation(awaitable) -> bool:
    """Finish an owned cleanup awaitable even if its caller is cancelled.

    Returns whether cancellation of the caller was observed. The caller may
    defer re-raising it until all owned resources have been attempted.
    """
    task = asyncio.ensure_future(awaitable)
    cancellation_seen = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
            cancellation_seen = True
    # Surface cleanup-task failures to the caller for logging.
    task.result()
    return cancellation_seen


async def close_owned_resources(
    resources,
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Close transferred resources in reverse order, best-effort and once.

    Supported cleanup methods are selected in this order:
    ``aclose()``, ``close()``, then ``shutdown()``. The shutdown method is a
    fallback for managers whose public lifecycle API is named ``shutdown``.
    Both synchronous and asynchronous implementations are supported,
    including methods that return an awaitable.

    Cancellation is deferred until every resource has had a cleanup attempt.
    The boolean return indicates whether the calling task was cancelled while
    cleanup was in progress; the application lifespan decides when to re-raise.
    """
    log = logger or logging.getLogger(__name__)
    seen: set[int] = set()
    unique_resources = []
    for resource in tuple(resources):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        unique_resources.append(resource)

    cancellation_seen = False
    for resource in reversed(unique_resources):
        try:
            closer = getattr(resource, "aclose", None)
            if not callable(closer):
                closer = getattr(resource, "close", None)
            if not callable(closer):
                closer = getattr(resource, "shutdown", None)
            if not callable(closer):
                continue

            result = closer()
            if inspect.isawaitable(result):
                cancellation_seen = (
                    await _await_completion_despite_cancellation(result)
                    or cancellation_seen
                )
        except asyncio.CancelledError:
            # A synchronous closer should not normally raise this, but do not
            # let it prevent attempts to close the remaining owned resources.
            cancellation_seen = True
        except Exception:
            log.exception(
                "Failed to close application-owned resource of type %s.",
                type(resource).__name__,
            )
    return cancellation_seen


__all__ = [
    "CapabilityExecutionResourceBundle",
    "BackgroundTaskRegistry",
    "close_owned_resources",
]
