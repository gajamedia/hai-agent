from __future__ import annotations

import asyncio
import logging

from api.resource_lifecycle import close_owned_resources


class ShutdownOnlyResource:
    def __init__(self, events):
        self.events = events

    def shutdown(self):
        self.events.append("shutdown")


class AsyncShutdownOnlyResource:
    def __init__(self, events):
        self.events = events

    async def shutdown(self):
        self.events.append("async-shutdown")


class CloseAndShutdownResource:
    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append("close")

    def shutdown(self):
        self.events.append("shutdown")


class FailingShutdownResource:
    def __init__(self, events):
        self.events = events

    def shutdown(self):
        self.events.append("failed-shutdown")
        raise RuntimeError("cleanup failed")


def test_shutdown_only_resource_is_cleaned_up():
    events = []
    asyncio.run(close_owned_resources([ShutdownOnlyResource(events)]))
    assert events == ["shutdown"]


def test_async_shutdown_is_awaited():
    events = []
    asyncio.run(close_owned_resources([AsyncShutdownOnlyResource(events)]))
    assert events == ["async-shutdown"]


def test_close_has_precedence_over_shutdown():
    events = []
    asyncio.run(close_owned_resources([CloseAndShutdownResource(events)]))
    assert events == ["close"]


def test_shutdown_failure_does_not_prevent_other_cleanup(caplog):
    events = []
    with caplog.at_level(logging.ERROR):
        asyncio.run(
            close_owned_resources(
                [ShutdownOnlyResource(events), FailingShutdownResource(events)]
            )
        )

    # Reverse construction order: failing resource first, then healthy resource.
    assert events == ["failed-shutdown", "shutdown"]
    assert "Failed to close application-owned resource" in caplog.text
