from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnifiedLearnedStateRestorationResult:
    """
    Result of restoring durable learned-agent state.

    Durable state currently consists of:

        1. Agent self-awareness state
        2. Calibrated self-aware decision influence

    Runtime-only telemetry and feedback traces are intentionally
    not represented as restored state.
    """

    agent_name: str

    self_awareness_restored: bool
    calibrated_influence_restored: bool

    self_awareness_version: int | None
    calibrated_influence_version: int | None

    consistent: bool
    reason: str

    @property
    def has_learned_state(self) -> bool:
        return (
            self.self_awareness_restored
            or self.calibrated_influence_restored
        )

    @property
    def fully_restored(self) -> bool:
        return (
            self.self_awareness_restored
            and self.calibrated_influence_restored
            and self.consistent
        )

    @property
    def partially_restored(self) -> bool:
        return (
            self.has_learned_state
            and not self.fully_restored
        )

    @property
    def snapshot(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "self_awareness_restored": (
                self.self_awareness_restored
            ),
            "calibrated_influence_restored": (
                self.calibrated_influence_restored
            ),
            "self_awareness_version": (
                self.self_awareness_version
            ),
            "calibrated_influence_version": (
                self.calibrated_influence_version
            ),
            "consistent": self.consistent,
            "reason": self.reason,
            "has_learned_state": (
                self.has_learned_state
            ),
            "fully_restored": (
                self.fully_restored
            ),
            "partially_restored": (
                self.partially_restored
            ),
        }