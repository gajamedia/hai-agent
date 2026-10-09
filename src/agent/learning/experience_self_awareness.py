from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from agent.learning.agent_self_awareness import (
    AgentSelfAwareness,
    AgentSelfAwarenessSnapshot,
)


@dataclass(frozen=True)
class ExperienceSelfAwarenessUpdateResult:
    """
    Result of updating self-awareness from experience
    history.
    """

    updated: bool
    episode_count: int
    capability_count: int
    capabilities: dict[str, dict[str, int]]
    snapshot: AgentSelfAwarenessSnapshot | None

    @property
    def has_evidence(self) -> bool:
        return self.capability_count > 0

    @property
    def snapshot_available(self) -> bool:
        return self.snapshot is not None


class ExperienceSelfAwarenessUpdater:
    """
    Convert episodic experience history into capability
    evidence and update AgentSelfAwareness.

    Capability identity is currently derived from the
    normalized episode goal/skill.

    Example:

        "Open the door"
            ↓
        "open_the_door"

    This intentionally aligns with Agent._knowledge_uncertainty()
    and procedural skill naming.

    The updater rebuilds evidence from the available episode
    history rather than incrementing an opaque counter. This
    prevents duplicate counting when self-awareness is
    refreshed repeatedly.
    """

    def __init__(
        self,
        self_awareness: AgentSelfAwareness,
    ):
        if self_awareness is None:
            raise ValueError(
                "self_awareness cannot be None."
            )

        self.self_awareness = self_awareness

    # =====================================================
    # Capability normalization
    # =====================================================

    @staticmethod
    def normalize_capability(
        value: Any,
    ) -> str | None:
        """
        Convert a goal/skill/action identifier into a stable
        capability key.
        """

        if value is None:
            return None

        text = str(value).strip().lower()

        if not text:
            return None

        normalized = (
            text
            .replace("-", "_")
            .replace(" ", "_")
        )

        while "__" in normalized:
            normalized = normalized.replace(
                "__",
                "_",
            )

        return normalized

    # =====================================================
    # Episode extraction
    # =====================================================

    @staticmethod
    def _episode_capability(
        episode: Any,
    ) -> str | None:
        """
        Resolve the capability represented by an episode.

        Goal/skill has priority because it maps naturally to
        the Agent's procedural skill representation.

        Action is used as fallback.
        """

        goal = getattr(
            episode,
            "goal",
            None,
        )

        capability = (
            ExperienceSelfAwarenessUpdater
            .normalize_capability(goal)
        )

        if capability is not None:
            return capability

        action = getattr(
            episode,
            "action",
            None,
        )

        return (
            ExperienceSelfAwarenessUpdater
            .normalize_capability(action)
        )

    @staticmethod
    def _episode_success(
        episode: Any,
    ) -> bool | None:
        """
        Extract success information from an episode.

        Returns:
            True  -> successful episode
            False -> failed episode
            None  -> outcome unavailable
        """

        episode_result = getattr(
            episode,
            "result",
            None,
        )

        if isinstance(
            episode_result,
            dict,
        ):
            if "success" in episode_result:
                return bool(
                    episode_result["success"]
                )

        # Some episode implementations may expose success
        # directly.
        direct_success = getattr(
            episode,
            "success",
            None,
        )

        if direct_success is not None:
            return bool(
                direct_success
            )

        # Some implementations may expose failed instead.
        direct_failed = getattr(
            episode,
            "failed",
            None,
        )

        if direct_failed is not None:
            return not bool(
                direct_failed
            )

        return None

    # =====================================================
    # Evidence building
    # =====================================================

    def build_capability_evidence(
        self,
        episodes: Iterable[Any],
    ) -> dict[str, dict[str, int]]:
        """
        Aggregate episode history into capability evidence.

        Each episode contributes at most one attempt.

        Episodes without a resolvable capability or outcome
        are ignored because assigning success/failure in those
        cases would create false evidence.
        """

        capabilities: dict[
            str,
            dict[str, int],
        ] = {}

        for episode in episodes:

            capability = (
                self._episode_capability(
                    episode
                )
            )

            if capability is None:
                continue

            success = (
                self._episode_success(
                    episode
                )
            )

            if success is None:
                continue

            evidence = capabilities.setdefault(
                capability,
                {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                },
            )

            evidence["attempts"] += 1

            if success:
                evidence["successes"] += 1
            else:
                evidence["failures"] += 1

        return capabilities

    # =====================================================
    # Self-awareness update
    # =====================================================

    def update(
        self,
        episodes: Iterable[Any],
        *,
        persist: bool = True,
    ) -> ExperienceSelfAwarenessUpdateResult:
        """
        Rebuild self-awareness from episode history.

        The evidence is cumulative over the supplied episode
        collection.
        """

        episode_list = list(
            episodes
        )

        capabilities = (
            self.build_capability_evidence(
                episode_list
            )
        )

        if not capabilities:
            return (
                ExperienceSelfAwarenessUpdateResult(
                    updated=False,
                    episode_count=len(
                        episode_list
                    ),
                    capability_count=0,
                    capabilities={},
                    snapshot=None,
                )
            )

        snapshot = (
            self.self_awareness.evaluate(
                capabilities,
                persist=persist,
            )
        )

        return (
            ExperienceSelfAwarenessUpdateResult(
                updated=True,
                episode_count=len(
                    episode_list
                ),
                capability_count=len(
                    capabilities
                ),
                capabilities=capabilities,
                snapshot=snapshot,
            )
        )
