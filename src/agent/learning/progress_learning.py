from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ProgressLearningResult:
    """
    Learning signal extracted from an experience.

    progress_before:
        Goal progress before the experience.

    progress_after:
        Goal progress after the experience.

    progress_delta:
        Change in goal progress.

    completed:
        Whether the goal was satisfied after the experience.
    """

    goal: str
    progress_before: float
    progress_after: float
    progress_delta: float
    completed: bool


class ProgressAwareLearning:
    """
    Extracts goal-progress signals from experiences.

    The current stage records the signal without changing
    strategy selection yet.
    """

    @staticmethod
    def _progress(
        target_state: dict[str, Any],
        world_state: dict[str, Any],
    ) -> float:

        if not target_state:
            return 0.0

        satisfied = 0

        for key, expected_value in (
            target_state.items()
        ):
            if (
                world_state.get(key)
                == expected_value
            ):
                satisfied += 1

        return round(
            satisfied / len(target_state),
            6,
        )

    def learn_from_experience(
        self,
        goal: Any,
        episodes: list[Any],
    ) -> Optional[ProgressLearningResult]:
        """
        Extract one progress signal from an experience.

        The progress delta is calculated from the world state
        before the first episode to the world state after the
        last episode.
        """

        if goal is None:
            return None

        if not episodes:
            return None

        target_state = getattr(
            goal,
            "target_state",
            {},
        )

        if not target_state:
            return ProgressLearningResult(
                goal=goal.description,
                progress_before=0.0,
                progress_after=0.0,
                progress_delta=0.0,
                completed=False,
            )

        first_episode = episodes[0]
        last_episode = episodes[-1]

        observation_before = (
            getattr(
                first_episode,
                "observation_before",
                {},
            )
            or {}
        )

        observation_after = (
            getattr(
                last_episode,
                "observation_after",
                {},
            )
            or {}
        )

        progress_before = self._progress(
            target_state=target_state,
            world_state=observation_before,
        )

        progress_after = self._progress(
            target_state=target_state,
            world_state=observation_after,
        )

        progress_delta = round(
            progress_after
            - progress_before,
            6,
        )

        completed = (
            progress_after
            >= 1.0
        )

        return ProgressLearningResult(
            goal=goal.description,
            progress_before=progress_before,
            progress_after=progress_after,
            progress_delta=progress_delta,
            completed=completed,
        )
