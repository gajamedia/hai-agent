from dataclasses import dataclass, field
from typing import List

from .models import Memory


@dataclass
class ProceduralMemory(Memory):
    """
    Represents a learned procedure or skill.

    A procedure is a reusable sequence of actions
    that has been observed to achieve a goal.
    """

    skill: str = ""

    steps: List[str] = field(
        default_factory=list
    )

    success_count: int = 0
    failure_count: int = 0

    source_episodes: List[str] = field(
        default_factory=list
    )

    def record_success(
        self,
        episode_ids: List[str],
    ) -> None:

        self.success_count += 1

        for episode_id in episode_ids:
            if episode_id not in self.source_episodes:
                self.source_episodes.append(
                    episode_id
                )

        self.confidence = round(
            min(
                1.0,
                self.confidence + 0.1,
            ),
            6,
        )

    def record_failure(
        self,
        episode_ids: List[str],
    ) -> None:

        self.failure_count += 1

        for episode_id in episode_ids:
            if episode_id not in self.source_episodes:
                self.source_episodes.append(
                    episode_id
                )

        self.confidence = round(
            max(
                0.0,
                self.confidence - 0.1,
            ),
            6,
        )

    @property
    def success_rate(self) -> float:

        total = (
            self.success_count
            + self.failure_count
        )

        if total == 0:
            return 0.0

        return round(
            self.success_count / total,
            6,
        )
