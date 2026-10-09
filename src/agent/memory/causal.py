from dataclasses import dataclass, field
from typing import Any, List

from .models import Memory


@dataclass
class CausalMemory(Memory):
    """
    Represents a causal relationship learned from experience.

    Example:
        open_box -> enables -> take_key
    """

    cause: str = ""
    relation: str = ""
    effect: str = ""

    condition: Any = None

    evidence_count: int = 0
    contradiction_count: int = 0

    source_episodes: List[str] = field(default_factory=list)

    def reinforce(
        self,
        episode_ids: List[str],
        amount: float = 0.1,
    ) -> None:

        new_evidence = 0

        for episode_id in episode_ids:
            if episode_id not in self.source_episodes:
                self.source_episodes.append(
                    episode_id
                )
                self.evidence_count += 1
                new_evidence += 1

        if new_evidence == 0:
            return

        self.confidence = round(
            min(
                1.0,
                self.confidence + amount,
            ),
            6,
        )

    def contradict(
        self,
        amount: float = 0.1,
    ) -> None:

        self.contradiction_count += 1

        self.confidence = round(
            max(0.0, self.confidence - amount),
            6,
        )
