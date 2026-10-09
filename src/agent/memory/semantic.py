from dataclasses import dataclass, field
from typing import Any, Dict, List

from .models import Memory


@dataclass
class SemanticMemory(Memory):
    """
    Represents knowledge extracted from one or more experiences.
    """

    subject: str = ""
    predicate: str = ""
    object: Any = None

    confidence: float = 0.5

    evidence_count: int = 1
    contradiction_count: int = 0

    source_episodes: List[str] = field(
        default_factory=list
    )

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    def reinforce(
        self,
        episode_id: str,
        amount: float = 0.1,
    ) -> None:

        self.evidence_count += 1

        self.confidence = round(
            min(
                1.0,
                self.confidence + amount,
            ),
            6,
        )

        if episode_id not in self.source_episodes:
            self.source_episodes.append(
                episode_id
            )

    def contradict(
        self,
        amount: float = 0.1,
    ) -> None:

        self.contradiction_count += 1

        self.confidence = round(
            max(
                0.0,
                self.confidence - amount,
            ),
            6,
        )

    def register_contradiction(
        self,
        episode_id: str,
        amount: float = 0.1,
    ) -> None:

        self.contradiction_count += 1

        if episode_id not in self.source_episodes:
            self.source_episodes.append(
                episode_id
            )

        self.confidence = round(
            max(
                0.0,
                self.confidence - amount,
            ),
            6,
        )

    def matches_context(
        self,
        context: Dict[str, Any],
    ) -> bool:

        for key, expected_value in self.context.items():

            if context.get(key) != expected_value:
                return False

        return True
