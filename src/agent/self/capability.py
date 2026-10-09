from dataclasses import dataclass, field
from typing import List


@dataclass
class Capability:
    """
    Represents an action or skill the agent believes
    it is capable of performing.
    """

    name: str
    confidence: float = 0.5

    success_count: int = 0
    failure_count: int = 0

    evidence: List[str] = field(
        default_factory=list
    )

    def record_success(
        self,
        source_id: str,
        amount: float = 0.1,
    ) -> None:

        self.success_count += 1

        if source_id not in self.evidence:
            self.evidence.append(
                source_id
            )

        self.confidence = round(
            min(
                1.0,
                self.confidence + amount,
            ),
            6,
        )

    def record_failure(
        self,
        source_id: str,
        amount: float = 0.1,
    ) -> None:

        self.failure_count += 1

        if source_id not in self.evidence:
            self.evidence.append(
                source_id
            )

        self.confidence = round(
            max(
                0.0,
                self.confidence - amount,
            ),
            6,
        )

    @property
    def attempts(self) -> int:
        return (
            self.success_count
            + self.failure_count
        )

    @property
    def success_rate(self) -> float:

        if self.attempts == 0:
            return 0.0

        return round(
            self.success_count
            / self.attempts,
            6,
        )
