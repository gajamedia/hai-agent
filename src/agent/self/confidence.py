from dataclasses import dataclass


@dataclass
class ConfidenceScore:
    """
    Represents confidence and uncertainty
    for one belief or capability.
    """

    confidence: float
    evidence_count: int = 0

    @property
    def uncertainty(self) -> float:
        return round(
            1.0 - self.confidence,
            6,
        )

    def reinforce(
        self,
        amount: float = 0.1,
    ) -> None:

        self.confidence = round(
            min(
                1.0,
                self.confidence + amount,
            ),
            6,
        )

        self.evidence_count += 1

    def weaken(
        self,
        amount: float = 0.1,
    ) -> None:

        self.confidence = round(
            max(
                0.0,
                self.confidence - amount,
            ),
            6,
        )

        self.evidence_count += 1
