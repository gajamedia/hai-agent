from dataclasses import dataclass


@dataclass
class Uncertainty:
    """
    Represents uncertainty associated with a belief,
    capability, or decision.
    """

    value: float

    @property
    def certainty(self) -> float:
        return round(
            1.0 - self.value,
            6,
        )

    def increase(
        self,
        amount: float = 0.1,
    ) -> None:

        self.value = round(
            min(
                1.0,
                self.value + amount,
            ),
            6,
        )

    def decrease(
        self,
        amount: float = 0.1,
    ) -> None:

        self.value = round(
            max(
                0.0,
                self.value - amount,
            ),
            6,
        )
