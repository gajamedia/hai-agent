from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorationDecision:
    """
    Represents whether the agent should explore or exploit.

    All normalized scores are in the range [0.0, 1.0].
    """

    explore: bool
    reason: str
    uncertainty: float
    expected_information_gain: float
    risk: float


class ExplorationDecisionMaker:
    """
    Decides whether additional exploration is justified.

    Exploration is favored when:
        - uncertainty is high
        - expected information gain is high
        - risk is acceptable

    Exploration is avoided when:
        - uncertainty is low
        - information gain is low
        - risk is too high
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.6,
        information_gain_threshold: float = 0.5,
        maximum_exploration_risk: float = 0.7,
    ):
        self.uncertainty_threshold = (
            uncertainty_threshold
        )

        self.information_gain_threshold = (
            information_gain_threshold
        )

        self.maximum_exploration_risk = (
            maximum_exploration_risk
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(1.0, value),
        )

    def decide(
        self,
        uncertainty: float,
        expected_information_gain: float,
        risk: float,
    ) -> ExplorationDecision:

        uncertainty = self._clamp(
            uncertainty
        )

        expected_information_gain = (
            self._clamp(
                expected_information_gain
            )
        )

        risk = self._clamp(
            risk
        )

        if (
            uncertainty
            >= self.uncertainty_threshold
            and expected_information_gain
            >= self.information_gain_threshold
            and risk
            <= self.maximum_exploration_risk
        ):
            return ExplorationDecision(
                explore=True,
                reason="High uncertainty with useful information gain.",
                uncertainty=uncertainty,
                expected_information_gain=(
                    expected_information_gain
                ),
                risk=risk,
            )

        if (
            risk
            > self.maximum_exploration_risk
        ):
            return ExplorationDecision(
                explore=False,
                reason="Exploration risk is too high.",
                uncertainty=uncertainty,
                expected_information_gain=(
                    expected_information_gain
                ),
                risk=risk,
            )

        if (
            uncertainty
            < self.uncertainty_threshold
        ):
            return ExplorationDecision(
                explore=False,
                reason="Uncertainty is sufficiently low.",
                uncertainty=uncertainty,
                expected_information_gain=(
                    expected_information_gain
                ),
                risk=risk,
            )

        if (
            expected_information_gain
            < self.information_gain_threshold
        ):
            return ExplorationDecision(
                explore=False,
                reason="Expected information gain is too low.",
                uncertainty=uncertainty,
                expected_information_gain=(
                    expected_information_gain
                ),
                risk=risk,
            )

        return ExplorationDecision(
            explore=False,
            reason="Exploration criteria were not satisfied.",
            uncertainty=uncertainty,
            expected_information_gain=(
                expected_information_gain
            ),
            risk=risk,
        )
