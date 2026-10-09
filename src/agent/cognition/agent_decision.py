from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DecisionMode(str, Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"


@dataclass(frozen=True)
class AgentDecision:
    """
    Represents the top-level decision of the agent.
    """

    mode: DecisionMode
    reason: str
    uncertainty: float
    expected_information_gain: float
    risk: float

    @property
    def should_explore(self) -> bool:
        return self.mode == DecisionMode.EXPLORE

    @property
    def should_exploit(self) -> bool:
        return self.mode == DecisionMode.EXPLOIT


class AgentDecisionMaker:
    """
    Decides whether the agent should explore or exploit.

    The decision is derived from memory uncertainty and risk.
    """

    def __init__(
        self,
        exploration_threshold: float = 0.6,
        maximum_exploration_risk: float = 0.7,
    ):
        self.exploration_threshold = (
            exploration_threshold
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
    ) -> AgentDecision:

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
            >= self.exploration_threshold
            and expected_information_gain
            >= self.exploration_threshold
            and risk
            <= self.maximum_exploration_risk
        ):
            return AgentDecision(
                mode=DecisionMode.EXPLORE,
                reason=(
                    "Knowledge is uncertain and "
                    "additional information is valuable."
                ),
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
            return AgentDecision(
                mode=DecisionMode.EXPLOIT,
                reason=(
                    "Exploration risk is too high."
                ),
                uncertainty=uncertainty,
                expected_information_gain=(
                    expected_information_gain
                ),
                risk=risk,
            )

        return AgentDecision(
            mode=DecisionMode.EXPLOIT,
            reason=(
                "Existing knowledge is sufficient "
                "for exploitation."
            ),
            uncertainty=uncertainty,
            expected_information_gain=(
                expected_information_gain
            ),
            risk=risk,
        )
