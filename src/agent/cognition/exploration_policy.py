from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.core.action import Action


@dataclass(frozen=True)
class ExplorationCandidate:
    """
    Candidate action for exploration.
    """

    action: Action | str
    expected_information_gain: float
    risk: float
    action_cost: float = 0.0

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    @property
    def score(self) -> float:
        """
        Raw candidate score.

        Information gain and risk are normalized to [0, 1].
        Action cost remains proportional so a cost of 5 is
        penalized more heavily than a cost of 1.
        """

        information_gain = self._clamp(
            self.expected_information_gain
        )

        risk = self._clamp(
            self.risk
        )

        return round(
            information_gain
            - risk
            - self.action_cost,
            6,
        )


class ExplorationPolicy:
    """
    Policy for ranking exploration candidates.
    """

    def __init__(
        self,
        information_gain_weight: float = 1.0,
        risk_weight: float = 1.0,
        action_cost_weight: float = 1.0,
    ):
        self.information_gain_weight = (
            information_gain_weight
        )

        self.risk_weight = risk_weight

        self.action_cost_weight = (
            action_cost_weight
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    def score(
        self,
        candidate: ExplorationCandidate,
    ) -> float:

        if candidate is None:
            raise ValueError(
                "candidate cannot be None"
            )

        information_gain = self._clamp(
            candidate.expected_information_gain
        )

        risk = self._clamp(
            candidate.risk
        )

        # Action cost is intentionally NOT clamped.
        # A cost of 5 must be more expensive than a cost of 1.
        action_cost = candidate.action_cost

        score = (
            (
                information_gain
                * self.information_gain_weight
            )
            - (
                risk
                * self.risk_weight
            )
            - (
                action_cost
                * self.action_cost_weight
            )
        )

        return round(
            score,
            6,
        )

    def rank(
        self,
        candidates: list[
            ExplorationCandidate
        ],
    ) -> list[
        ExplorationCandidate
    ]:

        return sorted(
            candidates,
            key=self.score,
            reverse=True,
        )

    def select(
        self,
        candidates: list[
            ExplorationCandidate
        ],
    ) -> Optional[
        ExplorationCandidate
    ]:

        ranked = self.rank(
            candidates
        )

        if not ranked:
            return None

        return ranked[0]


class ActionExplorationPolicy:
    """
    Builds exploration candidates from executable actions.
    """

    def __init__(
        self,
        policy: Optional[
            ExplorationPolicy
        ] = None,
    ):
        self.policy = (
            policy
            if policy is not None
            else ExplorationPolicy()
        )

    def candidates(
        self,
        actions: list[Action],
        world_state: dict,
        information_gain: Optional[
            dict[str, float]
        ] = None,
        action_risk: Optional[
            dict[str, float]
        ] = None,
        action_cost: Optional[
            dict[str, float]
        ] = None,
    ) -> list[
        ExplorationCandidate
    ]:

        information_gain = (
            information_gain
            if information_gain is not None
            else {}
        )

        action_risk = (
            action_risk
            if action_risk is not None
            else {}
        )

        action_cost = (
            action_cost
            if action_cost is not None
            else {}
        )

        candidates = []

        for action in actions:

            if not action.can_execute(
                world_state
            ):
                continue

            candidates.append(
                ExplorationCandidate(
                    action=action,
                    expected_information_gain=(
                        information_gain.get(
                            action.name,
                            0.0,
                        )
                    ),
                    risk=action_risk.get(
                        action.name,
                        0.0,
                    ),
                    action_cost=action_cost.get(
                        action.name,
                        0.0,
                    ),
                )
            )

        return candidates

    def rank(
        self,
        actions: list[Action],
        world_state: dict,
        information_gain: Optional[
            dict[str, float]
        ] = None,
        action_risk: Optional[
            dict[str, float]
        ] = None,
        action_cost: Optional[
            dict[str, float]
        ] = None,
    ) -> list[
        ExplorationCandidate
    ]:

        candidates = self.candidates(
            actions=actions,
            world_state=world_state,
            information_gain=information_gain,
            action_risk=action_risk,
            action_cost=action_cost,
        )

        return self.policy.rank(
            candidates
        )

    def select(
        self,
        actions: list[Action],
        world_state: dict,
        information_gain: Optional[
            dict[str, float]
        ] = None,
        action_risk: Optional[
            dict[str, float]
        ] = None,
        action_cost: Optional[
            dict[str, float]
        ] = None,
    ) -> Optional[
        ExplorationCandidate
    ]:

        candidates = self.candidates(
            actions=actions,
            world_state=world_state,
            information_gain=information_gain,
            action_risk=action_risk,
            action_cost=action_cost,
        )

        return self.policy.select(
            candidates
        )
