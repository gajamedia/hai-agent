from typing import Any, Dict, List, Optional, Tuple

from agent.core.action import Action
from agent.cognition.exploration_policy import (
    ExplorationCandidate,
    ExplorationPolicy,
)


class ActionExplorationPolicy:
    """
    Selects executable actions as exploration experiments.

    A candidate action is considered only when its
    preconditions are satisfied in the current world state.
    """

    def __init__(
        self,
        policy: Optional[ExplorationPolicy] = None,
    ):
        self.policy = (
            policy
            if policy is not None
            else ExplorationPolicy()
        )

    def candidates(
        self,
        actions: List[Action],
        world_state: Dict[str, Any],
        information_gain: Dict[str, float],
        risk: Dict[str, float],
        action_cost: Optional[Dict[str, float]] = None,
    ) -> List[ExplorationCandidate]:

        if action_cost is None:
            action_cost = {}

        result = []

        for action in actions:

            if not action.can_execute(
                world_state
            ):
                continue

            result.append(
                ExplorationCandidate(
                    action=action.name,
                    expected_information_gain=(
                        information_gain.get(
                            action.name,
                            0.0,
                        )
                    ),
                    risk=risk.get(
                        action.name,
                        0.0,
                    ),
                    action_cost=action_cost.get(
                        action.name,
                        0.0,
                    ),
                )
            )

        return result

    def rank(
        self,
        actions: List[Action],
        world_state: Dict[str, Any],
        information_gain: Dict[str, float],
        risk: Dict[str, float],
        action_cost: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Action, float]]:

        candidates = self.candidates(
            actions=actions,
            world_state=world_state,
            information_gain=information_gain,
            risk=risk,
            action_cost=action_cost,
        )

        ranked_candidates = self.policy.rank(
            candidates
        )

        action_by_name = {
            action.name: action
            for action in actions
        }

        return [
            (
                action_by_name[candidate.action],
                self.policy.score(candidate),
            )
            for candidate in ranked_candidates
        ]

    def select(
        self,
        actions: List[Action],
        world_state: Dict[str, Any],
        information_gain: Dict[str, float],
        risk: Dict[str, float],
        action_cost: Optional[Dict[str, float]] = None,
    ) -> Optional[Action]:

        ranked = self.rank(
            actions=actions,
            world_state=world_state,
            information_gain=information_gain,
            risk=risk,
            action_cost=action_cost,
        )

        if not ranked:
            return None

        return ranked[0][0]
