from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.world.environment import Environment
from agent.cognition.active_explorer import (
    ActiveExplorer,
    ActiveExplorationResult,
)


@dataclass(frozen=True)
class ExplorationLoopResult:
    """
    Result of a multi-step active exploration loop.
    """

    results: List[ActiveExplorationResult] = field(
        default_factory=list
    )

    @property
    def steps(self) -> int:
        return len(self.results)

    @property
    def successful_steps(self) -> int:
        return sum(
            1
            for result in self.results
            if (
                result.execution is not None
                and result.execution.result.get(
                    "success",
                    False,
                )
            )
        )

    @property
    def failed_steps(self) -> int:
        return sum(
            1
            for result in self.results
            if (
                result.execution is not None
                and not result.execution.result.get(
                    "success",
                    False,
                )
            )
        )


class ActiveExplorationLoop:
    """
    Executes repeated exploration experiments.

    The loop stops when:
        - exploration is no longer recommended
        - no executable candidate exists
        - max_steps is reached
    """

    def __init__(
        self,
        explorer: Optional[ActiveExplorer] = None,
        max_steps: int = 10,
    ):
        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        self.explorer = (
            explorer
            if explorer is not None
            else ActiveExplorer()
        )

        self.max_steps = max_steps

    def run(
        self,
        agent: Agent,
        environment: Environment,
        actions: List[Action],
        information_gain: Dict[str, float],
        risk: Dict[str, float],
        goal: Goal,
        exploration_risk: float = 0.0,
        action_cost: Optional[
            Dict[str, float]
        ] = None,
    ) -> ExplorationLoopResult:

        if agent is None:
            raise ValueError(
                "agent cannot be None."
            )

        if environment is None:
            raise ValueError(
                "environment cannot be None."
            )

        if goal is None:
            raise ValueError(
                "goal cannot be None."
            )

        if not actions:
            raise ValueError(
                "actions cannot be empty."
            )

        results: List[
            ActiveExplorationResult
        ] = []

        tried_actions = set()

        for step in range(
            self.max_steps
        ):

            current_actions = [
                action
                for action in actions
                if action.name not in tried_actions
            ]

            if not current_actions:
                break

            result = (
                self.explorer.explore_once(
                    agent=agent,
                    environment=environment,
                    actions=current_actions,
                    information_gain=(
                        information_gain
                    ),
                    risk=risk,
                    goal=goal,
                    experience_id=(
                        f"exploration-loop-{step + 1}"
                    ),
                    exploration_risk=(
                        exploration_risk
                    ),
                    action_cost=action_cost,
                )
            )

            if result.action is None:
                break

            results.append(
                result
            )

            tried_actions.add(
                result.action.name
            )

        return ExplorationLoopResult(
            results=results
        )
