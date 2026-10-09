from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.agent_decision import (
    AgentDecision,
    DecisionMode,
)
from agent.cognition.active_explorer import (
    ActiveExplorer,
    ActiveExplorationResult,
)
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.learning.strategy import Strategy
from agent.world.environment import Environment


@dataclass(frozen=True)
class UnifiedExecutionResult:
    """
    Result of executing one top-level agent decision.
    """

    decision: AgentDecision

    exploration_result: Optional[
        ActiveExplorationResult
    ] = None

    plan: Optional[
        List[Action]
    ] = None

    @property
    def mode(self) -> DecisionMode:
        return self.decision.mode

    @property
    def action(self) -> Optional[Action]:
        if self.exploration_result is not None:
            return self.exploration_result.action

        if self.plan:
            return self.plan[0]

        return None


class UnifiedDecisionExecutor:
    """
    Dispatches the agent's top-level decision.

    EXPLORE:
        -> ActiveExplorer

    EXPLOIT:
        -> RiskAwareHierarchicalPlanner
    """

    def __init__(
        self,
        explorer: Optional[
            ActiveExplorer
        ] = None,
        hierarchical_planner: Optional[
            RiskAwareHierarchicalPlanner
        ] = None,
    ):
        self.explorer = explorer

        self.hierarchical_planner = (
            hierarchical_planner
        )

    def execute(
        self,
        decision: AgentDecision,
        *,
        # Exploration inputs
        agent=None,
        environment: Optional[
            Environment
        ] = None,
        actions: Optional[
            List[Action]
        ] = None,
        information_gain: Optional[
            Dict[str, float]
        ] = None,
        action_risk: Optional[
            Dict[str, float]
        ] = None,
        exploration_risk: float = 0.0,
        experience_id: Optional[
            str
        ] = None,

        # Planning inputs
        goal: Optional[
            Goal
        ] = None,
        world_state: Optional[
            Dict[str, Any]
        ] = None,
        strategies: Optional[
            List[Strategy]
        ] = None,
        action_cost: Optional[
            Dict[str, float]
        ] = None,
    ) -> UnifiedExecutionResult:

        if decision is None:
            raise ValueError(
                "decision cannot be None."
            )

        if decision.mode == DecisionMode.EXPLORE:

            return self._execute_exploration(
                decision=decision,
                agent=agent,
                environment=environment,
                actions=actions,
                information_gain=(
                    information_gain
                    if information_gain is not None
                    else {}
                ),
                action_risk=(
                    action_risk
                    if action_risk is not None
                    else {}
                ),
                exploration_risk=(
                    exploration_risk
                ),
                experience_id=(
                    experience_id
                    if experience_id is not None
                    else "exploration"
                ),
                goal=goal,
                action_cost=action_cost,
            )

        if decision.mode == DecisionMode.EXPLOIT:

            return self._execute_exploitation(
                decision=decision,
                goal=goal,
                world_state=world_state,
                strategies=strategies,
            )

        raise ValueError(
            f"Unsupported decision mode: "
            f"{decision.mode}"
        )

    def _execute_exploration(
        self,
        decision: AgentDecision,
        agent,
        environment: Optional[
            Environment
        ],
        actions: Optional[
            List[Action]
        ],
        information_gain: Dict[str, float],
        action_risk: Dict[str, float],
        exploration_risk: float,
        experience_id: str,
        goal: Optional[Goal],
        action_cost: Optional[
            Dict[str, float]
        ],
    ) -> UnifiedExecutionResult:

        if self.explorer is None:
            raise RuntimeError(
                "ActiveExplorer is not configured."
            )

        if agent is None:
            raise ValueError(
                "agent is required for exploration."
            )

        if environment is None:
            raise ValueError(
                "environment is required for exploration."
            )

        if not actions:
            raise ValueError(
                "actions are required for exploration."
            )

        if goal is None:
            raise ValueError(
                "goal is required for exploration."
            )

        result = self.explorer.explore_once(
            agent=agent,
            environment=environment,
            actions=actions,
            information_gain=(
                information_gain
            ),
            risk=action_risk,
            goal=goal,
            experience_id=experience_id,
            exploration_risk=(
                exploration_risk
            ),
            action_cost=action_cost,
        )

        return UnifiedExecutionResult(
            decision=decision,
            exploration_result=result,
        )

    def _execute_exploitation(
        self,
        decision: AgentDecision,
        goal: Optional[Goal],
        world_state: Optional[
            Dict[str, Any]
        ],
        strategies: Optional[
            List[Strategy]
        ],
    ) -> UnifiedExecutionResult:

        if self.hierarchical_planner is None:
            raise RuntimeError(
                "RiskAwareHierarchicalPlanner "
                "is not configured."
            )

        if goal is None:
            raise ValueError(
                "goal is required for exploitation."
            )

        if world_state is None:
            raise ValueError(
                "world_state is required for exploitation."
            )

        plan = self.hierarchical_planner.plan(
            goal=goal,
            world_state=world_state,
            strategies=(
                strategies
                if strategies is not None
                else []
            ),
        )

        return UnifiedExecutionResult(
            decision=decision,
            plan=plan,
        )
