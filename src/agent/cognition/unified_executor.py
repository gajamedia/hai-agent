
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
    """Result of executing one top-level agent decision."""

    decision: AgentDecision
    exploration_result: Optional[ActiveExplorationResult] = None
    plan: Optional[List[Action]] = None

    # Populated only by the exact single-action execution path.
    direct_action: Optional[Action] = None
    direct_result: Any = None

    @property
    def mode(self) -> DecisionMode:
        return self.decision.mode

    @property
    def action(self) -> Optional[Action]:
        if self.direct_action is not None:
            return self.direct_action

        if self.exploration_result is not None:
            return self.exploration_result.action

        if self.plan:
            return self.plan[0]

        return None

    @property
    def success(self) -> bool:
        """Return execution success, not merely plan availability."""
        if self.direct_result is not None:
            if isinstance(self.direct_result, dict):
                return bool(
                    self.direct_result.get("success", False)
                )

            value = getattr(self.direct_result, "success", None)
            if value is not None:
                return bool(value)

            return False

        if self.exploration_result is not None:
            return bool(self.exploration_result.success)

        # A plan is not evidence that an action was executed.
        return False


class UnifiedDecisionExecutor:
    """
    Dispatch the agent's normal top-level decision.

    EXPLORE:
        -> ActiveExplorer

    EXPLOIT:
        -> RiskAwareHierarchicalPlanner

    execute_action():
        -> Execute one explicitly requested registered Action.

    The exact-action path is separate so existing execute() semantics
    remain unchanged.
    """

    def __init__(
        self,
        explorer: Optional[ActiveExplorer] = None,
        hierarchical_planner: Optional[
            RiskAwareHierarchicalPlanner
        ] = None,
    ):
        self.explorer = explorer
        self.hierarchical_planner = hierarchical_planner

    def execute(
        self,
        decision: AgentDecision,
        *,
        agent=None,
        environment: Optional[Environment] = None,
        actions: Optional[List[Action]] = None,
        information_gain: Optional[Dict[str, float]] = None,
        action_risk: Optional[Dict[str, float]] = None,
        exploration_risk: float = 0.0,
        experience_id: Optional[str] = None,
        goal: Optional[Goal] = None,
        world_state: Optional[Dict[str, Any]] = None,
        strategies: Optional[List[Strategy]] = None,
        action_cost: Optional[Dict[str, float]] = None,
    ) -> UnifiedExecutionResult:

        if decision is None:
            raise ValueError("decision cannot be None.")

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
                exploration_risk=exploration_risk,
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
            f"Unsupported decision mode: {decision.mode}"
        )

    def execute_action(
        self,
        *,
        decision: AgentDecision,
        action: Action,
        agent,
        environment: Environment,
        goal: Optional[Goal] = None,
        world_state: Optional[Dict[str, Any]] = None,
        experience_id: Optional[str] = None,
        **_: Any,
    ) -> UnifiedExecutionResult:
        """
        Execute exactly the supplied Action.

        This method does not ask ActiveExplorer to choose another action
        and does not ask the planner to replace the requested action.
        """

        if decision is None:
            raise ValueError("decision cannot be None.")

        if action is None:
            raise ValueError("action cannot be None.")

        if agent is None:
            raise ValueError("agent is required.")

        if environment is None:
            raise ValueError("environment is required.")

        if self.explorer is None:
            raise RuntimeError(
                "ActiveExplorer must be configured to execute "
                "a direct action."
            )

        action_name = str(
            getattr(action, "name", "")
        ).strip()

        if not action_name:
            raise ValueError(
                "action must expose a non-empty name."
            )

        # Always use the current environment state for preconditions.
        observation_before = environment.observe()

        can_execute = getattr(action, "can_execute", None)
        if callable(can_execute) and not can_execute(
            observation_before
        ):
            failure = {
                "success": False,
                "action": action_name,
                "message": (
                    "Preconditions are not satisfied for action "
                    f"'{action_name}'."
                ),
                "observation_before": observation_before,
                "observation_after": observation_before,
            }

            return UnifiedExecutionResult(
                decision=decision,
                direct_action=action,
                direct_result=failure,
            )

        # Only a Goal instance is accepted by Agent.set_goal().
        if isinstance(goal, Goal):
            agent.set_goal(goal)

        current_experience = getattr(
            agent,
            "current_experience",
            None,
        )
        owns_experience = current_experience is None

        if owns_experience:
            goal_description = (
                getattr(goal, "description", None)
                or str(goal or f"Execute action: {action_name}")
            )
            resolved_experience_id = (
                experience_id
                or (
                    f"{getattr(agent, 'name', 'agent')}"
                    f"-single-action-{agent.state.cycle + 1}"
                )
            )

            agent.start_experience(
                experience_id=resolved_experience_id,
                goal=goal_description,
            )

        try:
            # ExplorationExecutor performs agent.act(), environment
            # execution, observations, and episode recording.
            execution = self.explorer.executor.execute(
                agent=agent,
                environment=environment,
                action=action,
            )

            if owns_experience:
                if execution.success:
                    agent.complete_experience()
                else:
                    agent.fail_experience()

            return UnifiedExecutionResult(
                decision=decision,
                direct_action=action,
                direct_result=execution,
            )

        except Exception:
            if owns_experience:
                try:
                    agent.fail_experience()
                except (RuntimeError, AttributeError):
                    pass
            raise

        finally:
            # Never close an experience owned by the caller.
            if owns_experience:
                agent.close_experience()

    def _execute_exploration(
        self,
        decision: AgentDecision,
        agent,
        environment: Optional[Environment],
        actions: Optional[List[Action]],
        information_gain: Dict[str, float],
        action_risk: Dict[str, float],
        exploration_risk: float,
        experience_id: str,
        goal: Optional[Goal],
        action_cost: Optional[Dict[str, float]],
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
            information_gain=information_gain,
            risk=action_risk,
            goal=goal,
            experience_id=experience_id,
            exploration_risk=exploration_risk,
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
        world_state: Optional[Dict[str, Any]],
        strategies: Optional[List[Strategy]],
    ) -> UnifiedExecutionResult:

        if self.hierarchical_planner is None:
            raise RuntimeError(
                "RiskAwareHierarchicalPlanner is not configured."
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
                strategies if strategies is not None else []
            ),
        )

        return UnifiedExecutionResult(
            decision=decision,
            plan=plan,
        )
