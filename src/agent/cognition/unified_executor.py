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

    # Exact single-action execution.
    direct_action: Optional[Action] = None
    direct_result: Any = None

    # Multi-step plan execution.
    plan_execution_results: Optional[tuple[Any, ...]] = None
    goal_satisfied: Optional[bool] = None

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

        if self.plan_execution_results is not None:
            return (
                len(self.plan_execution_results) > 0
                and all(
                    bool(getattr(step, "success", False))
                    for step in self.plan_execution_results
                )
                and self.goal_satisfied is True
            )

        if self.direct_result is not None:
            if isinstance(self.direct_result, dict):
                return bool(
                    self.direct_result.get("success", False)
                )

            value = getattr(
                self.direct_result,
                "success",
                None,
            )

            if value is not None:
                return bool(value)

            return False

        if self.exploration_result is not None:
            return bool(self.exploration_result.success)

        # A plan alone is not evidence of execution.
        return False


@dataclass(frozen=True)
class _PlanStepResult:
    action: str
    success: bool
    result: Any = None


class UnifiedDecisionExecutor:
    """
    Dispatch the agent's normal top-level decision.

    EXPLORE:
        -> ActiveExplorer

    EXPLOIT:
        -> RiskAwareHierarchicalPlanner

    execute_action():
        -> Execute one explicitly requested registered Action.

    execute_plan():
        -> Execute an explicitly supplied multi-step plan.

    The exact-action path is separate so existing execute()
    semantics remain unchanged.
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
                agent=agent,
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
        """Execute exactly the supplied action and record its outcome.

        The requested action is never replaced by an explorer or planner.
        If the executor owns the experience, it manages that experience's
        lifecycle. If the caller already owns an experience, this method
        records the outcome but does not complete, fail, or close it.
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

        action_name = str(getattr(action, "name", "")).strip()
        if not action_name:
            raise ValueError("action must expose a non-empty name.")

        # Establish the goal context before recording an outcome so that
        # learning and reflection can associate the episode with the goal.
        if isinstance(goal, Goal):
            agent.set_goal(goal)

        current_experience = getattr(agent, "current_experience", None)
        owns_experience = current_experience is None

        if owns_experience:
            state = getattr(agent, "state", None)
            cycle = getattr(state, "cycle", 0)
            goal_description = (
                getattr(goal, "description", None)
                or str(goal or f"Execute action: {action_name}")
            )
            resolved_experience_id = (
                experience_id
                or f"{getattr(agent, 'name', 'agent')}"
                f"-single-action-{cycle + 1}"
            )
            agent.start_experience(
                experience_id=resolved_experience_id,
                goal=goal_description,
            )

        try:
            # Always check the live environment state immediately before
            # attempting the action.
            observation_before = dict(environment.observe())
            can_execute = getattr(action, "can_execute", None)

            if callable(can_execute) and not can_execute(observation_before):
                failure = {
                    "success": False,
                    "action": action_name,
                    "message": (
                        "Preconditions are not satisfied for action "
                        f"'{action_name}'."
                    ),
                    "failure_type": "precondition_failed",
                    "observation_before": dict(observation_before),
                    "observation_after": dict(observation_before),
                }

                # Use the Agent's canonical recording pipeline whenever it
                # exists. Lightweight test doubles may not implement it.
                record_experience = getattr(agent, "record_experience", None)
                if callable(record_experience):
                    record_experience(
                        action=action_name,
                        observation_before=dict(observation_before),
                        observation_after=dict(observation_before),
                        result=dict(failure),
                    )

                if owns_experience:
                    agent.fail_experience()

                return UnifiedExecutionResult(
                    decision=decision,
                    direct_action=action,
                    direct_result=failure,
                )

            # ExplorationExecutor handles act(), environment execution,
            # before/after observations, and normal episode recording.
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
            goal_satisfied=result.goal_satisfied,
        )

    def _execute_exploitation(
        self,
        decision: AgentDecision,
        goal: Optional[Goal],
        world_state: Optional[Dict[str, Any]],
        strategies: Optional[List[Strategy]],
        agent=None,
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

        planner_kwargs = {
            "goal": goal,
            "world_state": world_state,
            "strategies": (
                strategies if strategies is not None else []
            ),
        }

        # Pass episodic memory only when the agent exposes it. Omitting the
        # keyword for legacy agents/test doubles preserves their old planner
        # call contract.
        episodic_memory = getattr(agent, "episodic_memory", None)
        if episodic_memory is not None:
            planner_kwargs["episodic_memory"] = episodic_memory

        plan = self.hierarchical_planner.plan(**planner_kwargs)

        return UnifiedExecutionResult(
            decision=decision,
            plan=plan,
        )

    def execute_plan(
        self,
        *,
        decision: AgentDecision,
        agent,
        environment: Environment,
        goal: Goal,
        actions: List[Action],
        world_state: Optional[Dict[str, Any]] = None,
        experience_id: Optional[str] = None,
    ) -> UnifiedExecutionResult:
        """
        Execute a multi-step plan under one Agent experience.

        Stops at the first failed action and records each attempted
        action, including precondition failures, as an episode.
        """

        if decision is None:
            raise ValueError("decision cannot be None.")

        if decision.mode != DecisionMode.EXPLOIT:
            raise ValueError(
                "execute_plan requires an EXPLOIT decision."
            )

        if agent is None:
            raise ValueError("agent is required.")

        if environment is None:
            raise ValueError("environment is required.")

        if not isinstance(goal, Goal):
            raise ValueError("goal must be a Goal instance.")

        if not actions:
            return UnifiedExecutionResult(
                decision=decision,
                plan=[],
                plan_execution_results=(),
                goal_satisfied=goal.is_satisfied(
                    environment.observe()
                ),
            )

        if self.explorer is None:
            raise RuntimeError(
                "ActiveExplorer must be configured to execute a plan."
            )

        # Use the supplied plan, but reject actions that are not
        # registered with the Agent.
        registered_actions = getattr(agent, "actions", {})

        if isinstance(registered_actions, dict):
            registered_action_names = set(
                registered_actions.keys()
            )
        else:
            registered_action_names = {
                getattr(item, "name", None)
                for item in registered_actions
            }

        for action in actions:
            if not isinstance(action, Action):
                raise TypeError(
                    "Every plan step must be an Action."
                )

            if action.name not in registered_action_names:
                raise ValueError(
                    f"Action '{action.name}' is not registered."
                )

        agent.set_goal(goal)

        owns_experience = (
            getattr(agent, "current_experience", None) is None
        )

        if owns_experience:
            resolved_experience_id = (
                experience_id
                or (
                    f"{getattr(agent, 'name', 'agent')}"
                    f"-exploit-plan-{agent.state.cycle + 1}"
                )
            )

            agent.start_experience(
                experience_id=resolved_experience_id,
                goal=goal.description,
            )

        step_results = []
        completed_successfully = False
        goal_satisfied = False

        try:
            for action in actions:
                observation_before = dict(
                    environment.observe()
                )

                if not action.can_execute(observation_before):
                    # Record the failure through the normal Agent
                    # recording pipeline. This stores the episode in
                    # episodic memory and the active experience, and
                    # invokes the configured learning integrations.
                    failure_message = (
                        "Preconditions not satisfied."
                    )

                    failure_result = {
                        "success": False,
                        "action": action.name,
                        "message": failure_message,
                        "failure_type": "precondition_failed",
                        "observation_before": dict(
                            observation_before
                        ),
                        "observation_after": dict(
                            observation_before
                        ),
                    }

                    record_experience = getattr(
                        agent,
                        "record_experience",
                        None,
                    )

                    if not callable(record_experience):
                        raise RuntimeError(
                            "Agent must provide record_experience() "
                            "to record a failed plan step."
                        )

                    record_experience(
                        action=action.name,
                        observation_before=dict(
                            observation_before
                        ),
                        observation_after=dict(
                            observation_before
                        ),
                        result=dict(failure_result),
                    )

                    step_results.append(
                        {
                            "action": action.name,
                            "success": False,
                            "message": failure_message,
                            "result": failure_result,
                            "observation_before": dict(
                                observation_before
                            ),
                            "observation_after": dict(
                                observation_before
                            ),
                        }
                    )
                    break

                execution = self.explorer.executor.execute(
                    agent=agent,
                    environment=environment,
                    action=action,
                )

                step_results.append(
                    {
                        "action": action.name,
                        "success": execution.success,
                        "result": execution.result,
                        "observation_before": (
                            execution.observation_before
                        ),
                        "observation_after": (
                            execution.observation_after
                        ),
                    }
                )

                if not execution.success:
                    break

            final_state = environment.observe()
            goal_satisfied = goal.is_satisfied(final_state)

            completed_successfully = (
                len(step_results) == len(actions)
                and all(
                    step["success"]
                    for step in step_results
                )
                and goal_satisfied
            )

            if completed_successfully:
                goal.complete()
            else:
                goal.fail()

            if owns_experience:
                if completed_successfully:
                    agent.complete_experience()
                else:
                    agent.fail_experience()

            return UnifiedExecutionResult(
                decision=decision,
                plan=list(actions),
                plan_execution_results=tuple(
                    _PlanStepResult(
                        action=step["action"],
                        success=step["success"],
                        result=step.get("result"),
                    )
                    for step in step_results
                ),
                goal_satisfied=goal_satisfied,
            )

        except Exception:
            if owns_experience:
                try:
                    agent.fail_experience()
                except (RuntimeError, AttributeError):
                    pass
            raise

        finally:
            if owns_experience:
                agent.close_experience()