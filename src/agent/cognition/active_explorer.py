from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from agent.core.action import Action

from agent.cognition.exploration import (
    ExplorationDecision,
    ExplorationDecisionMaker,
)

from agent.cognition.exploration_policy import (
    ActionExplorationPolicy,
)

from agent.cognition.exploration_executor import (
    ExplorationExecutor,
    ExplorationResult,
)

from agent.cognition.memory_exploration import (
    MemoryAwareExploration,
)

if TYPE_CHECKING:
    from agent.core.agent import Agent
    from agent.core.goal import Goal


@dataclass(frozen=True)
class ActiveExplorationResult:
    """
    Result of one active exploration attempt.
    """

    decision: ExplorationDecision
    execution: Optional[ExplorationResult]
    selected_action: Optional[Action] = None

    @property
    def explored(self) -> bool:
        return self.execution is not None

    @property
    def success(self) -> bool:
        if self.execution is None:
            return False

        return bool(
            self.execution.result.get(
                "success",
                False,
            )
        )

    @property
    def action(self) -> Optional[Action]:
        """
        Return the Action object selected for exploration.
        """

        return self.selected_action


class ActiveExplorer:
    """
    Coordinates one active exploration cycle.

    Flow:

        memory
          ↓
        decision
          ↓
        action selection
          ↓
        experience
          ↓
        execution
          ↓
        learning
    """

    def __init__(
        self,
        decision_maker: Optional[
            ExplorationDecisionMaker
        ] = None,
        memory_exploration: Optional[
            MemoryAwareExploration
        ] = None,
        action_policy: Optional[
            ActionExplorationPolicy
        ] = None,
        executor: Optional[
            ExplorationExecutor
        ] = None,
    ):
        self.decision_maker = (
            decision_maker
            if decision_maker is not None
            else ExplorationDecisionMaker()
        )

        self.memory_exploration = (
            memory_exploration
            if memory_exploration is not None
            else MemoryAwareExploration(
                decision_maker=self.decision_maker
            )
        )

        self.action_policy = (
            action_policy
            if action_policy is not None
            else ActionExplorationPolicy()
        )

        self.executor = (
            executor
            if executor is not None
            else ExplorationExecutor()
        )

    # =====================================================
    # Decision
    # =====================================================

    def decide(
        self,
        procedural_memory: Any = None,
        causal_memory: Any = None,
        risk: float = 0.0,
    ) -> ExplorationDecision:
        """
        Decide whether exploration should happen.
        """

        assessment = self.memory_exploration.assess(
            procedural_memory=procedural_memory,
            causal_memory=causal_memory,
        )

        return self.decision_maker.decide(
            uncertainty=assessment.uncertainty,
            expected_information_gain=(
                assessment.expected_information_gain
            ),
            risk=risk,
        )

    # =====================================================
    # One exploration cycle
    # =====================================================

    def explore_once(
        self,
        agent: Agent,
        environment: Any,
        actions: list[Action],
        information_gain: Optional[
            dict[str, float]
        ] = None,
        risk: Optional[
            dict[str, float]
        ] = None,
        goal: Goal = None,
        experience_id: Optional[str] = None,
        exploration_risk: float = 0.0,
        action_cost: Optional[
            dict[str, float]
        ] = None,
        action_risk: Optional[
            dict[str, float]
        ] = None,
    ) -> ActiveExplorationResult:

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        if agent is None:
            raise ValueError(
                "agent is required."
            )

        if environment is None:
            raise ValueError(
                "environment is required."
            )

        if goal is None:
            raise ValueError(
                "goal is required."
            )

        # -------------------------------------------------
        # Defaults
        # -------------------------------------------------

        information_gain = (
            information_gain
            if information_gain is not None
            else {}
        )

        action_cost = (
            action_cost
            if action_cost is not None
            else {}
        )

        # -------------------------------------------------
        # Backward-compatible risk parameter
        #
        # Existing API:
        #
        #     risk={"open_box": 0.1}
        #
        # Newer API:
        #
        #     action_risk={"open_box": 0.1}
        #
        # Support both.
        # -------------------------------------------------

        if action_risk is None:
            action_risk = (
                risk
                if risk is not None
                else {}
            )

        # -------------------------------------------------
        # Read latest memory
        # -------------------------------------------------

        procedural_memory = (
            agent.procedural_memory.latest()
        )

        causal_memory = (
            agent.causal_memory.latest()
        )

        # -------------------------------------------------
        # Decide whether to explore
        # -------------------------------------------------

        decision = self.decide(
            procedural_memory=procedural_memory,
            causal_memory=causal_memory,
            risk=exploration_risk,
        )

        # -------------------------------------------------
        # Exploit selected
        # -------------------------------------------------

        if not decision.explore:
            return ActiveExplorationResult(
                decision=decision,
                execution=None,
                selected_action=None,
            )

        # -------------------------------------------------
        # Current world state
        # -------------------------------------------------

        world_state = environment.observe()

        # -------------------------------------------------
        # Select exploratory action
        # -------------------------------------------------

        candidate = self.action_policy.select(
            actions=actions,
            world_state=world_state,
            information_gain=information_gain,
            action_risk=action_risk,
            action_cost=action_cost,
        )

        if candidate is None:
            return ActiveExplorationResult(
                decision=decision,
                execution=None,
                selected_action=None,
            )

        action = candidate.action

        # -------------------------------------------------
        # Set agent goal
        # -------------------------------------------------

        agent.set_goal(goal)

        # -------------------------------------------------
        # Start experience
        # -------------------------------------------------

        if experience_id is None:
            experience_id = (
                f"{agent.name}-exploration-"
                f"{agent.state.cycle + 1}"
            )

        agent.start_experience(
            experience_id=experience_id,
            goal=goal.description,
        )

        # -------------------------------------------------
        # Execute exploration
        # -------------------------------------------------

        try:
            execution = self.executor.execute(
                agent=agent,
                environment=environment,
                action=action,
            )

            success = bool(
                execution.result.get(
                    "success",
                    False,
                )
            )

            if success:
                agent.complete_experience()
            else:
                agent.fail_experience()

            return ActiveExplorationResult(
                decision=decision,
                execution=execution,
                selected_action=action,
            )

        finally:
            agent.close_experience()
