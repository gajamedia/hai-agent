from typing import Any, Dict, List, Optional

from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.hierarchical_planner import (
    HierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)
from agent.learning.strategy import Strategy


class RiskAwareHierarchicalPlanner:
    """
    Combines hierarchical planning with risk-aware
    strategy selection.

    A learned strategy is accepted only when:

    1. Its actions are registered.
    2. Its preconditions are satisfied.
    3. Its resulting state satisfies the hierarchy target.
    4. Its dependency constraints are satisfied.
    5. Risk-aware selection chooses it over alternatives.

    If no strategy is suitable, the regular
    HierarchicalPlanner is used.
    """

    def __init__(
        self,
        hierarchical_planner: HierarchicalPlanner,
        risk_strategy_selector: Optional[
            RiskAwareStrategySelector
        ] = None,
    ):
        self.hierarchical_planner = (
            hierarchical_planner
        )

        self.risk_strategy_selector = (
            risk_strategy_selector
            if risk_strategy_selector is not None
            else RiskAwareStrategySelector()
        )

    def plan(
        self,
        goal: Goal,
        world_state: Dict[str, Any],
        strategies: Optional[
            List[Strategy]
        ] = None,
    ) -> List[Action]:

        # Always let the existing hierarchical planner
        # establish the structurally valid fallback.
        fallback_plan = (
            self.hierarchical_planner.plan(
                goal=goal,
                world_state=world_state,
            )
        )

        if not strategies:
            return fallback_plan

        root = (
            self.hierarchical_planner.decomposer
            .decompose_recursive(goal)
        )

        # Validate hierarchy dependencies first.
        all_goals = (
            self.hierarchical_planner
            ._all_goals(root)
        )

        for current_goal in all_goals:
            current_goal.validate_dependencies()

        matching_strategies = (
            self._filter_goal_strategies(
                strategies=strategies,
                goal=goal,
            )
        )

        if not matching_strategies:
            return fallback_plan

        for strategy, _score in (
            self.risk_strategy_selector.rank(
                matching_strategies
            )
        ):

            actions = (
                self._strategy_to_actions(
                    strategy=strategy,
                )
            )

            if actions is None:
                continue

            if not self._validate_strategy(
                actions=actions,
                root=root,
                world_state=world_state,
            ):
                continue

            return actions

        return fallback_plan

    def _filter_goal_strategies(
        self,
        strategies: List[Strategy],
        goal: Goal,
    ) -> List[Strategy]:

        matching = [
            strategy
            for strategy in strategies
            if strategy.goal == goal.description
        ]

        if matching:
            return matching

        return []

    def _strategy_to_actions(
        self,
        strategy: Strategy,
    ) -> Optional[List[Action]]:

        planner = (
            self.hierarchical_planner.planner
        )

        actions: List[Action] = []

        for step in strategy.steps:

            action = next(
                (
                    action
                    for action in planner.actions
                    if action.name == step
                ),
                None,
            )

            if action is None:
                return None

            actions.append(action)

        return actions

    def _validate_strategy(
        self,
        actions: List[Action],
        root: Goal,
        world_state: Dict[str, Any],
    ) -> bool:

        state = dict(
            world_state
        )

        for action in actions:

            if not action.can_execute(
                state
            ):
                return False

            state.update(
                action.effects
            )

        target_state = (
            self._hierarchy_target_state(
                root
            )
        )

        return self._state_satisfies(
            state=state,
            target_state=target_state,
        )

    def _hierarchy_target_state(
        self,
        root: Goal,
    ) -> Dict[str, Any]:

        target: Dict[str, Any] = {}

        all_goals = [
            root
        ] + root.flatten()

        for current_goal in all_goals:

            for key, value in (
                current_goal.target_state.items()
            ):

                # Conflicting target states indicate that
                # this strategy cannot represent a coherent
                # final hierarchy state.
                if (
                    key in target
                    and target[key] != value
                ):
                    raise ValueError(
                        "Conflicting goal target states."
                    )

                target[key] = value

        return target

    def _state_satisfies(
        self,
        state: Dict[str, Any],
        target_state: Dict[str, Any],
    ) -> bool:

        return all(
            state.get(key) == value
            for key, value in target_state.items()
        )
