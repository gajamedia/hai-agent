# from typing import Any, Dict, List, Optional

# from agent.core.action import Action
# from agent.cognition.planner import Planner
# from agent.learning.strategy import Strategy
# from agent.learning.strategy_selector import (
#     StrategySelector,
# )
# from agent.cognition.risk_strategy_selector import (
#     RiskAwareStrategySelector,
# )


# class AdaptivePlanner:

#     def __init__(
#         self,
#         planner: Planner,
#         strategy_selector: StrategySelector,
#         risk_strategy_selector: Optional[
#             RiskAwareStrategySelector
#         ] = None,
#     ):
#         self.planner = planner
#         self.strategy_selector = strategy_selector
#         self.risk_strategy_selector = (
#             risk_strategy_selector
#         )

#     def plan(
#         self,
#         world_state: Dict[str, Any],
#         goal_state: Dict[str, Any],
#         goal: Optional[str],
#         strategies: List[Strategy],
#     ) -> List[Action]:

#         strategies = [
#             strategy
#             for strategy in strategies
#             if strategy.is_active
#         ]

#         strategy = self._select_strategy(
#             strategies=strategies,
#             goal=goal,
#         )

#         if strategy is not None:

#             plan = self._strategy_to_actions(
#                 strategy=strategy,
#                 world_state=world_state,
#             )

#             if plan is not None:
#                 return plan

#         return self.planner.plan(
#             world_state=world_state,
#             goal_state=goal_state,
#         )

#     def _select_strategy(
#         self,
#         strategies: List[Strategy],
#         goal: Optional[str],
#     ) -> Optional[Strategy]:

#         if not strategies:
#             return None

#         # Risk-aware selection is optional.
#         #
#         # When configured, it takes priority over the
#         # historical StrategySelector.
#         if self.risk_strategy_selector is not None:

#             candidate_strategies = self._filter_goal_strategies(
#                 strategies=strategies,
#                 goal=goal,
#             )

#             if candidate_strategies:
#                 return self.risk_strategy_selector.select(
#                     candidate_strategies
#                 )

#         return self.strategy_selector.select(
#             strategies=strategies,
#             goal=goal,
#         )

#     def _filter_goal_strategies(
#         self,
#         strategies: List[Strategy],
#         goal: Optional[str],
#     ) -> List[Strategy]:

#         if goal is None:
#             return list(strategies)

#         matching = [
#             strategy
#             for strategy in strategies
#             if strategy.goal == goal
#         ]

#         if matching:
#             return matching

#         return list(strategies)

#     def _strategy_to_actions(
#         self,
#         strategy: Strategy,
#         world_state: Dict[str, Any],
#     ) -> Optional[List[Action]]:

#         actions = []

#         simulated_state = dict(
#             world_state
#         )

#         for step in strategy.steps:

#             action = next(
#                 (
#                     action
#                     for action in self.planner.actions
#                     if action.name == step
#                 ),
#                 None,
#             )

#             if action is None:
#                 return None

#             if not action.can_execute(
#                 simulated_state
#             ):
#                 return None

#             for key, value in action.effects.items():
#                 simulated_state[key] = value

#             actions.append(action)

#         return actions

from typing import Any, Dict, List, Optional

from agent.core.action import Action
from agent.cognition.planner import Planner
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)


class AdaptivePlanner:
    def __init__(
        self,
        planner: Planner,
        strategy_selector: StrategySelector,
        risk_strategy_selector: Optional[
            RiskAwareStrategySelector
        ] = None,
    ):
        self.planner = planner
        self.strategy_selector = strategy_selector
        self.risk_strategy_selector = (
            risk_strategy_selector
        )

    def plan(
        self,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        goal: Optional[str],
        strategies: List[Strategy],
    ) -> List[Action]:

        # Only active strategies may be considered.
        active_strategies = [
            strategy
            for strategy in (strategies or [])
            if strategy.is_active
        ]

        strategy = self._select_strategy(
            strategies=active_strategies,
            goal=goal,
        )

        if strategy is not None:
            plan = self._strategy_to_actions(
                strategy=strategy,
                world_state=world_state,
                goal_state=goal_state,
            )

            if plan is not None:
                return plan

        # Fallback when no strategy is suitable.
        return self.planner.plan(
            world_state=world_state,
            goal_state=goal_state,
        )

    def _select_strategy(
        self,
        strategies: List[Strategy],
        goal: Optional[str],
    ) -> Optional[Strategy]:

        if not strategies:
            return None

        # Risk-aware selection takes priority when configured.
        if self.risk_strategy_selector is not None:
            candidate_strategies = (
                self._filter_goal_strategies(
                    strategies=strategies,
                    goal=goal,
                )
            )

            if candidate_strategies:
                return self.risk_strategy_selector.select(
                    candidate_strategies
                )

        return self.strategy_selector.select(
            strategies=strategies,
            goal=goal,
        )

    def _filter_goal_strategies(
        self,
        strategies: List[Strategy],
        goal: Optional[str],
    ) -> List[Strategy]:

        if goal is None:
            return list(strategies)

        matching = [
            strategy
            for strategy in strategies
            if strategy.goal == goal
        ]

        if matching:
            return matching

        return list(strategies)

    def _strategy_to_actions(
        self,
        strategy: Strategy,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
    ) -> Optional[List[Action]]:

        actions: List[Action] = []
        simulated_state = dict(world_state)

        for step in strategy.steps:
            action = next(
                (
                    candidate
                    for candidate in self.planner.actions
                    if candidate.name == step
                ),
                None,
            )

            # Unknown action: reject this strategy.
            if action is None:
                return None

            # Preconditions must hold at each step.
            if not action.can_execute(simulated_state):
                return None

            # Simulate the action effects in order.
            simulated_state.update(action.effects)
            actions.append(action)

        # Reject incomplete strategies whose final state
        # does not satisfy every requested goal condition.
        if not all(
            simulated_state.get(key) == expected_value
            for key, expected_value in goal_state.items()
        ):
            return None

        return actions