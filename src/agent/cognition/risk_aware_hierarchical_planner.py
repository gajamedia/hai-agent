# from typing import Any, Dict, List, Optional

# from agent.core.action import Action
# from agent.core.goal import Goal
# from agent.cognition.hierarchical_planner import (
#     HierarchicalPlanner,
# )
# from agent.cognition.risk_strategy_selector import (
#     RiskAwareStrategySelector,
# )
# from agent.learning.strategy import Strategy


# class RiskAwareHierarchicalPlanner:
#     """
#     Combines hierarchical planning with risk-aware
#     strategy selection.

#     A learned strategy is accepted only when:

#     1. Its actions are registered.
#     2. Its preconditions are satisfied.
#     3. Its resulting state satisfies the hierarchy target.
#     4. Its dependency constraints are satisfied.
#     5. Risk-aware selection chooses it over alternatives.

#     If no strategy is suitable, the regular
#     HierarchicalPlanner is used.
#     """

#     def __init__(
#         self,
#         hierarchical_planner: HierarchicalPlanner,
#         risk_strategy_selector: Optional[
#             RiskAwareStrategySelector
#         ] = None,
#     ):
#         self.hierarchical_planner = (
#             hierarchical_planner
#         )

#         self.risk_strategy_selector = (
#             risk_strategy_selector
#             if risk_strategy_selector is not None
#             else RiskAwareStrategySelector()
#         )

#     def plan(
#         self,
#         goal: Goal,
#         world_state: Dict[str, Any],
#         strategies: Optional[
#             List[Strategy]
#         ] = None,
#     ) -> List[Action]:


#         strategies = [
#             strategy
#             for strategy in (strategies or [])
#             if strategy.is_active
#         ]
#         # Always let the existing hierarchical planner
#         # establish the structurally valid fallback.
#         fallback_plan = (
#             self.hierarchical_planner.plan(
#                 goal=goal,
#                 world_state=world_state,
#             )
#         )

#         if not strategies:
#             return fallback_plan

#         root = (
#             self.hierarchical_planner.decomposer
#             .decompose_recursive(goal)
#         )

#         # Validate hierarchy dependencies first.
#         all_goals = (
#             self.hierarchical_planner
#             ._all_goals(root)
#         )

#         for current_goal in all_goals:
#             current_goal.validate_dependencies()

#         matching_strategies = (
#             self._filter_goal_strategies(
#                 strategies=strategies,
#                 goal=goal,
#             )
#         )

#         if not matching_strategies:
#             return fallback_plan

#         for strategy, _score in (
#             self.risk_strategy_selector.rank(
#                 matching_strategies
#             )
#         ):

#             actions = (
#                 self._strategy_to_actions(
#                     strategy=strategy,
#                 )
#             )

#             if actions is None:
#                 continue

#             if not self._validate_strategy(
#                 actions=actions,
#                 root=root,
#                 world_state=world_state,
#             ):
#                 continue

#             return actions

#         return fallback_plan

#     def _filter_goal_strategies(
#         self,
#         strategies: List[Strategy],
#         goal: Goal,
#     ) -> List[Strategy]:

#         matching = [
#             strategy
#             for strategy in strategies
#             if strategy.goal == goal.description
#         ]

#         if matching:
#             return matching

#         return []

#     def _strategy_to_actions(
#         self,
#         strategy: Strategy,
#     ) -> Optional[List[Action]]:

#         planner = (
#             self.hierarchical_planner.planner
#         )

#         actions: List[Action] = []

#         for step in strategy.steps:

#             action = next(
#                 (
#                     action
#                     for action in planner.actions
#                     if action.name == step
#                 ),
#                 None,
#             )

#             if action is None:
#                 return None

#             actions.append(action)

#         return actions

#     def _validate_strategy(
#         self,
#         actions: List[Action],
#         root: Goal,
#         world_state: Dict[str, Any],
#     ) -> bool:

#         state = dict(
#             world_state
#         )

#         for action in actions:

#             if not action.can_execute(
#                 state
#             ):
#                 return False

#             state.update(
#                 action.effects
#             )

#         target_state = (
#             self._hierarchy_target_state(
#                 root
#             )
#         )

#         return self._state_satisfies(
#             state=state,
#             target_state=target_state,
#         )

#     def _hierarchy_target_state(
#         self,
#         root: Goal,
#     ) -> Dict[str, Any]:

#         target: Dict[str, Any] = {}

#         all_goals = [
#             root
#         ] + root.flatten()

#         for current_goal in all_goals:

#             for key, value in (
#                 current_goal.target_state.items()
#             ):

#                 # Conflicting target states indicate that
#                 # this strategy cannot represent a coherent
#                 # final hierarchy state.
#                 if (
#                     key in target
#                     and target[key] != value
#                 ):
#                     raise ValueError(
#                         "Conflicting goal target states."
#                     )

#                 target[key] = value

#         return target

#     def _state_satisfies(
#         self,
#         state: Dict[str, Any],
#         target_state: Dict[str, Any],
#     ) -> bool:

#         return all(
#             state.get(key) == value
#             for key, value in target_state.items()
#         )

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
    3. Its resulting state satisfies a non-empty,
       coherent hierarchy target.
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
        strategies: Optional[List[Strategy]] = None,
        episodic_memory: Any = None,
    ) -> List[Action]:

        active_strategies = [
            strategy
            for strategy in (strategies or [])
            if strategy.is_active
        ]

        # Always establish the regular hierarchical plan first.
        fallback_plan = self.hierarchical_planner.plan(
            goal=goal,
            world_state=world_state,
        )

        if not active_strategies:
            return fallback_plan

        root = (
            self.hierarchical_planner.decomposer
            .decompose_recursive(goal)
        )

        # Validate dependency cycles across the hierarchy.
        all_goals = (
            self.hierarchical_planner._all_goals(root)
        )

        for current_goal in all_goals:
            current_goal.validate_dependencies()

        matching_strategies = self._filter_goal_strategies(
            strategies=active_strategies,
            goal=goal,
        )

        if not matching_strategies:
            return fallback_plan

        ranked_strategies = self._rank_with_episodic_memory(
            strategies=matching_strategies,
            goal=goal,
            world_state=world_state,
            episodic_memory=episodic_memory,
        )

        for strategy, _score in ranked_strategies:
            actions = self._strategy_to_actions(
                strategy=strategy,
            )

            if actions is None:
                continue

            try:
                valid = self._validate_strategy(
                    actions=actions,
                    root=root,
                    world_state=world_state,
                )
            except ValueError:
                # An incoherent learned target must not prevent
                # the regular hierarchical fallback from being used.
                continue

            if not valid:
                continue

            return actions

        return fallback_plan

    def _rank_with_episodic_memory(
        self,
        strategies: List[Strategy],
        goal: Goal,
        world_state: Dict[str, Any],
        episodic_memory: Any = None,
    ) -> List[tuple[Strategy, float]]:
        """Rank strategies using the normal risk score plus bounded
        evidence from relevant past episodes.

        Episodic retrieval scores indicate relevance/evidence quality, not
        outcome probability. Only episodes whose action matches a strategy
        step and whose result explicitly contains a boolean ``success`` are
        used. Missing memory, malformed episodes, or retrieval errors leave
        the original selector ranking unchanged.
        """
        base_ranked = self.risk_strategy_selector.rank(strategies)
        retrieve = getattr(episodic_memory, "retrieve", None)
        if not callable(retrieve):
            return base_ranked

        adjusted = []
        for strategy, base_score in base_ranked:
            evidence: list[tuple[bool, float]] = []

            for step in getattr(strategy, "steps", []) or []:
                try:
                    episodes = retrieve(
                        goal=getattr(goal, "description", None),
                        action=step,
                        context=world_state,
                        limit=20,
                        min_score=0.0,
                    )
                except (TypeError, ValueError, RuntimeError):
                    # A non-conforming or unavailable memory backend must
                    # not break planning.
                    continue
                except Exception:
                    continue

                for episode in episodes or []:
                    episode_action = getattr(episode, "action", None)
                    if episode_action != step:
                        continue

                    episode_goal = getattr(episode, "goal", None)
                    if (
                        episode_goal
                        and getattr(goal, "description", None)
                        and episode_goal != goal.description
                    ):
                        continue

                    result = getattr(episode, "result", None)
                    if not isinstance(result, dict):
                        continue
                    outcome = result.get("success")
                    if not isinstance(outcome, bool):
                        continue

                    # Relevance scores are intentionally not treated as
                    # success probabilities. Evidence is bounded by a
                    # conservative per-step prior and sample count.
                    evidence.append((outcome, 1.0))

            if not evidence:
                adjusted.append((strategy, base_score))
                continue

            successes = sum(1 for success, _ in evidence if success)
            failures = len(evidence) - successes

            # Beta(1, 1) smoothing prevents one observation from causing
            # an extreme ranking change.
            success_rate = (successes + 1.0) / (
                successes + failures + 2.0
            )
            confidence = len(evidence) / (len(evidence) + 4.0)

            # Maximum absolute adjustment is 0.10. Existing risk,
            # uncertainty, cost, reflection and progress terms remain
            # authoritative.
            memory_adjustment = 0.20 * (
                success_rate - 0.5
            ) * confidence
            adjusted.append(
                (strategy, round(base_score + memory_adjustment, 6))
            )

        # Stable sorting preserves the selector's order on equal adjusted
        # scores, so legacy ranking remains the tie-breaker.
        return sorted(
            adjusted,
            key=lambda item: item[1],
            reverse=True,
        )

    def _filter_goal_strategies(
        self,
        strategies: List[Strategy],
        goal: Goal,
    ) -> List[Strategy]:

        return [
            strategy
            for strategy in strategies
            if strategy.goal == goal.description
        ]

    def _strategy_to_actions(
        self,
        strategy: Strategy,
    ) -> Optional[List[Action]]:

        planner = self.hierarchical_planner.planner
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

        state = dict(world_state)

        for action in actions:
            if not action.can_execute(state):
                return False

            state.update(action.effects)

        target_state = self._hierarchy_target_state(root)

        # An empty target provides no evidence that the strategy
        # reaches the intended goal.
        if not target_state:
            return False

        return self._state_satisfies(
            state=state,
            target_state=target_state,
        )

    def _hierarchy_target_state(
        self,
        root: Goal,
    ) -> Dict[str, Any]:

        target: Dict[str, Any] = {}
        all_goals = [root] + root.flatten()

        for current_goal in all_goals:
            for key, value in current_goal.target_state.items():
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

        if not target_state:
            return False

        return all(
            state.get(key) == value
            for key, value in target_state.items()
        )