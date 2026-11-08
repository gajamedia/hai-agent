"""Adaptive planning with feasibility checks and a conservative plan-quality gate.

A learned strategy is considered only after it has been simulated against the
current world state. Its historical score cannot justify a longer plan than a
valid plan produced by the configured base planner. The base planner is treated
as a safe reference, not as a universally optimal planner.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Mapping, Tuple

from agent.core.action import Action
from agent.cognition.planner import Planner
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector
from agent.cognition.risk_strategy_selector import RiskAwareStrategySelector


class AdaptivePlanner:
    """Prefer a feasible learned strategy unless the base plan is safer/shorter."""

    def __init__(
        self,
        planner: Planner,
        strategy_selector: StrategySelector,
        risk_strategy_selector: Optional[RiskAwareStrategySelector] = None,
        avoid_unreliable_routes: bool = False,
        min_failure_evidence: int = 2,
        unreliable_success_rate: float = 0.5,
        alternative_max_depth: int = 8,
        alternative_max_nodes: int = 5000,
    ):
        self.planner = planner
        self.strategy_selector = strategy_selector
        self.risk_strategy_selector = risk_strategy_selector
        # Opt-in. Off by default so the base planner remains the sole safe
        # reference (history can never override a shorter valid route).
        self.avoid_unreliable_routes = avoid_unreliable_routes
        # A route is "unreliable" once it has at least ``min_failure_evidence``
        # recorded attempts and a success rate below ``unreliable_success_rate``.
        # Unreliable routes are avoided when a valid alternative exists, even if
        # the base planner (the safe reference) would otherwise choose them.
        self.min_failure_evidence = min_failure_evidence
        self.unreliable_success_rate = unreliable_success_rate
        self.alternative_max_depth = alternative_max_depth
        self.alternative_max_nodes = alternative_max_nodes

    def plan(
        self,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        goal: Optional[str],
        strategies: List[Strategy],
    ) -> List[Action]:
        """Return a verified feasible plan, preferring no worse than baseline.

        The base planner is run on the same state, goal and action registry for
        every call. If both plans are valid, a learned plan is accepted only if
        it is no longer than the base plan. If the base planner cannot produce a
        valid plan, a valid learned plan remains usable. This guards against
        misleading strategy history without discarding useful learned plans.
        """
        baseline_plan = self.planner.plan(
            world_state=world_state,
            goal_state=goal_state,
        )
        baseline_valid = self._plan_reaches_goal(
            baseline_plan, world_state, goal_state
        )

        # An already-satisfied goal should not trigger extra learned actions.
        if self._goal_satisfied(world_state, goal_state):
            return baseline_plan if baseline_valid else []

        active = [
            strategy for strategy in (strategies or [])
            if getattr(strategy, "is_active", getattr(strategy, "status", "active") == "active")
        ]
        blocked = (
            self._unreliable_routes(active, goal)
            if self.avoid_unreliable_routes
            else set()
        )
        feasible = self._filter_feasible_strategies(
            strategies=[
                strategy for strategy in active
                if tuple(strategy.steps) not in blocked
            ],
            world_state=world_state,
            goal_state=goal_state,
            goal=goal,
        )
        selected = self._select_strategy(
            strategies=feasible,
            goal=goal,
            world_state=world_state,
            goal_state=goal_state,
        )
        learned_plan = None
        if selected is not None:
            learned_plan = self._strategy_to_actions(
                strategy=selected,
                world_state=world_state,
                goal_state=goal_state,
            )

        learned_valid = learned_plan is not None and self._plan_reaches_goal(
            learned_plan, world_state, goal_state
        )

        # Evidence-based override: the base planner stays the safe reference
        # except when its chosen route has repeatedly failed in practice.
        if (
            baseline_valid
            and tuple(a.name for a in baseline_plan) in blocked
        ):
            if learned_valid:
                return learned_plan
            alternative = self._search_alternative(
                world_state, goal_state, blocked
            )
            if alternative is not None:
                return alternative
            return baseline_plan

        if learned_valid:
            if not baseline_valid:
                return learned_plan
            if len(learned_plan) <= len(baseline_plan):
                return learned_plan
            return baseline_plan

        # Keep the base planner's historical fallback contract, even when its
        # output is empty or invalid; callers can inspect/verify the outcome.
        return baseline_plan

    def _unreliable_routes(
        self, strategies: List[Strategy], goal: Optional[str]
    ) -> set:
        """Action-name sequences with enough evidence of repeated failure."""
        blocked = set()
        for strategy in strategies:
            if goal is not None and strategy.goal != goal:
                continue
            if (
                strategy.attempts >= self.min_failure_evidence
                and strategy.success_rate < self.unreliable_success_rate
            ):
                blocked.add(tuple(strategy.steps))
        return blocked

    def _search_alternative(
        self,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        blocked: set,
    ) -> Optional[List[Action]]:
        """Breadth-first search for the shortest valid plan not in ``blocked``.

        Bounded by depth and node budget so it cannot run away on large
        action spaces; returns ``None`` when nothing is found in budget.
        """
        queue: deque = deque([(dict(world_state), ())])
        explored = 0
        while queue and explored < self.alternative_max_nodes:
            state, sequence = queue.popleft()
            explored += 1
            names = tuple(a.name for a in sequence)
            # Padding a known-bad route with extra steps is still that route.
            if any(self._is_subsequence(route, names) for route in blocked):
                continue
            if sequence and self._goal_satisfied(state, goal_state):
                return list(sequence)
            if len(sequence) >= self.alternative_max_depth:
                continue
            for action in self.planner.actions:
                if not action.can_execute(state):
                    continue
                if all(state.get(k) == v for k, v in action.effects.items()):
                    continue  # no-op in this state
                next_state = dict(state)
                next_state.update(action.effects)
                queue.append((next_state, sequence + (action,)))
        return None

    @staticmethod
    def _is_subsequence(route: Tuple[str, ...], names: Tuple[str, ...]) -> bool:
        if not route:
            return False
        remaining = iter(names)
        return all(step in remaining for step in route)

    def _select_strategy(
        self,
        strategies: List[Strategy],
        goal: Optional[str],
        world_state: Optional[Dict[str, Any]] = None,
        goal_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Strategy]:
        if not strategies:
            return None

        candidates = self._filter_goal_strategies(strategies, goal)
        if not candidates:
            return None

        # Risk-aware ranking remains supported. The plan-quality gate above
        # prevents its historical score from overriding a shorter valid route.
        if self.risk_strategy_selector is not None:
            return self.risk_strategy_selector.select(candidates)

        if world_state is None:
            # Retain compatibility for direct legacy calls to this helper.
            return self.strategy_selector.select(candidates, goal=goal)

        return self.strategy_selector.select(
            candidates,
            goal=goal,
            world_state=world_state,
            actions=self.planner.actions,
            goal_state=goal_state,
        )

    @staticmethod
    def _filter_goal_strategies(
        strategies: List[Strategy], goal: Optional[str]
    ) -> List[Strategy]:
        if goal is None:
            return list(strategies)
        matching = [strategy for strategy in strategies if strategy.goal == goal]
        return matching if matching else list(strategies)

    def _filter_feasible_strategies(
        self,
        strategies: List[Strategy],
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        goal: Optional[str],
    ) -> List[Strategy]:
        candidates = self._filter_goal_strategies(strategies, goal)
        return [
            strategy for strategy in candidates
            if self._strategy_to_actions(
                strategy=strategy,
                world_state=world_state,
                goal_state=goal_state,
            ) is not None
        ]

    def _strategy_to_actions(
        self,
        strategy: Strategy,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
    ) -> Optional[List[Action]]:
        if not strategy.steps:
            return None

        action_by_name = {action.name: action for action in self.planner.actions}
        simulated_state = dict(world_state)
        plan: List[Action] = []

        for step in strategy.steps:
            action = action_by_name.get(step)
            if action is None or not action.can_execute(simulated_state):
                return None
            simulated_state.update(action.effects)
            plan.append(action)

        if not self._goal_satisfied(simulated_state, goal_state):
            return None
        return plan

    @staticmethod
    def _goal_satisfied(
        world_state: Mapping[str, Any], goal_state: Mapping[str, Any]
    ) -> bool:
        return all(world_state.get(key) == value for key, value in goal_state.items())

    @classmethod
    def _plan_reaches_goal(
        cls,
        plan: Optional[List[Action]],
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
    ) -> bool:
        if plan is None:
            return False
        state = dict(world_state)
        for action in plan:
            if action is None or not action.can_execute(state):
                return False
            state.update(action.effects)
        return cls._goal_satisfied(state, goal_state)
