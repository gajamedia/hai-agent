"""Context-aware selection of learned strategies.

The historical goal-only API remains supported. Callers that have a world state
and action catalog should pass both so infeasible strategies are removed before
the learned score is compared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from agent.core.action import Action
from agent.learning.strategy import Strategy


class StrategySelector:
    """Rank learned strategies, optionally filtering by current-state feasibility."""

    def select(
        self,
        strategies: Sequence[Strategy],
        goal: Optional[str] = None,
        *,
        world_state: Optional[Mapping[str, Any]] = None,
        actions: Optional[Sequence[Action] | Mapping[str, Action]] = None,
        goal_state: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Strategy]:
        """Return the highest-scoring eligible strategy.

        Args:
            strategies: Strategies to consider.
            goal: Optional goal label used to filter candidates.
            world_state: Current world state. When supplied, each strategy's
                action sequence is simulated against preconditions and effects.
            actions: Action catalog as a sequence or name-to-action mapping.
                Required when ``world_state`` is supplied.
            goal_state: Optional target state that the simulated strategy must
                achieve. It is only evaluated when context-aware filtering is on.

        Backward compatibility:
            ``select(strategies, goal=...)`` retains the previous score-only
            behavior. Context-aware callers should pass both ``world_state`` and
            ``actions`` explicitly.
        """
        if (world_state is None) != (actions is None):
            raise ValueError(
                "world_state and actions must be supplied together for "
                "context-aware strategy selection."
            )

        candidates = list(strategies or [])

        if goal is not None:
            candidates = [
                strategy for strategy in candidates
                if strategy.goal == goal
            ]

        if world_state is not None and actions is not None:
            action_map = self._normalize_actions(actions)
            candidates = [
                strategy
                for strategy in candidates
                if self._is_feasible(
                    strategy=strategy,
                    world_state=world_state,
                    actions=action_map,
                    goal_state=goal_state,
                )
            ]

        if not candidates:
            return None

        # Stable sort preserves caller order for equal scores.
        return max(candidates, key=self._score)

    @staticmethod
    def _normalize_actions(
        actions: Sequence[Action] | Mapping[str, Action],
    ) -> dict[str, Action]:
        if isinstance(actions, Mapping):
            action_map = dict(actions)
        else:
            action_map = {
                action.name: action
                for action in actions
            }

        # Reject malformed catalogs rather than silently treating a broken
        # action catalog as a feasible plan.
        for name, action in action_map.items():
            if not isinstance(action, Action):
                raise TypeError(
                    f"Action catalog entry {name!r} is not an Action instance."
                )
            if name != action.name:
                raise ValueError(
                    f"Action catalog key {name!r} does not match "
                    f"action.name {action.name!r}."
                )
        return action_map

    @staticmethod
    def _is_feasible(
        strategy: Strategy,
        world_state: Mapping[str, Any],
        actions: Mapping[str, Action],
        goal_state: Optional[Mapping[str, Any]],
    ) -> bool:
        """Simulate declared preconditions/effects without executing handlers."""
        if not strategy.steps:
            return False

        simulated_state = dict(world_state)

        for step_name in strategy.steps:
            action = actions.get(step_name)
            if action is None:
                return False

            if not action.can_execute(simulated_state):
                return False

            # Only declared effects are simulated. No handlers or external
            # side effects are run during candidate filtering.
            simulated_state.update(action.effects)

        if goal_state is not None:
            if any(
                simulated_state.get(key) != expected
                for key, expected in goal_state.items()
            ):
                return False

        return True

    @staticmethod
    def _score(strategy: Strategy) -> float:
        if strategy.attempts == 0:
            return 0.0

        success_score = strategy.success_rate
        action_penalty = strategy.average_actions * 0.01
        reflection_penalty = strategy.reflection_penalty

        return round(
            success_score - action_penalty - reflection_penalty,
            6,
        )
