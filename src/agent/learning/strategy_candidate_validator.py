
"""
Strategy candidate validation subsystem.

Validates evolved strategy candidates before they are
considered for registration or selection.

Validation is deterministic and side-effect free with
respect to the supplied strategy store and world state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from agent.core.action import Action
from agent.learning.strategy import Strategy


@dataclass(frozen=True)
class StrategyValidationResult:
    """
    Result of validating a strategy candidate.
    """

    strategy: Strategy
    valid: bool
    reason: str
    failed_step: Optional[str] = None

    @property
    def strategy_name(self) -> str:
        return self.strategy.name


class StrategyCandidateValidator:
    """
    Validate a strategy candidate against available actions,
    world state, goal state, lifecycle status, and optional
    registered strategies.
    """

    def __init__(
        self,
        actions: Optional[Iterable[Action]] = None,
        strategies: Optional[Iterable[Strategy]] = None,
    ) -> None:
        self.actions: Dict[str, Action] = {}

        if actions is not None:
            for action in actions:
                if action is None:
                    continue

                self.actions[action.name] = action

        self.strategies = list(
            strategies
            if strategies is not None
            else []
        )

    # =====================================================
    # Helpers
    # =====================================================

    def _same_strategy(
        self,
        first: Strategy,
        second: Strategy,
    ) -> bool:
        return (
            first.goal == second.goal
            and first.steps == second.steps
        )

    def _is_duplicate(
        self,
        candidate: Strategy,
    ) -> bool:
        for strategy in self.strategies:
            if self._same_strategy(
                strategy,
                candidate,
            ):
                return True

        return False

    def _goal_satisfied(
        self,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
    ) -> bool:
        if not goal_state:
            return False

        return all(
            world_state.get(key) == expected
            for key, expected in goal_state.items()
        )

    def _simulate(
        self,
        candidate: Strategy,
        world_state: Dict[str, Any],
    ) -> tuple[
        bool,
        Dict[str, Any],
        Optional[str],
        str,
    ]:
        simulated_state = dict(world_state)

        if not candidate.steps:
            return (
                False,
                simulated_state,
                None,
                "candidate has no steps",
            )

        for step in candidate.steps:
            action = self.actions.get(step)

            if action is None:
                return (
                    False,
                    simulated_state,
                    step,
                    "action is not registered",
                )

            if not action.can_execute(simulated_state):
                return (
                    False,
                    simulated_state,
                    step,
                    "action preconditions are not satisfied",
                )

            for key, value in action.effects.items():
                simulated_state[key] = value

        return (
            True,
            simulated_state,
            None,
            "sequence is executable",
        )

    # =====================================================
    # Public validation API
    # =====================================================

    def validate(
        self,
        source_strategy: Optional[Strategy],
        candidate: Optional[Strategy],
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
    ) -> StrategyValidationResult:

        # -------------------------------------------------
        # Required argument validation
        # -------------------------------------------------

        if candidate is None:
            raise ValueError(
                "candidate cannot be None."
            )

        # -------------------------------------------------
        # Candidate lifecycle validation
        # -------------------------------------------------

        # Strategies are expected to be active by default.
        # The fallback preserves compatibility with strategy
        # test doubles that do not expose a status attribute.
        candidate_status = getattr(
            candidate,
            "status",
            "active",
        )

        if candidate_status != "active":
            return StrategyValidationResult(
                strategy=candidate,
                valid=False,
                reason=(
                    "candidate is not active "
                    f"(status: {candidate_status})"
                ),
            )

        if world_state is None:
            raise ValueError(
                "world_state cannot be None."
            )

        if goal_state is None:
            raise ValueError(
                "goal_state cannot be None."
            )

        # -------------------------------------------------
        # Source strategy comparison
        # -------------------------------------------------

        if source_strategy is not None:

            if candidate.goal != source_strategy.goal:
                return StrategyValidationResult(
                    strategy=candidate,
                    valid=False,
                    reason=(
                        "candidate goal differs "
                        "from source strategy"
                    ),
                )

            if self._same_strategy(
                source_strategy,
                candidate,
            ):
                return StrategyValidationResult(
                    strategy=candidate,
                    valid=False,
                    reason=(
                        "candidate is identical "
                        "to source strategy"
                    ),
                )

        # -------------------------------------------------
        # Basic structure
        # -------------------------------------------------

        if not candidate.goal.strip():
            return StrategyValidationResult(
                strategy=candidate,
                valid=False,
                reason="candidate goal is empty",
            )

        if not candidate.steps:
            return StrategyValidationResult(
                strategy=candidate,
                valid=False,
                reason="candidate has no steps",
            )

        # -------------------------------------------------
        # Duplicate strategy
        # -------------------------------------------------

        if self._is_duplicate(candidate):
            return StrategyValidationResult(
                strategy=candidate,
                valid=False,
                reason=(
                    "candidate already exists "
                    "in strategy registry"
                ),
            )

        # -------------------------------------------------
        # Simulate candidate
        # -------------------------------------------------

        (
            executable,
            simulated_state,
            failed_step,
            simulation_reason,
        ) = self._simulate(
            candidate=candidate,
            world_state=world_state,
        )

        if not executable:
            return StrategyValidationResult(
                strategy=candidate,
                valid=False,
                reason=simulation_reason,
                failed_step=failed_step,
            )

        # -------------------------------------------------
        # Goal validation
        # -------------------------------------------------

        if not self._goal_satisfied(
            simulated_state,
            goal_state,
        ):
            return StrategyValidationResult(
                strategy=candidate,
                valid=False,
                reason=(
                    "candidate does not "
                    "reach goal state"
                ),
            )

        return StrategyValidationResult(
            strategy=candidate,
            valid=True,
            reason="candidate is valid",
        )

    # =====================================================
    # Batch validation
    # =====================================================

    def validate_all(
        self,
        source_strategy: Optional[Strategy],
        candidates: Iterable[Strategy],
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
    ) -> list[StrategyValidationResult]:

        if candidates is None:
            raise ValueError(
                "candidates cannot be None."
            )

        return [
            self.validate(
                source_strategy=source_strategy,
                candidate=candidate,
                world_state=world_state,
                goal_state=goal_state,
            )
            for candidate in candidates
        ]
