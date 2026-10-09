from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from agent.core.action import Action

if TYPE_CHECKING:
    from agent.core.agent import Agent
    from agent.core.experience import Experience


@dataclass(frozen=True)
class ExplorationResult:
    """
    Result of executing one exploration action.
    """

    action: str
    observation_before: dict[str, Any]
    observation_after: dict[str, Any]
    result: dict[str, Any]
    episode: Optional[Experience] = None
    cycle: int = 0

    @property
    def success(self) -> bool:
        return bool(
            self.result.get(
                "success",
                False,
            )
        )


class ExplorationExecutor:
    """
    Executes an exploratory action through the environment
    and records the experience through Agent.
    """

    def execute(
        self,
        agent: Agent,
        environment: Any,
        action: Action,
    ) -> ExplorationResult:

        # -------------------------------------------------
        # Validate dependencies
        # -------------------------------------------------

        if agent is None:
            raise ValueError(
                "agent cannot be None."
            )

        if environment is None:
            raise ValueError(
                "environment cannot be None."
            )

        if action is None:
            raise ValueError(
                "action cannot be None."
            )

        # -------------------------------------------------
        # Active experience is mandatory
        # -------------------------------------------------

        if agent.current_experience is None:
            raise RuntimeError(
                "No active experience."
            )

        # -------------------------------------------------
        # Observe before
        # -------------------------------------------------

        observation_before = environment.observe()

        # -------------------------------------------------
        # Tell agent which action is being executed
        # -------------------------------------------------

        agent.act(
            action.name
        )

        # -------------------------------------------------
        # Execute action in environment
        # -------------------------------------------------

        result = environment.execute(
            action.name
        )

        # -------------------------------------------------
        # Observe after
        # -------------------------------------------------

        observation_after = environment.observe()

        # -------------------------------------------------
        # Record experience
        # -------------------------------------------------

        episode = agent.record_experience(
            action=action.name,
            observation_before=observation_before,
            observation_after=observation_after,
            result=result,
        )

        # -------------------------------------------------
        # Return result
        #
        # IMPORTANT:
        # ExplorationResult.action is intentionally a
        # string because this is the existing public API.
        # -------------------------------------------------

        return ExplorationResult(
            action=action.name,
            observation_before=observation_before,
            observation_after=observation_after,
            result=result,
            episode=episode,
            cycle=agent.state.cycle,
        )
