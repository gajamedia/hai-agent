from typing import Any, Dict, List, Optional

from agent.core.action import Action
from agent.core.experience import Experience

from agent.cognition.adaptive_planner import (
    AdaptivePlanner,
)

from agent.learning.reflection import (
    ReflectionEngine,
)

from agent.learning.reflection_strategy_updater import (
    ReflectionStrategyUpdater,
)

from agent.learning.strategy import Strategy

from agent.memory.reflection import ReflectionMemory


class SelfCorrectionResult:

    def __init__(
        self,
        success: bool,
        experience: Experience,
        reflection: Optional[ReflectionMemory],
        replanned: bool,
    ):
        self.success = success
        self.experience = experience
        self.reflection = reflection
        self.replanned = replanned


class SelfCorrectionRuntime:

    def __init__(
        self,
        adaptive_planner: AdaptivePlanner,
        reflection_engine: ReflectionEngine,
        strategy_updater: ReflectionStrategyUpdater,
    ):
        self.adaptive_planner = adaptive_planner
        self.reflection_engine = reflection_engine
        self.strategy_updater = strategy_updater

    def execute_with_correction(
        self,
        environment,
        action_system,
        agent,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        goal: str,
        strategies: List[Strategy],
        experience_id: str,
        max_attempts: int = 2,
    ) -> SelfCorrectionResult:

        reflection = None
        last_experience = None

        for attempt in range(
            1,
            max_attempts + 1,
        ):

            plan = self.adaptive_planner.plan(
                world_state=world_state,
                goal_state=goal_state,
                goal=goal,
                strategies=strategies,
            )

            experience = Experience(
                experience_id=(
                    f"{experience_id}-attempt-{attempt}"
                ),
                goal=goal,
            )

            last_experience = experience

            failed_action = None
            failed_episode = None

            for action in plan:

                observation_before = (
                    environment.observe()
                )

                agent.act(
                    action.name
                )

                result = action_system.execute(
                    action.name
                )

                observation_after = (
                    environment.observe()
                )

                episode = agent.record_experience(
                    action=action.name,
                    observation_before=(
                        observation_before
                    ),
                    observation_after=(
                        observation_after
                    ),
                    result=result,
                )

                experience.add_episode(
                    episode
                )

                agent.observe(
                    observation_after
                )

                agent.step()

                if not result.get(
                    "success",
                    False,
                ):

                    failed_action = action
                    failed_episode = episode

                    break

            final_world = environment.observe()

            if (
                failed_action is None
                and final_world.get("door")
                == "unlocked"
            ):

                experience.complete()

                return SelfCorrectionResult(
                    success=True,
                    experience=experience,
                    reflection=None,
                    replanned=(
                        attempt > 1
                    ),
                )

            experience.fail()

            if (
                failed_action is None
                or failed_episode is None
            ):
                continue

            reflection = (
                self.reflection_engine.reflect(
                    episode=failed_episode,
                    action=failed_action,
                )
            )

            if reflection is None:
                continue

            strategy = self._find_related_strategy(
                failed_action=failed_action.name,
                strategies=strategies,
            )

            if strategy is not None:
                self.strategy_updater.apply(
                    strategy=strategy,
                    reflection=reflection,
                )

            world_state = environment.observe()

        return SelfCorrectionResult(
            success=False,
            experience=last_experience,
            reflection=reflection,
            replanned=(
                max_attempts > 1
            ),
        )

    def _find_related_strategy(
        self,
        failed_action: str,
        strategies: List[Strategy],
    ) -> Optional[Strategy]:

        candidates = [
            strategy
            for strategy in strategies
            if failed_action in strategy.steps
        ]

        if not candidates:
            return None

        return candidates[0]
