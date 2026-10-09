# from typing import List, Optional

# from agent.core.experience import Experience
# from agent.learning.strategy import Strategy
# from agent.learning.strategy_store import StrategyStore


# class StrategyLearning:

#     def __init__(
#         self,
#         strategy_store: StrategyStore,
#     ):
#         self.strategy_store = strategy_store

#     def learn_from_experience(
#         self,
#         experience: Experience,
#     ) -> Optional[Strategy]:

#         if not experience.episodes:
#             return None

#         steps = experience.actions

#         if not steps:
#             return None

#         strategy = (
#             self.strategy_store.get_or_create(
#                 goal=experience.goal,
#                 steps=steps,
#             )
#         )

#         if experience.completed:

#             strategy.record_success(
#                 experience_id=(
#                     experience.experience_id
#                 ),
#                 action_count=(
#                     experience.action_count
#                 ),
#             )

#         elif experience.failed:

#             strategy.record_failure(
#                 experience_id=(
#                     experience.experience_id
#                 ),
#                 action_count=(
#                     experience.action_count
#                 ),
#             )

#         return strategy

#     def learn_from_experiences(
#         self,
#         experiences: List[Experience],
#     ) -> List[Strategy]:

#         learned = []

#         for experience in experiences:

#             strategy = (
#                 self.learn_from_experience(
#                     experience
#                 )
#             )

#             if strategy is not None:
#                 learned.append(strategy)

#         return learned

from typing import TYPE_CHECKING, List, Optional

from agent.core.experience import Experience
from agent.learning.strategy import Strategy

if TYPE_CHECKING:
    from agent.learning.strategy_store import StrategyStore


class StrategyLearning:
    def __init__(
        self,
        strategy_store: "StrategyStore",
    ):
        self.strategy_store = strategy_store

    def learn_from_experience(
        self,
        experience: Experience,
    ) -> Optional[Strategy]:
        if not experience.episodes:
            return None

        steps = experience.actions

        if not steps:
            return None

        strategy = self.strategy_store.get_or_create(
            goal=experience.goal,
            steps=steps,
        )

        if experience.completed:
            strategy.record_success(
                experience_id=experience.experience_id,
                action_count=experience.action_count,
            )
        elif experience.failed:
            strategy.record_failure(
                experience_id=experience.experience_id,
                action_count=experience.action_count,
            )

        return strategy

    def learn_from_experiences(
        self,
        experiences: List[Experience],
    ) -> List[Strategy]:
        learned: List[Strategy] = []

        for experience in experiences:
            strategy = self.learn_from_experience(experience)

            if strategy is not None:
                learned.append(strategy)

        return learned
