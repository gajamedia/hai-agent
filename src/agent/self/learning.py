from agent.core.experience import Experience
from agent.learning.strategy import Strategy
from agent.self.capability import Capability
from agent.self.model import SelfModel
from agent.self.assessment import (
    SelfAssessmentEngine,
)


class SelfLearning:

    def __init__(
        self,
        self_model: SelfModel,
    ):
        self.self_model = self_model
        self.assessment_engine = (
            SelfAssessmentEngine()
        )

    def learn_from_experience(
        self,
        experience: Experience,
    ) -> None:

        self._learn_actions(
            experience
        )

        self._update_capability_assessments()

        self._learn_outcome(
            experience
        )

        self._learn_condition(
            experience
        )

    def learn_from_strategy(
        self,
        strategy: Strategy,
    ) -> None:

        self.self_model.register_strategy(
            strategy
        )

        assessment = (
            self.assessment_engine.assess_strategy(
                strategy
            )
        )

        self.self_model.register_assessment(
            key=(
                f"strategy:{strategy.name}"
            ),
            assessment=assessment,
        )

    def _learn_actions(
        self,
        experience: Experience,
    ) -> None:

        for episode in experience.episodes:

            if episode.action is None:
                continue

            capability = (
                self.self_model.get_capability(
                    episode.action
                )
            )

            if capability is None:

                capability = Capability(
                    name=episode.action,
                    confidence=0.5,
                )

                self.self_model.register_capability(
                    episode.action,
                    capability,
                )

            if episode.result is None:
                continue

            if episode.result.get(
                "success",
                False,
            ):

                capability.record_success(
                    source_id=(
                        episode.memory_id
                    )
                )

            else:

                capability.record_failure(
                    source_id=(
                        episode.memory_id)
                    )

    def _learn_outcome(
        self,
        experience: Experience,
    ) -> None:

        if experience.completed:

            self.self_model.record_success(
                description=(
                    f"Achieved goal: "
                    f"{experience.goal}"
                )
            )

        elif experience.failed:

            self.self_model.record_failure(
                description=(
                    f"Failed goal: "
                    f"{experience.goal}"
                )
            )

    def _learn_condition(
        self,
        experience: Experience,
    ) -> None:

        if not experience.episodes:
            return

        first_episode = (
            experience.episodes[0]
        )

        if (
            first_episode.observation_before
            is not None
        ):

            self.self_model.update_condition(
                first_episode.observation_before
            )

    def _update_capability_assessments(
        self,
    ) -> None:

        for (
            name,
            capability,
        ) in self.self_model.capabilities.items():

            assessment = (
                self.assessment_engine.assess_capability(
                    capability
                )
            )

            self.self_model.register_assessment(
                key=f"capability:{name}",
                assessment=assessment,
            )
