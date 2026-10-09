from typing import Optional

from agent.learning.context_extractor import (
    ContextExtractor,
)

from agent.learning.contradiction import (
    Contradiction,
)

from agent.memory.episodic import (
    EpisodicMemory,
)

from agent.memory.semantic import (
    SemanticMemory,
)


class ContradictionDetector:

    def __init__(self):
        self.context_extractor = (
            ContextExtractor()
        )

    def detect_semantic(
        self,
        memory: SemanticMemory,
        episode: EpisodicMemory,
    ) -> Optional[Contradiction]:
        """
        Detect whether an episode contradicts
        an existing semantic memory.

        A failed action is NOT a contradiction.

        A successful observation can be a contradiction
        when it occurs in the same context and produces
        an unexpected outcome.
        """

        # ----------------------------------------------
        # No result -> cannot evaluate
        # ----------------------------------------------

        if episode.result is None:
            return None

        # ----------------------------------------------
        # Failure is not automatically contradiction
        # ----------------------------------------------

        if not episode.result.get(
            "success",
            False,
        ):
            return None

        # ----------------------------------------------
        # Need observations
        # ----------------------------------------------

        if (
            episode.observation_before is None
            or episode.observation_after is None
        ):
            return None

        # ----------------------------------------------
        # Only state-transition knowledge is handled
        # here.
        # ----------------------------------------------

        if memory.predicate != "changes_from":
            return None

        # ----------------------------------------------
        # Context evaluation
        #
        # An empty memory context means the knowledge
        # is general and therefore applies to any context.
        # A non-empty memory context must match.
        # ----------------------------------------------

        context = (
            self.context_extractor.extract(
                episode
            )
        )

        if memory.context:

            if not memory.matches_context(
                context
            ):
                return None

        # ----------------------------------------------
        # Validate semantic object
        # ----------------------------------------------

        expected = memory.object

        if not isinstance(
            expected,
            dict,
        ):
            return None

        key = memory.subject

        # ----------------------------------------------
        # Actual observation
        # ----------------------------------------------

        before = (
            episode.observation_before.get(
                key
            )
        )

        after = (
            episode.observation_after.get(
                key
            )
        )

        # ----------------------------------------------
        # Expected transition
        # ----------------------------------------------

        expected_before = expected.get(
            "before"
        )

        expected_after = expected.get(
            "after"
        )

        # ----------------------------------------------
        # Different starting condition means this
        # experience is not evidence about this
        # particular transition.
        # ----------------------------------------------

        if before != expected_before:
            return None

        # ----------------------------------------------
        # Expected outcome occurred -> consistent
        # ----------------------------------------------

        if after == expected_after:
            return None

        # ----------------------------------------------
        # Same context + different outcome
        # -> contradiction
        # ----------------------------------------------

        return Contradiction(
            subject=memory.subject,
            predicate=memory.predicate,
            expected_object={
                "before": expected_before,
                "after": expected_after,
            },
            observed_object={
                "before": before,
                "after": after,
            },
            source_episode_id=(
                episode.memory_id
            ),
            reason=(
                f"Expected {key} to change "
                f"from {expected_before!r} "
                f"to {expected_after!r}, "
                f"but observed "
                f"{before!r} -> {after!r}."
            ),
        )
