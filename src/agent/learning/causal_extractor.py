from typing import Dict, List

from agent.core.action import Action
from agent.memory.episodic import EpisodicMemory


class CausalExtractor:
    """
    Infers causal relationships between consecutive actions.

    If:

        Action A changes:
            state X -> value Y

        Action B requires:
            X == Y

    Then:

        A ENABLES B
    """

    def extract(
        self,
        previous_episode: EpisodicMemory,
        next_episode: EpisodicMemory,
        actions: Dict[str, Action],
    ) -> List[dict]:

        if previous_episode.result is None:
            return []

        if not previous_episode.result.get("success", False):
            return []

        if next_episode.result is None:
            return []

        if not next_episode.result.get(
            "success",
            False,
        ):
            return []

        previous_action = actions.get(
            previous_episode.action
        )

        next_action = actions.get(
            next_episode.action
        )

        if previous_action is None:
            return []

        if next_action is None:
            return []

        before = previous_episode.observation_before
        after = previous_episode.observation_after

        if before is None or after is None:
            return []

        causal_relationships = []

        for key in set(before.keys()) | set(after.keys()):

            before_value = before.get(key)
            after_value = after.get(key)

            if before_value == after_value:
                continue

            required_value = next_action.preconditions.get(
                key
            )

            if required_value != after_value:
                continue

            causal_relationships.append(
                {
                    "cause": previous_action.name,
                    "relation": "enables",
                    "effect": next_action.name,
                    "condition": {
                        "key": key,
                        "value": after_value,
                    },
                }
            )

        return causal_relationships
