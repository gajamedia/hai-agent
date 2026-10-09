from typing import Any, Dict, Optional

from agent.memory.episodic import EpisodicMemory
from agent.memory.episodic_store import EpisodicMemoryStore


class EpisodeRecorder:

    def __init__(self, store: EpisodicMemoryStore):
        self.store = store

    def record(
        self,
        goal: Optional[str],
        action: str,
        observation_before: Dict[str, Any],
        observation_after: Dict[str, Any],
        result: Dict[str, Any],
        cycle: int,
    ) -> EpisodicMemory:

        success = result.get("success", False)

        memory = EpisodicMemory(
            content=f"Agent executed action '{action}'.",

            goal=goal,

            action=action,

            observation_before=observation_before.copy(),

            observation_after=observation_after.copy(),

            result=result.copy(),

            cycle=cycle,

            importance=1.0 if success else 0.8,

            confidence=1.0,

            source="agent",

            tags=[
                "experience",
                action,
                "success" if success else "failure",
            ],
        )

        self.store.add(memory)

        return memory
