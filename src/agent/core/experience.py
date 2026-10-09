from dataclasses import dataclass, field
from typing import List, Optional

from agent.memory.episodic import EpisodicMemory


@dataclass
class Experience:
    """
    Represents one complete attempt by the agent
    to achieve a goal.

    An experience contains multiple episodes/actions.
    """

    experience_id: str
    goal: str

    episodes: List[EpisodicMemory] = field(
        default_factory=list
    )

    completed: bool = False
    failed: bool = False

    cycle_start: Optional[int] = None
    cycle_end: Optional[int] = None

    def add_episode(
        self,
        episode: EpisodicMemory,
    ) -> None:

        if not self.episodes:
            self.cycle_start = episode.cycle

        self.episodes.append(episode)

        self.cycle_end = episode.cycle

    def complete(self) -> None:
        self.completed = True
        self.failed = False

    def fail(self) -> None:
        self.failed = True
        self.completed = False

    @property
    def action_count(self) -> int:
        return len(self.episodes)

    @property
    def actions(self) -> List[str]:
        return [
            episode.action
            for episode in self.episodes
            if episode.action is not None
        ]
