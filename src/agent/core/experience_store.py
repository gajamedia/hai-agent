from typing import List, Optional

from agent.core.experience import Experience


class ExperienceStore:

    def __init__(self):
        self.experiences: List[Experience] = []

    def add(
        self,
        experience: Experience,
    ) -> None:

        self.experiences.append(
            experience
        )

    def all(self) -> List[Experience]:
        return list(self.experiences)

    def latest(
        self,
    ) -> Optional[Experience]:

        if not self.experiences:
            return None

        return self.experiences[-1]

    def count(self) -> int:
        return len(self.experiences)

    def successful(self) -> List[Experience]:

        return [
            experience
            for experience in self.experiences
            if experience.completed
        ]

    def failed(self) -> List[Experience]:

        return [
            experience
            for experience in self.experiences
            if experience.failed
        ]

    def clear(self) -> None:
        self.experiences.clear()
