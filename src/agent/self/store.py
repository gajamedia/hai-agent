from typing import Optional

from agent.self.model import SelfModel


class SelfModelStore:

    def __init__(
        self,
        repository=None,
    ):
        self.repository = repository

        if self.repository is not None:
            loaded = self.repository.get()

            self.model = (
                loaded
                if loaded is not None
                else SelfModel()
            )
        else:
            self.model = SelfModel()

    def get(self) -> SelfModel:
        return self.model

    def reset(self) -> None:
        self.model = SelfModel()

        self.save()

    def save(self) -> None:
        if self.repository is None:
            return

        self.repository.save(
            self.model
        )

    def load(self) -> None:
        if self.repository is None:
            return

        loaded = self.repository.get()

        if loaded is not None:
            self.model = loaded
