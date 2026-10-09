from typing import Any, Dict


class WorldModel:

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.events = []

    def update(self, observation: Dict[str, Any]) -> None:
        for key, value in observation.items():
            self.entities[key] = {
                "value": value
            }

    def get(self, entity: str) -> Any:
        data = self.entities.get(entity)

        if data is None:
            return None

        return data["value"]

    def record_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
