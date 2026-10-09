from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class Action:
    name: str
    description: str
    handler: Callable[..., Dict[str, Any]]

    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)

    def execute(self, **kwargs) -> Dict[str, Any]:
        return self.handler(**kwargs)

    def can_execute(self, state: Dict[str, Any]) -> bool:
        for key, expected_value in self.preconditions.items():
            if state.get(key) != expected_value:
                return False

        return True
