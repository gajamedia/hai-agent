from typing import Any, Dict


class Environment:

    def __init__(self):
        self.state: Dict[str, Any] = {}

    def reset(self) -> None:
        self.state = {}

    def observe(self) -> Dict[str, Any]:
        return self.state.copy()

    def execute(self, action: str) -> Dict[str, Any]:
        raise NotImplementedError
