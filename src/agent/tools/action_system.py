from typing import Any, Dict

from agent.core.action import Action


class ActionSystem:
    def __init__(self):
        self.actions: Dict[str, Action] = {}

    def register(self, action: Action) -> None:
        self.actions[action.name] = action

    def get(self, action_name: str) -> Action | None:
        return self.actions.get(action_name)

    def available_actions(self) -> list[str]:
        return list(self.actions.keys())

    def execute(self, action_name: str, **kwargs) -> Dict[str, Any]:
        action = self.get(action_name)

        if action is None:
            return {
                "success": False,
                "message": f"Unknown action: {action_name}",
            }

        return action.execute(**kwargs)
