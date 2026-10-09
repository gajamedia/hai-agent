from typing import Any, Dict

from .environment import Environment


class RoomEnvironment(Environment):

    def __init__(
        self,
        spare_key_available: bool = True,
    ):
        super().__init__()

        self.spare_key_available = (
            spare_key_available
        )

        self.reset()

    def reset(self) -> None:

        self.state = {
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": (
                "available"
                if self.spare_key_available
                else "unavailable"
            ),
            "agent_location": "room",
        }

    def observe(self) -> Dict[str, Any]:
        return self.state.copy()
    
    def execute(
        self,
        action: str,
    ) -> Dict[str, Any]:

        if action == "open_box":

            if self.state["box"] == "closed":
                self.state["box"] = "open"

                return {
                    "success": True,
                    "message": (
                        "The box is now open."
                    ),
                }

            return {
                "success": False,
                "message": (
                    "The box is already open."
                ),
            }

        if action == "take_key":

            if self.state["box"] != "open":
                return {
                    "success": False,
                    "message": (
                        "The box is closed."
                    ),
                }

            if self.state["key"] != "inside_box":
                return {
                    "success": False,
                    "message": (
                        "The key is not inside "
                        "the box."
                    ),
                }

            self.state["key"] = "agent"

            return {
                "success": True,
                "message": (
                    "The agent picked up "
                    "the key."
                ),
            }

        if action == "find_spare_key":

            if (
                self.state["spare_key"]
                != "available"
            ):
                return {
                    "success": False,
                    "message": (
                        "The spare key is "
                        "not available."
                    ),
                }

            self.state["spare_key"] = "found"
            self.state["key"] = "agent"

            return {
                "success": True,
                "message": (
                    "The agent found "
                    "the spare key."
                ),
            }

        if action == "unlock_door":

            if self.state["key"] != "agent":
                return {
                    "success": False,
                    "message": (
                        "The agent does not "
                        "have the key."
                    ),
                }

            if self.state["door"] != "locked":
                return {
                    "success": False,
                    "message": (
                        "The door is already "
                        "unlocked."
                    ),
                }

            self.state["door"] = "unlocked"

            return {
                "success": True,
                "message": (
                    "The door is now unlocked."
                ),
            }

        return {
            "success": False,
            "message": (
                f"Unknown action: {action}"
            ),
        }
