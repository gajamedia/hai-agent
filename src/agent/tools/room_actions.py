from agent.core.action import Action
from agent.world.room import RoomEnvironment


def create_room_actions(
    environment: RoomEnvironment,
) -> list[Action]:

    return [
        Action(
            name="open_box",
            description="Open the box",
            handler=lambda: (
                environment.execute(
                    "open_box"
                )
            ),
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),

        Action(
            name="take_key",
            description="Take the key from the box",
            handler=lambda: (
                environment.execute(
                    "take_key"
                )
            ),
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),

        Action(
            name="find_spare_key",
            description="Find the spare key",
            handler=lambda: (
                environment.execute(
                    "find_spare_key"
                )
            ),
            preconditions={
                "spare_key": "available",
            },
            effects={
                "spare_key": "found",
                "key": "agent",
            },
        ),

        Action(
            name="unlock_door",
            description="Unlock the door using the key",
            handler=lambda: (
                environment.execute(
                    "unlock_door"
                )
            ),
            preconditions={
                "door": "locked",
                "key": "agent",
            },
            effects={
                "door": "unlocked",
            },
        ),
    ]
