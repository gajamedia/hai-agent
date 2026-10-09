from agent.core.experience import Experience
from agent.learning.strategy import Strategy
from agent.memory.episodic import EpisodicMemory
from agent.self.learning import SelfLearning
from agent.self.model import SelfModel


def create_experience():

    experience = Experience(
        experience_id="experience-1",
        goal="Open the door",
    )

    experience.add_episode(
        EpisodicMemory(
            content="Opened box.",
            action="open_box",
            observation_before={
                "box": "closed",
                "door": "locked",
            },
            observation_after={
                "box": "open",
                "door": "locked",
            },
            result={
                "success": True,
            },
            cycle=1,
        )
    )

    experience.add_episode(
        EpisodicMemory(
            content="Unlocked door.",
            action="unlock_door",
            observation_before={
                "box": "open",
                "door": "locked",
            },
            observation_after={
                "box": "open",
                "door": "unlocked",
            },
            result={
                "success": True,
            },
            cycle=2,
        )
    )

    experience.complete()

    return experience


def test_self_learning():

    model = SelfModel()

    learning = SelfLearning(
        self_model=model
    )

    experience = create_experience()

    learning.learn_from_experience(
        experience
    )

    open_box = (
        model.get_capability(
            "open_box"
        )
    )

    unlock_door = (
        model.get_capability(
            "unlock_door"
        )
    )

    assert open_box is not None
    assert unlock_door is not None

    assert open_box.success_count == 1
    assert unlock_door.success_count == 1

    assert (
        open_box.confidence == 0.6
    )

    assert (
        unlock_door.confidence == 0.6
    )

    assert (
        "Achieved goal: Open the door"
        in model.known_successes
    )

    assert (
        model.current_condition[
            "box"
        ]
        == "closed"
    )


def test_self_learning_strategy():

    model = SelfModel()

    learning = SelfLearning(
        self_model=model
    )

    strategy = Strategy(
        name="box_strategy",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    learning.learn_from_strategy(
        strategy
    )

    assert (
        model.learned_strategies[
            "box_strategy"
        ]
        is strategy
    )
