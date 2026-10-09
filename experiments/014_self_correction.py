from agent.core.agent import Agent
from agent.core.experience import Experience

from agent.cognition.adaptive_planner import (
    AdaptivePlanner,
)
from agent.cognition.planner import Planner

from agent.learning.reflection import (
    ReflectionEngine,
)
from agent.learning.reflection_strategy_updater import (
    ReflectionStrategyUpdater,
)

from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import (
    StrategySelector,
)

from agent.memory.reflection_store import (
    ReflectionMemoryStore,
)

from agent.tools.action_system import ActionSystem
from agent.tools.room_actions import (
    create_room_actions,
)

from agent.world.model import WorldModel
from agent.world.room import RoomEnvironment


def build_world_state(
    environment: RoomEnvironment,
):

    world_model = WorldModel()

    observation = environment.observe()

    world_model.update(
        observation
    )

    return {
        "door": world_model.get("door"),
        "box": world_model.get("box"),
        "key": world_model.get("key"),
        "spare_key": world_model.get(
            "spare_key"
        ),
        "agent_location": world_model.get(
            "agent_location"
        ),
    }


def execute_attempt(
    agent,
    environment,
    action_system,
    plan,
    goal,
    experience_id,
):

    experience = Experience(
        experience_id=experience_id,
        goal=goal,
    )

    agent.set_plan(
        [
            action.name
            for action in plan
        ]
    )

    failed_action = None
    failed_episode = None

    print("Executing plan:")

    for index, action in enumerate(
        plan,
        start=1,
    ):

        print(
            f"  {index}. {action.name}"
        )

        observation_before = (
            environment.observe()
        )

        agent.act(
            action.name
        )

        result = action_system.execute(
            action.name
        )

        observation_after = (
            environment.observe()
        )

        episode = agent.record_experience(
            action=action.name,
            observation_before=(
                observation_before
            ),
            observation_after=(
                observation_after
            ),
            result=result,
        )

        experience.add_episode(
            episode
        )

        agent.observe(
            observation_after
        )

        agent.step()

        print(
            "     Success:",
            result.get("success"),
        )

        if not result.get(
            "success",
            False,
        ):

            failed_action = action
            failed_episode = episode

            break

    final_world = environment.observe()

    if (
        failed_action is None
        and final_world.get("door")
        == "unlocked"
    ):

        experience.complete()

    else:

        experience.fail()

    return (
        experience,
        failed_action,
        failed_episode,
    )


def print_strategy(
    strategy,
    selector,
    label,
):

    print(label)

    print(
        "  Steps:",
        strategy.steps,
    )

    print(
        "  Attempts:",
        strategy.attempts,
    )

    print(
        "  Success:",
        strategy.success_count,
    )

    print(
        "  Failure:",
        strategy.failure_count,
    )

    print(
        "  Success rate:",
        strategy.success_rate,
    )

    print(
        "  Average actions:",
        strategy.average_actions,
    )

    print(
        "  Reflection penalty:",
        strategy.reflection_penalty,
    )

    print(
        "  Score:",
        selector._score(
            strategy
        ),
    )

    print()


def main():

    print("================================")
    print("SELF-CORRECTION EXPERIMENT")
    print("================================")
    print()

    agent = Agent(
        "Agent-001"
    )

    environment = RoomEnvironment(
        spare_key_available=True
    )

    action_system = ActionSystem()

    actions = create_room_actions(
        environment
    )

    for action in actions:
        action_system.register(
            action
        )

    classical_planner = Planner(
        actions
    )

    selector = StrategySelector()

    adaptive_planner = AdaptivePlanner(
        planner=classical_planner,
        strategy_selector=selector,
    )

    reflection_store = (
        ReflectionMemoryStore()
    )

    reflection_engine = ReflectionEngine(
        reflection_store
    )

    strategy_updater = (
        ReflectionStrategyUpdater()
    )

    goal = "Open the door"

    # ==================================================
    # LEARNED STRATEGIES
    # ==================================================

    # Strategy A:
    # reliable but requires 3 actions.
    strategy_a = Strategy(
        name="box_strategy",
        goal=goal,
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=4,
        failure_count=4,
        total_actions=24,
    )

    # Strategy B:
    # currently considered better.
    strategy_b = Strategy(
        name="spare_key_strategy",
        goal=goal,
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
        success_count=5,
        failure_count=1,
        total_actions=12,
    )

    strategies = [
        strategy_a,
        strategy_b,
    ]

    # ==================================================
    # BEFORE SELF-CORRECTION
    # ==================================================

    print("================================")
    print("BEFORE SELF-CORRECTION")
    print("================================")
    print()

    print_strategy(
        strategy_a,
        selector,
        "Strategy A",
    )

    print_strategy(
        strategy_b,
        selector,
        "Strategy B",
    )

    selected_before = selector.select(
        strategies,
        goal=goal,
    )

    print(
        "Initially selected strategy:",
        selected_before.name,
    )

    print(
        "Steps:",
        selected_before.steps,
    )

    print()

    # ==================================================
    # ATTEMPT #1
    # ==================================================

    print("================================")
    print("ATTEMPT #1")
    print("================================")
    print()

    environment.reset()

    # World state at planning time:
    #
    # spare_key = available
    #
    # Therefore Strategy B is executable.

    world_state = build_world_state(
        environment
    )

    plan = adaptive_planner.plan(
        world_state=world_state,
        goal_state={
            "door": "unlocked",
        },
        goal=goal,
        strategies=strategies,
    )

    print("Planner selected:")

    for index, action in enumerate(
        plan,
        start=1,
    ):

        print(
            f"  {index}. {action.name}"
        )

    print()

    # ==================================================
    # UNEXPECTED WORLD CHANGE
    # ==================================================

    print(
        "Unexpected world change:"
    )

    print(
        "  spare_key:",
        environment.state["spare_key"],
        "-> unavailable",
    )

    # World changed AFTER planning.
    environment.state["spare_key"] = (
        "unavailable"
    )

    print()

    # ==================================================
    # EXECUTE ATTEMPT #1
    # ==================================================

    (
        experience_1,
        failed_action,
        failed_episode,
    ) = execute_attempt(
        agent=agent,
        environment=environment,
        action_system=action_system,
        plan=plan,
        goal=goal,
        experience_id="experience-1",
    )

    print()

    print(
        "Attempt #1 completed:",
        experience_1.completed,
    )

    print(
        "Attempt #1 failed:",
        experience_1.failed,
    )

    print(
        "Actions executed:",
        experience_1.actions,
    )

    print()

    # ==================================================
    # REFLECTION
    # ==================================================

    print("================================")
    print("REFLECTION")
    print("================================")
    print()

    reflection = None

    if (
        failed_action is not None
        and failed_episode is not None
    ):

        reflection = (
            reflection_engine.reflect(
                episode=failed_episode,
                action=failed_action,
            )
        )

    if reflection is None:

        print(
            "No reflection generated."
        )

        return

    print(
        "Failed action:",
        reflection.action,
    )

    print(
        "Problem:",
        reflection.problem,
    )

    print(
        "Cause:",
        reflection.cause,
    )

    print(
        "Condition:",
        reflection.condition,
    )

    print(
        "Lesson:",
        reflection.lesson,
    )

    print(
        "Confidence:",
        reflection.confidence,
    )

    print()

    # ==================================================
    # APPLY REFLECTION
    # ==================================================

    print("================================")
    print("STRATEGY UPDATE")
    print("================================")
    print()

    strategy_updater.apply(
        strategy=strategy_b,
        reflection=reflection,
        penalty=0.4,
    )

    print(
        "Reflection applied:",
        reflection.applied,
    )

    print()

    print_strategy(
        strategy_a,
        selector,
        "Strategy A",
    )

    print_strategy(
        strategy_b,
        selector,
        "Strategy B",
    )

    # ==================================================
    # REPLAN
    # ==================================================

    print("================================")
    print("REPLANNING")
    print("================================")
    print()

    replanning_world = (
        build_world_state(
            environment
        )
    )

    print(
        "Current world state:"
    )

    print(
        " ",
        replanning_world,
    )

    print()

    replanned = adaptive_planner.plan(
        world_state=replanning_world,
        goal_state={
            "door": "unlocked",
        },
        goal=goal,
        strategies=strategies,
    )

    print(
        "Replanned actions:"
    )

    for index, action in enumerate(
        replanned,
        start=1,
    ):

        print(
            f"  {index}. {action.name}"
        )

    print()

    # ==================================================
    # ATTEMPT #2
    # ==================================================

    print("================================")
    print("ATTEMPT #2")
    print("================================")
    print()

    (
        experience_2,
        failed_action_2,
        failed_episode_2,
    ) = execute_attempt(
        agent=agent,
        environment=environment,
        action_system=action_system,
        plan=replanned,
        goal=goal,
        experience_id="experience-2",
    )

    print()

    print(
        "Attempt #2 completed:",
        experience_2.completed,
    )

    print(
        "Attempt #2 failed:",
        experience_2.failed,
    )

    print(
        "Actions executed:",
        experience_2.actions,
    )

    print()

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("================================")
    print("FINAL SUMMARY")
    print("================================")
    print()

    print(
        "Reflection memories:",
        reflection_store.count(),
    )

    print(
        "Strategy B penalty:",
        strategy_b.reflection_penalty,
    )

    print(
        "Attempt #1 success:",
        experience_1.completed,
    )

    print(
        "Attempt #2 success:",
        experience_2.completed,
    )

    print(
        "Final world:",
        environment.observe(),
    )


if __name__ == "__main__":
    main()
