from agent.core.agent import Agent
from agent.core.experience import Experience
from agent.core.experience_store import ExperienceStore
from agent.core.goal import Goal

from agent.cognition.adaptive_planner import (
    AdaptivePlanner,
)
from agent.cognition.planner import Planner

from agent.learning.strategy_selector import (
    StrategySelector,
)
from agent.learning.strategy_store import (
    StrategyStore,
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
    }


def execute_plan(
    agent,
    environment,
    action_system,
    plan,
    goal,
    experience_id,
):

    experience = Experience(
        experience_id=experience_id,
        goal=goal.description,
    )

    agent.set_plan(
        [
            action.name
            for action in plan
        ]
    )

    for action in plan:

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
            f"  Action: {action.name}"
        )

        print(
            "  Success:",
            result.get("success"),
        )

        if not result.get(
            "success",
            False,
        ):

            experience.fail()

            return experience

    final_world = environment.observe()

    if (
        final_world.get("door")
        == "unlocked"
    ):

        experience.complete()

    else:

        experience.fail()

    return experience


def print_strategy(
    strategy,
    selector,
    index,
):

    print(
        f"Strategy #{index}"
    )

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
        "  Score:",
        selector._score(strategy),
    )

    print()


def main():

    print("================================")
    print("ADAPTIVE PLANNER")
    print("================================")
    print()

    agent = Agent(
        "Agent-001"
    )

    environment = RoomEnvironment()

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

    adaptive_planner = AdaptivePlanner(
        planner=classical_planner,
        strategy_selector=(
            StrategySelector()
        ),
    )

    strategy_store = StrategyStore()

    experience_store = ExperienceStore()

    goal_text = "Open the door"

    # ==================================================
    # PHASE 1
    # ==================================================

    print("================================")
    print("PHASE 1 - BEFORE LEARNING")
    print("================================")
    print()

    environment.reset()

    goal = Goal(
        description=goal_text,
        priority=10,
    )

    agent.set_goal(
        goal
    )

    world_state = build_world_state(
        environment
    )

    plan = adaptive_planner.plan(
        world_state=world_state,
        goal_state={
            "door": "unlocked",
        },
        goal=goal.description,
        strategies=(
            strategy_store.for_goal(
                goal.description
            )
        ),
    )

    print("Selected plan:")

    for index, action in enumerate(
        plan,
        start=1,
    ):
        print(
            f"{index}. {action.name}"
        )

    print()

    experience = execute_plan(
        agent=agent,
        environment=environment,
        action_system=action_system,
        plan=plan,
        goal=goal,
        experience_id="experience-1",
    )

    experience_store.add(
        experience
    )

    print()

    print(
        "Goal completed:",
        experience.completed,
    )

    print(
        "Actions used:",
        experience.actions,
    )

    print()

    # ==================================================
    # LEARN STRATEGY A
    # ==================================================

    strategy_a = strategy_store.get_or_create(
        goal=experience.goal,
        steps=experience.actions,
    )

    if experience.completed:

        strategy_a.record_success(
            experience_id=(
                experience.experience_id
            ),
            action_count=(
                experience.action_count
            ),
        )

    else:

        strategy_a.record_failure(
            experience_id=(
                experience.experience_id
            ),
            action_count=(
                experience.action_count
            ),
        )

    # Add failure evidence.
    strategy_a.record_failure(
        experience_id="strategy-a-failure",
        action_count=3,
    )

    # ==================================================
    # PHASE 2
    # Introduce a second learned strategy.
    # ==================================================

    print("================================")
    print("PHASE 2 - LEARN ALTERNATIVE")
    print("================================")
    print()

    strategy_b = strategy_store.get_or_create(
        goal=goal_text,
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
    )

    for number in range(
        1,
        4,
    ):

        strategy_b.record_success(
            experience_id=(
                f"strategy-b-exp-{number}"
            ),
            action_count=2,
        )

    strategies = strategy_store.for_goal(
        goal_text
    )

    selector = adaptive_planner.strategy_selector

    for index, strategy in enumerate(
        strategies,
        start=1,
    ):

        print_strategy(
            strategy= strategy,
            selector=selector,
            index=index,
        )

    # ==================================================
    # PHASE 3
    # Adaptive Planner
    # ==================================================

    print("================================")
    print("PHASE 3 - ADAPTIVE DECISION")
    print("================================")
    print()

    environment.reset()

    goal = Goal(
        description=goal_text,
        priority=10,
    )

    agent.set_goal(
        goal
    )

    world_state = build_world_state(
        environment
    )

    selected_plan = adaptive_planner.plan(
        world_state=world_state,
        goal_state={
            "door": "unlocked",
        },
        goal=goal.description,
        strategies=strategies,
    )

    print("Adaptive Planner selected:")

    for index, action in enumerate(
        selected_plan,
        start=1,
    ):

        print(
            f"{index}. {action.name}"
        )

    print()

    # ==================================================
    # PHASE 4
    # Execute learned strategy.
    # ==================================================

    print("================================")
    print("PHASE 4 - EXECUTE LEARNED STRATEGY")
    print("================================")
    print()

    experience = execute_plan(
        agent=agent,
        environment=environment,
        action_system=action_system,
        plan=selected_plan,
        goal=goal,
        experience_id="experience-adaptive",
    )

    experience_store.add(
        experience
    )

    print()

    print(
        "Goal completed:",
        experience.completed,
    )

    print(
        "Actions used:",
        experience.actions,
    )

    print()

    # ==================================================
    # FINAL
    # ==================================================

    print("================================")
    print("FINAL DECISION")
    print("================================")
    print()

    selected_strategy = selector.select(
        strategies=strategies,
        goal=goal_text,
    )

    if selected_strategy is not None:

        print(
            "Selected strategy:",
            selected_strategy.steps,
        )

        print(
            "Success rate:",
            selected_strategy.success_rate,
        )

        print(
            "Average actions:",
            selected_strategy.average_actions,
        )

        print(
            "Score:",
            selector._score(
                selected_strategy
            ),
        )

    print()

    print("================================")
    print("SUMMARY")
    print("================================")

    print(
        "Total experiences:",
        experience_store.count(),
    )

    print(
        "Learned strategies:",
        strategy_store.count(),
    )

    print(
        "Final world:",
        environment.observe(),
    )


if __name__ == "__main__":
    main()
