from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.agent_decision import (
    AgentDecision,
    DecisionMode,
)
from agent.cognition.active_explorer import ActiveExplorer
from agent.cognition.unified_executor import UnifiedDecisionExecutor
from agent.learning.strategy import Strategy
from agent.world.room import RoomEnvironment


def create_agent():
    agent = Agent(name="closed-loop-strategy-agent")

    agent.register_actions(
        [
            Action(
                name="open_box",
                description="Open box",
                handler=lambda **kwargs: {"success": True},
                preconditions={"box": "closed"},
                effects={"box": "open"},
            ),
            Action(
                name="take_key",
                description="Take key",
                handler=lambda **kwargs: {"success": True},
                preconditions={
                    "box": "open",
                    "key": "inside_box",
                },
                effects={"key": "agent"},
            ),
            Action(
                name="unlock_door",
                description="Unlock door",
                handler=lambda **kwargs: {"success": True},
                preconditions={
                    "key": "agent",
                    "door": "locked",
                },
                effects={"door": "unlocked"},
            ),
        ]
    )

    return agent


def create_decision():
    return AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason="Closed-loop strategy integration test.",
        uncertainty=0.1,
        expected_information_gain=0.0,
        risk=0.1,
    )


def test_real_failure_updates_strategy_before_next_planning_cycle():
    agent = create_agent()
    environment = RoomEnvironment()

    goal = Goal(
        "unlock the door",
        target_state={"door": "unlocked"},
    )

    risky = Strategy(
        name="direct_unlock",
        goal="unlock the door",
        steps=["unlock_door"],
    )

    alternative = Strategy(
        name="open_box_then_unlock",
        goal="unlock the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    # Establish historical evidence before the new failure.
    for index in range(5):
        risky.record_success(
            experience_id=f"direct-success-{index}",
            action_count=1,
            progress_delta=1.0,
        )

    for index in range(4):
        alternative.record_success(
            experience_id=f"alternative-success-{index}",
            action_count=3,
            progress_delta=1.0,
        )

    agent.register_strategies([risky, alternative])

    selector = agent.risk_strategy_selector

    selected_before = selector.select(
        [risky, alternative]
    )

    assert selected_before is risky

    previous_failure_count = risky.failure_count
    previous_episode_count = agent.episodic_memory.count()

    executor = UnifiedDecisionExecutor(
        explorer=ActiveExplorer(),
    )

    # Attempt the action while its preconditions are unsatisfied.
    result = executor.execute_action(
        decision=create_decision(),
        action=agent.actions["unlock_door"],
        agent=agent,
        environment=environment,
        goal=goal,
        experience_id="closed-loop-real-failure",
    )

    assert result.success is False
    assert (
        result.direct_result["failure_type"]
        == "precondition_failed"
    )

    # The real execution must produce an episode and update
    # the strategy attributed to that failed experience.
    assert (
        agent.episodic_memory.count()
        == previous_episode_count + 1
    )

    episode = agent.episodic_memory.all()[-1]

    assert episode.action == "unlock_door"
    assert episode.result["success"] is False

    assert risky.failure_count == previous_failure_count + 1

    # The next selection must use the updated outcome history.
    selected_after = selector.select(
        [risky, alternative]
    )

    assert selected_after is alternative

    # Verify the next plan can use the alternative strategy.
    world_state = environment.observe()

    plan = agent.choose_plan(
        world_state=world_state,
        goal_state={"door": "unlocked"},
        goal="unlock the door",
        strategies=agent.get_available_strategies(),
    )

    assert [action.name for action in plan] == [
        "open_box",
        "take_key",
        "unlock_door",
    ]