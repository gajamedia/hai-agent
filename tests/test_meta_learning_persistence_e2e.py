from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.agent_decision import DecisionMode


class SimpleEnvironment:
    """
    Minimal environment for an end-to-end agent test.

    The agent must discover that executing
    'unlock_door' changes the world into the target state.
    """

    def __init__(self):
        self.state = {
            "door": "locked",
        }

    def observe(self):
        return self.state.copy()

    def execute(self, action):
        if action == "unlock_door":

            if self.state["door"] != "locked":
                return {
                    "success": False,
                    "reason": "door already unlocked",
                }

            self.state["door"] = "unlocked"

            return {
                "success": True,
                "action": action,
            }

        return {
            "success": False,
            "reason": "unknown action",
        }


def create_unlock_action():
    return Action(
        name="unlock_door",
        description="Unlock the door.",
        handler=lambda **kwargs: {
            "success": True,
        },
    )


def create_goal():
    return Goal(
        description="unlock_door",
        target_state={
            "door": "unlocked",
        },
    )


def configure_agent(agent):
    agent.register_action(
        create_unlock_action()
    )


def test_decide_and_act_creates_meta_learning_event(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-e2e",
        memory_path=database,
    )

    configure_agent(agent)

    environment = SimpleEnvironment()
    goal = create_goal()

    result = agent.decide_and_act(
        goal=goal,
        environment=environment,
        information_gain={
            "unlock_door": 1.0,
        },
        action_risk={
            "unlock_door": 0.1,
        },
        risk=0.1,
    )

    assert result is not None

    assert (
        result.decision.mode
        == DecisionMode.EXPLORE
    )

    assert (
        environment.observe()[
            "door"
        ]
        == "unlocked"
    )

    assert (
        agent.meta_learning_profile
        .exploration_count
        == 1
    )

    assert (
        agent.meta_learning_profile
        .exploration_success_count
        == 1
    )

    assert (
        agent.meta_learning_profile
        .total_learning_events
        >= 1
    )


def test_decide_and_act_persists_meta_learning_event(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-e2e-persist",
        memory_path=database,
    )

    configure_agent(agent)

    environment = SimpleEnvironment()

    agent.decide_and_act(
        goal=create_goal(),
        environment=environment,
        information_gain={
            "unlock_door": 1.0,
        },
        action_risk={
            "unlock_door": 0.1,
        },
        risk=0.1,
    )

    persisted = (
        agent.meta_learning_repository.get(
            "agent-e2e-persist"
        )
    )

    assert persisted is not None

    assert (
        persisted.exploration_count
        == 1
    )

    assert (
        persisted.exploration_success_count
        == 1
    )


def test_meta_learning_survives_end_to_end_restart(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    # -------------------------------------------------
    # First process
    # -------------------------------------------------

    first = Agent(
        name="agent-e2e-restart",
        memory_path=database,
    )

    configure_agent(first)

    first_environment = (
        SimpleEnvironment()
    )

    first.decide_and_act(
        goal=create_goal(),
        environment=first_environment,
        information_gain={
            "unlock_door": 1.0,
        },
        action_risk={
            "unlock_door": 0.1,
        },
        risk=0.1,
    )

    assert (
        first.meta_learning_profile
        .exploration_count
        == 1
    )

    # -------------------------------------------------
    # Second process
    # -------------------------------------------------

    second = Agent(
        name="agent-e2e-restart",
        memory_path=database,
    )

    assert (
        second.meta_learning_profile
        .exploration_count
        == 1
    )

    assert (
        second.meta_learning_profile
        .exploration_success_count
        == 1
    )

    assert (
        second.meta_learning_profile
        .total_learning_events
        >= 1
    )


def test_second_end_to_end_decision_adds_new_meta_learning_event(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    # -------------------------------------------------
    # First process
    # -------------------------------------------------

    first = Agent(
        name="agent-e2e-continuation",
        memory_path=database,
    )

    configure_agent(first)

    first.decide_and_act(
        goal=create_goal(),
        environment=SimpleEnvironment(),
        information_gain={
            "unlock_door": 1.0,
        },
        action_risk={
            "unlock_door": 0.1,
        },
        risk=0.1,
    )

    # -------------------------------------------------
    # Restart
    # -------------------------------------------------

    second = Agent(
        name="agent-e2e-continuation",
        memory_path=database,
    )

    configure_agent(second)

    previous_count = (
        second.meta_learning_profile
        .exploration_count
    )

    # -------------------------------------------------
    # Second real decision
    # -------------------------------------------------

    second.decide_and_act(
        goal=create_goal(),
        environment=SimpleEnvironment(),
        information_gain={
            "unlock_door": 1.0,
        },
        action_risk={
            "unlock_door": 0.1,
        },
        risk=0.1,
    )

    assert (
        second.meta_learning_profile
        .exploration_count
        == previous_count + 1
    )

    assert (
        second.meta_learning_profile
        .exploration_success_count
        == 2
    )

    # -------------------------------------------------
    # Verify database state
    # -------------------------------------------------

    persisted = (
        second.meta_learning_repository.get(
            "agent-e2e-continuation"
        )
    )

    assert persisted is not None

    assert (
        persisted.exploration_count
        == 2
    )

    assert (
        persisted.exploration_success_count
        == 2
    )


def test_end_to_end_experience_and_meta_learning_are_both_created(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-e2e-experience",
        memory_path=database,
    )

    configure_agent(agent)

    environment = SimpleEnvironment()

    agent.decide_and_act(
        goal=create_goal(),
        environment=environment,
        information_gain={
            "unlock_door": 1.0,
        },
        action_risk={
            "unlock_door": 0.1,
        },
        risk=0.1,
    )

    assert (
        agent.meta_learning_profile
        .exploration_count
        == 1
    )

    assert (
        agent.meta_learning_profile
        .exploration_success_count
        == 1
    )

    # The actual action must also have produced
    # a regular episodic experience.
    assert (
        agent.episodic_memory.count()
        >= 1
    )


def test_end_to_end_goal_completion_matches_meta_success(
    tmp_path,
):
    database = (
        tmp_path
        / "agent.db"
    )

    agent = Agent(
        name="agent-e2e-goal",
        memory_path=database,
    )

    configure_agent(agent)

    environment = SimpleEnvironment()
    goal = create_goal()

    agent.decide_and_act(
        goal=goal,
        environment=environment,
        information_gain={
            "unlock_door": 1.0,
        },
        action_risk={
            "unlock_door": 0.1,
        },
        risk=0.1,
    )

    assert goal.completed is True

    assert (
        agent.meta_learning_profile
        .exploration_success_count
        == 1
    )
