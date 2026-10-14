
from unittest.mock import Mock

from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal

from agent.cognition.agent_decision import (
    AgentDecision,
    DecisionMode,
)
from agent.cognition.active_explorer import ActiveExplorer
from agent.cognition.unified_executor import UnifiedDecisionExecutor
from agent.world.room import RoomEnvironment


def create_agent():
    agent = Agent(name="direct-action-integration-agent")

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
        reason="Test direct-action precondition failure.",
        uncertainty=0.1,
        expected_information_gain=0.0,
        risk=0.1,
    )


def test_direct_action_precondition_failure_records_real_agent_episode():
    agent = create_agent()
    environment = RoomEnvironment()

    executor = UnifiedDecisionExecutor(
        explorer=ActiveExplorer(),
    )

    goal = Goal(
        "unlock the door",
        target_state={"door": "unlocked"},
    )

    state_before = dict(environment.observe())
    episodic_count_before = agent.episodic_memory.count()

    requested_action = agent.actions["unlock_door"]

    result = executor.execute_action(
        decision=create_decision(),
        action=requested_action,
        agent=agent,
        environment=environment,
        goal=goal,
        experience_id="direct-action-precondition-integration",
    )

    assert result.action is requested_action
    assert result.success is False
    assert result.direct_result["failure_type"] == "precondition_failed"
    assert environment.observe() == state_before

    assert (
        agent.episodic_memory.count()
        == episodic_count_before + 1
    )

    episode = agent.episodic_memory.all()[-1]

    assert episode.action == "unlock_door"
    assert episode.result["success"] is False
    assert episode.result["failure_type"] == "precondition_failed"
    assert "failure" in episode.tags
    assert episode.observation_before == state_before
    assert episode.observation_after == state_before

    experience = agent.experience_recorder.store.latest()

    assert experience is not None
    assert (
        experience.experience_id
        == "direct-action-precondition-integration"
    )
    assert experience.failed is True
    assert experience.completed is False
    assert experience.action_count == 1
    assert experience.episodes[0].action == "unlock_door"
    assert agent.current_experience is None


def test_direct_action_failure_invokes_learning_pipeline():
    agent = create_agent()
    environment = RoomEnvironment()

    executor = UnifiedDecisionExecutor(
        explorer=ActiveExplorer(),
    )

    # Wrap the real methods: record calls without bypassing their
    # original implementations.
    semantic_learning = Mock(
        wraps=agent.semantic_learning.learn_from_episode,
    )
    causal_learning = Mock(
        wraps=agent.online_causal_learning.observe_episode,
    )
    metacognitive_reflection = Mock(
        wraps=agent.reflect_on_episode,
    )

    agent.semantic_learning.learn_from_episode = semantic_learning
    agent.online_causal_learning.observe_episode = causal_learning
    agent.reflect_on_episode = metacognitive_reflection

    result = executor.execute_action(
        decision=create_decision(),
        action=agent.actions["unlock_door"],
        agent=agent,
        environment=environment,
        goal=Goal(
            "unlock the door",
            target_state={"door": "unlocked"},
        ),
        experience_id="direct-action-learning-pipeline",
    )

    assert result.success is False

    # The real Agent recording pipeline must invoke all three stages.
    semantic_learning.assert_called_once()
    causal_learning.assert_called_once()
    metacognitive_reflection.assert_called_once()

    episode = agent.episodic_memory.all()[-1]

    assert semantic_learning.call_args.args[0] is episode
    assert causal_learning.call_args.kwargs["episode"] is episode
    assert metacognitive_reflection.call_args.args[0] is episode

    # Reflection must remain accessible after the failed action.
    reflection = agent.get_metacognitive_reflection()
    assert reflection is not None
    assert reflection.success is False

    assert agent.current_experience is None
