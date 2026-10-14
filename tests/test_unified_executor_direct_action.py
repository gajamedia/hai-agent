
from types import SimpleNamespace

from agent.cognition.agent_decision import DecisionMode
from agent.cognition.unified_executor import (
    UnifiedDecisionExecutor,
)


class FakeAction:
    def __init__(
        self,
        name,
        preconditions=None,
    ):
        self.name = name
        self.preconditions = preconditions or {}

    def can_execute(self, state):
        return all(
            state.get(key) == expected
            for key, expected in self.preconditions.items()
        )


class FakeExplorationExecutor:
    def __init__(self):
        self.executed = []

    def execute(
        self,
        *,
        agent,
        environment,
        action,
    ):
        self.executed.append(action.name)

        return SimpleNamespace(
            action=action.name,
            result={
                "success": True,
                "action": action.name,
            },
            success=True,
        )


class RecordingAgent:
    """Minimal agent double for experience lifecycle tests."""

    def __init__(self, current_experience=None):
        self.name = "test-agent"
        self.state = SimpleNamespace(cycle=0)
        self.current_experience = current_experience
        self.current_goal = None

        self.recorded_episodes = []
        self.lifecycle_calls = []

    def set_goal(self, goal):
        self.current_goal = goal
        self.lifecycle_calls.append("set_goal")

    def start_experience(self, *, experience_id, goal):
        if self.current_experience is not None:
            raise RuntimeError("An experience is already active.")

        self.current_experience = SimpleNamespace(
            experience_id=experience_id,
            goal=goal,
        )
        self.lifecycle_calls.append("start_experience")

    def record_experience(
        self,
        *,
        action,
        observation_before,
        observation_after,
        result,
    ):
        episode = {
            "action": action,
            "observation_before": dict(observation_before),
            "observation_after": dict(observation_after),
            "result": dict(result),
        }
        self.recorded_episodes.append(episode)
        return episode

    def complete_experience(self):
        self.lifecycle_calls.append("complete_experience")

    def fail_experience(self):
        self.lifecycle_calls.append("fail_experience")

    def close_experience(self):
        self.lifecycle_calls.append("close_experience")
        self.current_experience = None


def make_executor():
    primitive = FakeExplorationExecutor()
    explorer = SimpleNamespace(executor=primitive)

    return (
        UnifiedDecisionExecutor(explorer=explorer),
        primitive,
    )


def make_decision(mode):
    return SimpleNamespace(mode=mode)


def make_blocked_action():
    return FakeAction(
        "take_key",
        preconditions={
            "box": "open",
            "key": "inside_box",
        },
    )


def make_blocked_environment():
    state = {
        "box": "closed",
        "key": "inside_box",
    }

    environment = SimpleNamespace(
        observe=lambda: state.copy(),
    )

    return environment, state


def test_direct_execution_runs_the_requested_action():
    executor, primitive = make_executor()

    requested = FakeAction(
        "take_key",
        preconditions={
            "box": "open",
            "key": "inside_box",
        },
    )

    environment = SimpleNamespace(
        observe=lambda: {
            "box": "open",
            "key": "inside_box",
        }
    )

    # An existing experience belongs to the caller.
    agent = SimpleNamespace(
        current_experience=object(),
    )

    result = executor.execute_action(
        decision=make_decision(DecisionMode.EXPLORE),
        action=requested,
        agent=agent,
        environment=environment,
    )

    assert primitive.executed == ["take_key"]
    assert result.action is requested
    assert result.success is True


def test_direct_execution_does_not_replace_action_in_exploit_mode():
    executor, primitive = make_executor()

    requested = FakeAction("unlock_door")

    environment = SimpleNamespace(
        observe=lambda: {
            "door": "locked",
            "key": "agent",
        }
    )

    agent = SimpleNamespace(
        current_experience=object(),
    )

    result = executor.execute_action(
        decision=make_decision(DecisionMode.EXPLOIT),
        action=requested,
        agent=agent,
        environment=environment,
    )

    assert primitive.executed == ["unlock_door"]
    assert result.action is requested
    assert result.success is True


def test_direct_execution_rejects_unsatisfied_preconditions():
    executor, primitive = make_executor()

    requested = make_blocked_action()
    environment, state = make_blocked_environment()

    agent = SimpleNamespace(
        current_experience=object(),
    )

    result = executor.execute_action(
        decision=make_decision(DecisionMode.EXPLORE),
        action=requested,
        agent=agent,
        environment=environment,
    )

    assert primitive.executed == []
    assert result.action is requested
    assert result.success is False
    assert result.direct_result["success"] is False
    assert result.direct_result["failure_type"] == "precondition_failed"
    assert result.direct_result["observation_before"] == state
    assert result.direct_result["observation_after"] == state


def test_precondition_failure_records_episode_without_taking_over_caller_experience():
    executor, primitive = make_executor()
    requested = make_blocked_action()
    environment, state = make_blocked_environment()

    caller_experience = object()
    agent = RecordingAgent(
        current_experience=caller_experience,
    )

    result = executor.execute_action(
        decision=make_decision(DecisionMode.EXPLORE),
        action=requested,
        agent=agent,
        environment=environment,
    )

    assert result.success is False
    assert result.action is requested
    assert primitive.executed == []

    assert len(agent.recorded_episodes) == 1

    episode = agent.recorded_episodes[0]
    assert episode["action"] == "take_key"
    assert episode["result"]["success"] is False
    assert episode["result"]["failure_type"] == "precondition_failed"
    assert episode["observation_before"] == state
    assert episode["observation_after"] == state

    # The caller still owns the existing experience.
    assert agent.current_experience is caller_experience
    assert "start_experience" not in agent.lifecycle_calls
    assert "complete_experience" not in agent.lifecycle_calls
    assert "fail_experience" not in agent.lifecycle_calls
    assert "close_experience" not in agent.lifecycle_calls


def test_precondition_failure_fails_and_closes_executor_owned_experience():
    executor, primitive = make_executor()
    requested = make_blocked_action()
    environment, state = make_blocked_environment()

    agent = RecordingAgent()

    result = executor.execute_action(
        decision=make_decision(DecisionMode.EXPLORE),
        action=requested,
        agent=agent,
        environment=environment,
        experience_id="direct-precondition-failure",
    )

    assert result.success is False
    assert primitive.executed == []
    assert len(agent.recorded_episodes) == 1

    episode = agent.recorded_episodes[0]
    assert episode["action"] == "take_key"
    assert episode["result"]["failure_type"] == "precondition_failed"
    assert episode["observation_before"] == state
    assert episode["observation_after"] == state

    assert agent.lifecycle_calls == [
        "start_experience",
        "fail_experience",
        "close_experience",
    ]
    assert agent.current_experience is None
