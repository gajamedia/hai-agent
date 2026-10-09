
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


def make_executor():
    primitive = FakeExplorationExecutor()
    explorer = SimpleNamespace(executor=primitive)

    return (
        UnifiedDecisionExecutor(explorer=explorer),
        primitive,
    )


def make_decision(mode):
    return SimpleNamespace(mode=mode)


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

    requested = FakeAction(
        "take_key",
        preconditions={
            "box": "open",
            "key": "inside_box",
        },
    )

    state = {
        "box": "closed",
        "key": "inside_box",
    }

    environment = SimpleNamespace(
        observe=lambda: state.copy(),
    )

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
    assert result.direct_result["observation_after"] == state
