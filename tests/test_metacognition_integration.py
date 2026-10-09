from types import SimpleNamespace
from unittest.mock import Mock

from agent.core.agent import Agent
from agent.learning.metacognition import (
    MetacognitiveReflection,
    MetacognitionEngine,
)


class DummyGoal:
    def __init__(
        self,
        description="metacognition integration",
        target_state=None,
    ):
        self.description = description
        self.target_state = (
            target_state
            if target_state is not None
            else {}
        )
        self.completed = False

    def is_satisfied(
        self,
        world_state,
    ):
        if not self.target_state:
            return False

        return all(
            world_state.get(key) == value
            for key, value in self.target_state.items()
        )

    def complete(self):
        self.completed = True


def build_agent():
    agent = Agent(
        name="metacognitive-agent",
    )

    # Isolate this integration test from the lower-level
    # learning pipelines. We are testing that Agent correctly
    # invokes and exposes the metacognition layer.
    agent.episode_recorder.record = Mock()
    agent.semantic_learning.learn_from_episode = Mock()
    agent.online_causal_learning.observe_episode = Mock()

    return agent


def test_agent_initializes_metacognition_engine():
    agent = build_agent()

    assert isinstance(
        agent.metacognition_engine,
        MetacognitionEngine,
    )

    assert (
        agent.get_metacognitive_reflection()
        is None
    )


def test_agent_can_reflect_on_episode_directly():
    agent = build_agent()

    episode = {
        "episode_id": "ep-direct",
        "goal": "finish task",
        "action": "execute",
        "result": {
            "success": True,
        },
    }

    reflection = agent.reflect_on_episode(
        episode,
        progress_before=0.2,
        progress_after=0.8,
    )

    assert isinstance(
        reflection,
        MetacognitiveReflection,
    )

    assert reflection.progress_delta == 0.6
    assert (
        agent.get_metacognitive_reflection()
        is reflection
    )


def test_record_experience_generates_metacognitive_reflection():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    agent.set_goal(goal)

    episode = {
        "episode_id": "ep-success",
        "goal": goal.description,
        "action": "complete_task",
        "result": {
            "success": True,
        },
    }

    agent.episode_recorder.record.return_value = (
        episode
    )

    recorded = agent.record_experience(
        action="complete_task",
        observation_before={
            "done": False,
        },
        observation_after={
            "done": True,
        },
        result={
            "success": True,
        },
    )

    reflection = (
        agent.get_metacognitive_reflection()
    )

    assert recorded is episode
    assert reflection is not None
    assert reflection.success is True
    assert reflection.progress_before == 0.0
    assert reflection.progress_after == 1.0
    assert reflection.progress_delta == 1.0
    assert (
        reflection.diagnosis
        == "successful_outcome_with_progress"
    )
    assert reflection.change_required is False


def test_record_experience_generates_failure_metacognition():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    agent.set_goal(goal)

    episode = {
        "episode_id": "ep-failure",
        "goal": goal.description,
        "action": "complete_task",
        "result": {
            "success": False,
            "error": "timeout",
        },
    }

    agent.episode_recorder.record.return_value = (
        episode
    )

    agent.record_experience(
        action="complete_task",
        observation_before={
            "done": False,
        },
        observation_after={
            "done": False,
        },
        result={
            "success": False,
            "error": "timeout",
        },
    )

    reflection = (
        agent.get_metacognitive_reflection()
    )

    assert reflection is not None
    assert reflection.success is False
    assert reflection.progress_delta == 0.0
    assert (
        reflection.diagnosis
        == "failure_with_reported_error"
    )
    assert reflection.change_required is True


def test_record_experience_without_goal_preserves_missing_progress_evidence():
    agent = build_agent()

    episode = {
        "episode_id": "ep-no-goal",
        "action": "inspect",
        "result": {
            "success": True,
        },
    }

    agent.episode_recorder.record.return_value = (
        episode
    )

    agent.record_experience(
        action="inspect",
        observation_before={
            "value": 1,
        },
        observation_after={
            "value": 2,
        },
        result={
            "success": True,
        },
    )

    reflection = (
        agent.get_metacognitive_reflection()
    )

    assert reflection is not None
    assert reflection.progress_before is None
    assert reflection.progress_after is None
    assert reflection.progress_delta is None
    assert reflection.evidence_quality == 0.6
    assert reflection.confidence == 0.68


def test_clear_memory_invalidates_metacognitive_cache():
    agent = build_agent()

    reflection = agent.reflect_on_episode(
        {
            "result": {
                "success": False,
            },
        }
    )

    assert reflection is not None
    assert (
        agent.get_metacognitive_reflection()
        is reflection
    )

    agent.clear_memory()

    assert (
        agent.get_metacognitive_reflection()
        is None
    )