from agent.core.agent import Agent
from agent.learning.meta_learning import MetaLearningProfile


def test_agent_creates_meta_learning_runtime():
    agent = Agent("runtime-agent")

    assert agent.meta_learning_runtime is not None
    assert (
        agent.meta_learning_runtime.profile
        is agent.meta_learning_profile
    )


def test_agent_meta_learning_runtime_uses_agent_decision_maker():
    agent = Agent("runtime-agent")

    assert (
        agent.meta_learning_runtime
        .decision_maker
        .base_decision_maker
        is agent.decision_maker
    )


def test_agent_can_evaluate_meta_learning_runtime():
    agent = Agent("runtime-agent")

    result = agent.evaluate_meta_learning_runtime(
        uncertainty=0.2,
        information_gain=0.2,
        risk=0.2,
    )

    assert result is not None
    assert result.feedback is not None
    assert result.feedback_decision is not None
    assert result.decision is not None


def test_runtime_decision_preserves_base_agent_decision():
    agent = Agent("runtime-agent")

    result = agent.evaluate_meta_learning_runtime(
        uncertainty=0.2,
        information_gain=0.2,
        risk=0.2,
    )

    assert (
        result.decision.base_decision
        is not None
    )


def test_runtime_profile_is_same_profile_as_agent():
    agent = Agent("runtime-agent")

    assert (
        agent.meta_learning_runtime.profile
        is agent.get_meta_learning_profile()
    )


def test_runtime_evaluation_does_not_record_learning_event():
    agent = Agent("runtime-agent")

    before = agent.get_meta_learning_snapshot()

    agent.evaluate_meta_learning_runtime(
        uncertainty=0.8,
        information_gain=0.8,
        risk=0.2,
    )

    after = agent.get_meta_learning_snapshot()

    assert after == before


def test_runtime_respects_explore_constraint():
    agent = Agent("runtime-agent")

    result = agent.evaluate_meta_learning_runtime(
        uncertainty=1.0,
        information_gain=1.0,
        risk=0.0,
        explore_allowed=False,
        exploit_allowed=True,
    )

    assert result.mode == "exploit"


def test_runtime_respects_exploit_constraint():
    agent = Agent("runtime-agent")

    result = agent.evaluate_meta_learning_runtime(
        uncertainty=0.0,
        information_gain=0.0,
        risk=0.0,
        explore_allowed=True,
        exploit_allowed=False,
    )

    assert result.mode == "explore"
