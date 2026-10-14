from types import SimpleNamespace

from agent.core.action import Action
from agent.core.experience import Experience
from agent.core.goal import Goal
from agent.core.agent import Agent
from agent.learning.strategy import Strategy


def make_episode(
    action: str,
    success: bool,
    cycle: int,
    observation_before: dict,
    observation_after: dict,
):
    return SimpleNamespace(
        action=action,
        result={
            "success": success,
        },
        cycle=cycle,
        observation_before=observation_before,
        observation_after=observation_after,
    )


def make_actions():
    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock door",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={
                "door": "unlocked",
            },
        ),
        Action(
            name="check_door",
            description="Check door",
            handler=lambda: {
                "success": False,
            },
            preconditions={
                "door": "unlocked",
            },
            effects={},
        ),
    ]


def make_source_strategy():
    return Strategy(
        name="open_door_original",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
            "check_door",
        ],
    )


def make_failed_experience():
    experience = Experience(
        experience_id="exp-evolution-evidence",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="open_box",
            success=True,
            cycle=1,
            observation_before={
                "door": "locked",
                "box": "closed",
                "key": "inside_box",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "inside_box",
            },
        )
    )

    experience.add_episode(
        make_episode(
            action="take_key",
            success=True,
            cycle=2,
            observation_before={
                "door": "locked",
                "box": "open",
                "key": "inside_box",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.add_episode(
        make_episode(
            action="unlock_door",
            success=True,
            cycle=3,
            observation_before={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
            observation_after={
                "door": "unlocked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.add_episode(
        make_episode(
            action="check_door",
            success=False,
            cycle=4,
            observation_before={
                "door": "unlocked",
                "box": "open",
                "key": "agent",
            },
            observation_after={
                "door": "unlocked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.fail()

    return experience


def test_evolution_creates_evidence():
    agent = Agent(
        name="test-agent"
    )

    agent.register_actions(
        make_actions()
    )

    source = make_source_strategy()

    agent.register_strategy(
        source
    )

    agent.set_active_strategy(
        source
    )

    goal = Goal(
        description="Open door",
        target_state={
            "door": "unlocked",
        },
    )

    experience = make_failed_experience()

    result = (
        agent.evolve_strategy_from_experience(
            experience=experience,
            goal=goal,
        )
    )

    assert result is not None
    assert result.selection.selected is True

    candidate = (
        result.selection.strategy
    )

    assert candidate is not None

    assert candidate.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    evidence = (
        agent.get_strategy_evolution_evidence(
            candidate.name
        )
    )

    assert evidence is not None

    assert evidence.strategy_name == (
        candidate.name
    )

    assert (
        evidence.source_strategy_name
        == source.name
    )

    assert (
        evidence.source_experience_id
        == experience.experience_id
    )

    assert evidence.generation == 1
    assert evidence.status == "experimental"

    assert evidence.trial_count == 0


def test_non_evolved_strategy_has_no_evolution_evidence():
    agent = Agent(
        name="test-agent"
    )

    strategy = Strategy(
        name="normal_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    agent.register_strategy(
        strategy
    )

    assert (
        agent.get_strategy_evolution_evidence(
            strategy.name
        )
        is None
    )


def test_evolution_evidence_can_be_retrieved_from_agent():
    agent = Agent(
        name="test-agent"
    )

    source = make_source_strategy()

    evidence = (
        agent.strategy_evolution_evidence
        .get_or_create(
            strategy_name="evolved_strategy",
            source_strategy_name=source.name,
            source_experience_id="exp-1",
        )
    )

    all_evidence = (
        agent.get_all_strategy_evolution_evidence()
    )

    assert len(all_evidence) == 1
    assert all_evidence[0] is evidence


def test_evolution_evidence_records_real_trial():
    agent = Agent(
        name="test-agent"
    )

    source = make_source_strategy()

    candidate = Strategy(
        name="evolved_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    agent.register_strategy(
        source
    )

    agent.register_strategy(
        candidate
    )

    evidence = (
        agent.strategy_evolution_evidence
        .get_or_create(
            strategy_name=candidate.name,
            source_strategy_name=source.name,
            source_experience_id="exp-origin",
        )
    )

    # Explicitly select the evolved strategy so
    # Agent.attribute_strategy() associates the trial
    # with this strategy.
    agent.set_active_strategy(
        candidate
    )

    experience = Experience(
        experience_id="exp-trial-1",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="open_box",
            success=True,
            cycle=1,
            observation_before={
                "door": "locked",
                "box": "closed",
                "key": "inside_box",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "inside_box",
            },
        )
    )

    experience.add_episode(
        make_episode(
            action="take_key",
            success=True,
            cycle=2,
            observation_before={
                "door": "locked",
                "box": "open",
                "key": "inside_box",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.add_episode(
        make_episode(
            action="unlock_door",
            success=True,
            cycle=3,
            observation_before={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
            observation_after={
                "door": "unlocked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.complete()

    agent.set_goal(
        Goal(
            description="Open door",
            target_state={
                "door": "unlocked",
            },
        )
    )

    agent.last_progress_learning = (
        SimpleNamespace(
            progress_delta=1.0
        )
    )

    agent._update_strategy_from_experience(
        experience=experience,
        success=True,
    )

    assert evidence.trial_count == 1
    assert evidence.success_count == 1
    assert evidence.failure_count == 0
    assert evidence.average_progress == 1.0


def test_failed_trial_updates_evolution_evidence():
    agent = Agent(
        name="test-agent"
    )

    source = make_source_strategy()

    candidate = Strategy(
        name="evolved_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    agent.register_strategy(
        source
    )

    agent.register_strategy(
        candidate
    )

    evidence = (
        agent.strategy_evolution_evidence
        .get_or_create(
            strategy_name=candidate.name,
            source_strategy_name=source.name,
            source_experience_id="exp-origin",
        )
    )

    agent.set_active_strategy(
        candidate
    )

    experience = Experience(
        experience_id="exp-trial-failure",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="open_box",
            success=True,
            cycle=1,
            observation_before={
                "door": "locked",
                "box": "closed",
                "key": "inside_box",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "inside_box",
            },
        )
    )

    experience.add_episode(
        make_episode(
            action="take_key",
            success=True,
            cycle=2,
            observation_before={
                "door": "locked",
                "box": "open",
                "key": "inside_box",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.add_episode(
        make_episode(
            action="unlock_door",
            success=False,
            cycle=3,
            observation_before={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.fail()

    agent.last_progress_learning = (
        SimpleNamespace(
            progress_delta=-0.5
        )
    )

    agent._update_strategy_from_experience(
        experience=experience,
        success=False,
    )

    assert evidence.trial_count == 1
    assert evidence.success_count == 0
    assert evidence.failure_count == 1
    assert evidence.average_progress == -0.5

def make_bootstrap_experience(
    experience_id: str,
    goal: str = "Open door",
    actions=None,
):
    from agent.core.experience import Experience

    if actions is None:
        actions = [
            "open_box",
            "take_key",
            "unlock_door",
        ]

    experience = Experience(
        experience_id=experience_id,
        goal=goal,
    )

    for cycle, action_name in enumerate(actions, start=1):
        experience.add_episode(
            make_episode(
                action=action_name,
                success=True,
                cycle=cycle,
                observation_before={"cycle": cycle - 1},
                observation_after={"cycle": cycle},
            )
        )

    return experience


def test_bootstrap_creates_strategy_from_successful_experience():
    agent = Agent(name="bootstrap-test")
    agent.register_actions(make_actions())

    experience = make_bootstrap_experience("bootstrap-success")

    agent._update_strategy_from_experience(
        experience=experience,
        success=True,
    )

    strategies = agent.get_registered_strategies()

    assert len(strategies) == 1

    strategy = strategies[0]
    assert strategy.goal == "Open door"
    assert strategy.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]
    assert strategy.success_count == 1
    assert strategy.failure_count == 0
    assert "bootstrap-success" in strategy.source_experiences


def test_bootstrap_records_failed_experience():
    agent = Agent(name="bootstrap-failure-test")
    agent.register_actions(make_actions())

    experience = make_bootstrap_experience("bootstrap-failure")

    agent._update_strategy_from_experience(
        experience=experience,
        success=False,
    )

    strategies = agent.get_registered_strategies()

    assert len(strategies) == 1
    assert strategies[0].success_count == 0
    assert strategies[0].failure_count == 1


def test_bootstrap_reuses_strategy_for_same_goal_and_steps():
    agent = Agent(name="bootstrap-dedupe-test")
    agent.register_actions(make_actions())

    first = make_bootstrap_experience("bootstrap-repeat-1")
    second = make_bootstrap_experience("bootstrap-repeat-2")

    agent._update_strategy_from_experience(first, success=True)
    agent._update_strategy_from_experience(second, success=False)

    strategies = agent.get_registered_strategies()

    assert len(strategies) == 1
    assert strategies[0].success_count == 1
    assert strategies[0].failure_count == 1
    assert set(strategies[0].source_experiences) == {
        "bootstrap-repeat-1",
        "bootstrap-repeat-2",
    }


def test_bootstrap_does_not_count_same_experience_twice():
    agent = Agent(name="bootstrap-idempotency-test")
    agent.register_actions(make_actions())

    experience = make_bootstrap_experience("bootstrap-once")

    agent._update_strategy_from_experience(experience, success=True)
    agent._update_strategy_from_experience(experience, success=True)

    strategies = agent.get_registered_strategies()

    assert len(strategies) == 1
    assert strategies[0].success_count == 1
    assert strategies[0].attempts == 1


def test_bootstrap_rejects_unregistered_action():
    agent = Agent(name="bootstrap-invalid-action-test")
    agent.register_actions(make_actions())

    experience = make_bootstrap_experience(
        "bootstrap-invalid",
        actions=["action_that_does_not_exist"],
    )

    agent._update_strategy_from_experience(
        experience=experience,
        success=True,
    )

    assert agent.get_registered_strategies() == []


def test_bootstrap_strategy_persists_after_restart(tmp_path):
    database_path = tmp_path / "bootstrap-memory.db"

    agent = Agent(
        name="bootstrap-persistence-test",
        memory_path=str(database_path),
    )
    agent.register_actions(make_actions())

    experience = make_bootstrap_experience("bootstrap-persisted")

    agent._update_strategy_from_experience(
        experience=experience,
        success=True,
    )

    restarted_agent = Agent(
        name="bootstrap-persistence-test",
        memory_path=str(database_path),
    )

    strategies = restarted_agent.get_registered_strategies()

    assert len(strategies) == 1
    assert strategies[0].steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]
    assert strategies[0].success_count == 1
    assert "bootstrap-persisted" in strategies[0].source_experiences
