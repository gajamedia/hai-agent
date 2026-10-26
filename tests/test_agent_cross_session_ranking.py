import importlib.util
from pathlib import Path

from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.world.room import RoomEnvironment


_HELPER_PATH = (
    Path(__file__).parent
    / "test_episodic_end_to_end_learning.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "_episodic_e2e_helpers",
    _HELPER_PATH,
)
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


GOAL_DESCRIPTION = "open the door"


def test_real_experience_changes_ranking_after_agent_restart(
    tmp_path,
):
    database_path = tmp_path / "cross_session_learning.db"

    # Session 1: record actual execution outcomes in persistent memory.
    first_agent = Agent(
        name="cross-session-learning-agent",
        memory_path=database_path,
    )
    first_agent.register_actions(_HELPERS.make_actions())

    environment = RoomEnvironment()

    for _ in range(8):
        episode = _HELPERS.record_action(
            first_agent,
            environment,
            action_name="take_key",
            success=False,
        )
        assert episode.result["success"] is False

    for _ in range(8):
        episode = _HELPERS.record_action(
            first_agent,
            environment,
            action_name="find_spare_key",
            success=True,
        )
        assert episode.result["success"] is True

    assert first_agent.episodic_memory.count() == 16
    assert first_agent.persistent_memory.episodic.count() == 16

    del first_agent

    # Session 2: create a fresh agent using the same database.
    restored_agent = Agent(
        name="cross-session-learning-agent",
        memory_path=database_path,
    )
    restored_agent.register_actions(_HELPERS.make_actions())

    restored_store = restored_agent.episodic_memory
    assert restored_store.count() == 16

    episodes = restored_store.all()
    failures = [
        episode for episode in episodes
        if episode.action == "take_key"
    ]
    successes = [
        episode for episode in episodes
        if episode.action == "find_spare_key"
    ]

    assert len(failures) == 8
    assert all(
        episode.result["success"] is False
        for episode in failures
    )
    assert len(successes) == 8
    assert all(
        episode.result["success"] is True
        for episode in successes
    )

    # Verify that restored experience affects planner ranking.
    planner = _HELPERS.make_planner()
    strategies = _HELPERS.make_strategies()
    goal = Goal(GOAL_DESCRIPTION)
    world_state = environment.observe()

    baseline = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=goal,
        world_state=world_state,
        episodic_memory=None,
    )
    learned = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=goal,
        world_state=world_state,
        episodic_memory=restored_store,
    )

    assert len(baseline) == 2
    assert len(learned) == 2
    assert learned[0][0].name == "spare-key"
    assert learned[0][1] > learned[1][1]


def test_public_planner_plan_uses_restored_episodic_memory(
    tmp_path,
):
    database_path = tmp_path / "public_planner_memory.db"

    first_agent = Agent(
        name="public-planner-cross-session-agent",
        memory_path=database_path,
    )
    first_agent.register_actions(_HELPERS.make_actions())

    environment = RoomEnvironment()

    for _ in range(8):
        episode = _HELPERS.record_action(
            first_agent,
            environment,
            action_name="take_key",
            success=False,
        )
        assert episode.result["success"] is False

    for _ in range(8):
        episode = _HELPERS.record_action(
            first_agent,
            environment,
            action_name="find_spare_key",
            success=True,
        )
        assert episode.result["success"] is True

    del first_agent

    restored_agent = Agent(
        name="public-planner-cross-session-agent",
        memory_path=database_path,
    )
    restored_agent.register_actions(_HELPERS.make_actions())

    assert restored_agent.episodic_memory.count() == 16

    planner = _HELPERS.make_planner()
    goal = Goal(GOAL_DESCRIPTION)
    strategies = _HELPERS.make_strategies()

    # Use a fresh planning environment so recorded experiences
    # cannot accidentally mutate the state used for planning.
    planning_environment = RoomEnvironment()
    world_state = planning_environment.observe()

    plan_without_memory = planner.plan(
        goal=goal,
        world_state=world_state,
        strategies=strategies,
        episodic_memory=None,
    )

    plan_with_restored_memory = planner.plan(
        goal=goal,
        world_state=world_state,
        strategies=strategies,
        episodic_memory=restored_agent.episodic_memory,
    )

    assert plan_without_memory
    assert plan_with_restored_memory
    assert plan_with_restored_memory[0].name == "find_spare_key"
