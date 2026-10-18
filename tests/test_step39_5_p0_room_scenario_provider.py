from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.llm.p0_canonical_runtime_adapter import P0RuntimeScenario
from agent.llm.p0_runtime_benchmark import P0BenchmarkTask
from agent.llm.p0_scenario_readiness import preflight_provider
from agent.llm.p0_room_scenario_provider import RoomP0ScenarioProvider, build_provider
from agent.world.room import RoomEnvironment


def test_provider_exposes_exact_canonical_task_matrix():
    provider = build_provider()
    tasks = list(provider.tasks())
    assert len(tasks) == 30
    assert [task.task_id for task in tasks] == [f"task-{i:02d}" for i in range(1, 31)]
    assert all(isinstance(task, P0BenchmarkTask) for task in tasks)


def test_build_scenario_uses_fresh_real_agent_and_room_environment():
    provider = RoomP0ScenarioProvider()
    task = list(provider.tasks())[0]
    first = provider.build_scenario(task, 1)
    second = provider.build_scenario(task, 2)
    assert isinstance(first, P0RuntimeScenario)
    assert isinstance(first.agent, Agent)
    assert isinstance(first.environment, RoomEnvironment)
    assert first.agent is not second.agent
    assert first.environment is not second.environment
    assert set(first.agent.actions) == {"open_box", "take_key", "find_spare_key", "unlock_door"}


def test_all_90_scenarios_pass_readiness_without_executing_actions():
    report = preflight_provider(build_provider(), expected_tasks=30, runs_per_task=3)
    assert report.valid, report.to_dict()
    assert report.tasks_seen == 30
    assert report.scenarios_built == 90
    assert report.unique_agents == 90
    assert report.unique_environments == 90


def test_task_goal_and_plan_route_are_consistent():
    provider = RoomP0ScenarioProvider()
    tasks = list(provider.tasks())
    by_id = {task.task_id: task for task in tasks}
    box_task = provider.build_scenario(by_id["task-01"], 1)
    spare_task = provider.build_scenario(by_id["task-11"], 1)
    open_task = provider.build_scenario(by_id["task-21"], 1)
    take_task = provider.build_scenario(by_id["task-26"], 1)
    assert [step.action for step in box_task.plan.steps] == ["open_box", "take_key", "unlock_door"]
    assert [step.action for step in spare_task.plan.steps] == ["find_spare_key", "unlock_door"]
    assert [step.action for step in open_task.plan.steps] == ["open_box"]
    assert [step.action for step in take_task.plan.steps] == ["open_box", "take_key"]
    assert box_task.goal.target_state == {"door": "unlocked"}
    assert open_task.goal.target_state == {"box": "open"}
    assert take_task.goal.target_state == {"key": "agent"}
