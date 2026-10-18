from __future__ import annotations

from dataclasses import dataclass

from agent.llm.p0_scenario_readiness import preflight_provider


@dataclass
class Task:
    task_id: str
    description: str
    payload: dict


@dataclass
class Step:
    action: str


@dataclass
class Plan:
    steps: tuple


class Goal:
    target_state = {"box": "open"}

    def is_satisfied(self, state):
        return state.get("box") == "open"


class Environment:
    def __init__(self):
        self.state = {"box": "closed"}

    def observe(self):
        return dict(self.state)


class Action:
    name = "open_box"


class Agent:
    def __init__(self):
        self.actions = {"open_box": Action()}


class Scenario:
    def __init__(self):
        self.agent = Agent()
        self.environment = Environment()
        self.goal = Goal()
        self.plan = Plan((Step("open_box"),))
        self.action_risks = {"open_box": 0.1}
        self.approval_required_actions = ()
        self.approval_checker = None
        self.risk_threshold = 0.65
        self.duplicate_harmful_side_effect_count = lambda: 0


class Provider:
    def __init__(self, *, reuse=False, missing_action=False):
        self.reuse = reuse
        self.missing_action = missing_action
        self.cached = None

    def tasks(self):
        return [Task(f"task-{i:02d}", f"task {i}", {}) for i in range(1, 31)]

    def build_scenario(self, task, run_index):
        if self.reuse:
            if self.cached is None:
                self.cached = Scenario()
            return self.cached
        scenario = Scenario()
        if self.missing_action:
            scenario.agent.actions = {}
        return scenario


def test_valid_provider_builds_90_isolated_scenarios_without_execution():
    report = preflight_provider(Provider(), provider_spec="test:provider")
    assert report.valid is True
    assert report.tasks_seen == 30
    assert report.scenarios_built == 90
    assert report.unique_agents == 90
    assert report.unique_environments == 90
    assert not report.issues
    assert "No plan was executed" in report.notes[0]


def test_reused_agent_and_environment_fail_closed():
    report = preflight_provider(Provider(reuse=True), provider_spec="test:provider")
    assert report.valid is False
    assert any(issue.code == "agent_reused" for issue in report.issues)
    assert any(issue.code == "environment_reused" for issue in report.issues)


def test_unregistered_action_fails_closed():
    report = preflight_provider(Provider(missing_action=True), provider_spec="test:provider")
    assert report.valid is False
    assert any(issue.code == "action_unregistered" for issue in report.issues)


def test_task_catalog_must_have_canonical_ids_and_count():
    class BadProvider(Provider):
        def tasks(self):
            return [Task("wrong-id", "wrong", {})]

    report = preflight_provider(BadProvider(), provider_spec="test:bad")
    assert report.valid is False
    codes = {issue.code for issue in report.issues}
    assert "task_count_mismatch" in codes
    assert "task_order_or_id_mismatch" in codes
