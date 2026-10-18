from types import SimpleNamespace

from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.plan_validation import LLMPlanValidationResult
from agent.llm.planning import LLMPlan, LLMPlanStep


class FakeGoal:
    def __init__(self, expected):
        self.expected = expected

    def is_satisfied(self, world_state):
        return world_state.get("done") is self.expected


class FakeBridge:
    def __init__(self, *, success=True):
        self.api = SimpleNamespace(agent=SimpleNamespace(actions={}))
        self.last_execution_request = None
        self.success = success
        self.calls = []

    def execute_step(self, plan, step, **kwargs):
        self.calls.append(step.index)
        return SimpleNamespace(
            success=self.success,
            result={"success": self.success},
        )


class FakeValidator:
    def validate(self, plan, *, goal, available_actions):
        return LLMPlanValidationResult(
            valid=True,
            issues=(),
            checked_step_count=len(plan.steps),
            executable_step_indices=tuple(step.index for step in plan.steps),
        )


class FakeEnvironment:
    def __init__(self, state):
        self.state = dict(state)

    def observe(self):
        return dict(self.state)


def make_plan():
    return LLMPlan(
        goal="finish task",
        rationale="test plan",
        steps=(LLMPlanStep(index=1, action="noop", purpose="perform step"),),
        confidence=0.9,
    )


def make_runtime(bridge, **kwargs):
    return LLMPlanRuntimeExecutor(
        bridge,
        validator=FakeValidator(),
        world_state_observer=lambda env: env.observe(),
        **kwargs,
    )


def test_successful_actions_do_not_fake_goal_success():
    bridge = FakeBridge(success=True)
    runtime = make_runtime(bridge)
    report = runtime.execute(
        make_plan(),
        goal=FakeGoal(expected=True),
        environment=FakeEnvironment({"done": False}),
    )

    assert report.completed is True
    assert report.goal_satisfied is False
    assert report.success is False
    assert report.stop_reason == "goal_not_satisfied"
    assert report.metadata["goal_verification_performed"] is True


def test_goal_satisfaction_allows_success():
    bridge = FakeBridge(success=True)
    runtime = make_runtime(bridge)
    report = runtime.execute(
        make_plan(),
        goal=FakeGoal(expected=True),
        environment=FakeEnvironment({"done": True}),
    )

    assert report.completed is True
    assert report.goal_satisfied is True
    assert report.success is True
    assert report.stop_reason is None


def test_text_goal_preserves_legacy_behavior():
    bridge = FakeBridge(success=True)
    runtime = make_runtime(bridge)
    report = runtime.execute(
        make_plan(),
        goal="finish task",
        environment=FakeEnvironment({"done": False}),
    )

    assert report.completed is True
    assert report.goal_satisfied is None
    assert report.success is True
    assert report.metadata["goal_verification_performed"] is False


def test_explicit_goal_evaluator_takes_precedence():
    bridge = FakeBridge(success=True)
    calls = []

    def verify(goal, state, context):
        calls.append((goal, state, context))
        return False

    runtime = make_runtime(bridge, goal_satisfaction_evaluator=verify)
    report = runtime.execute(
        make_plan(),
        goal="finish task",
        environment=FakeEnvironment({"done": True}),
    )

    assert report.goal_satisfied is False
    assert report.success is False
    assert len(calls) == 1
    assert calls[0][1] == {"done": True}
    assert calls[0][2]["records"][0].success is True


def test_invalid_goal_evaluator_configuration_is_rejected():
    bridge = FakeBridge()
    try:
        LLMPlanRuntimeExecutor(
            bridge,
            validator=FakeValidator(),
            goal_satisfaction_evaluator="not-callable",
        )
    except Exception as exc:
        assert "goal_satisfaction_evaluator" in str(exc)
    else:
        raise AssertionError("Expected invalid evaluator configuration to fail")
