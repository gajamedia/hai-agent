from types import SimpleNamespace

import pytest

from agent.core.action import Action
from agent.llm.dynamic_workspace_actions import (
    DynamicActionSpec,
    DynamicWorkspaceActionBuilder,
    DynamicWorkspaceActionError,
    DynamicWorkspaceValidationError,
    DynamicWorkspaceRegistrationError,
)


class FakeAgent:
    def __init__(self):
        self.actions = {}

    def register_action(self, action):
        if action.name in self.actions:
            raise ValueError(f"Duplicate action: {action.name}")
        self.actions[action.name] = action


class FakeEnvironment:
    def observe(self):
        return {
            "door": "locked",
            "box": "closed",
        }


def make_plan(*action_names):
    return SimpleNamespace(
        steps=[
            SimpleNamespace(action=name)
            for name in action_names
        ]
    )


def observe_spec(name="inspect_room", **kwargs):
    return DynamicActionSpec(
        name=name,
        description="Inspect the current room state.",
        capability_id="environment.observe",
        **kwargs,
    )


def test_generates_and_registers_missing_plan_action():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)

    report = builder.ensure_plan_actions(
        make_plan("inspect_room"),
        {"inspect_room": observe_spec()},
    )

    assert report.generated_actions == ("inspect_room",)
    assert report.registered_actions == ("inspect_room",)
    assert "inspect_room" in agent.actions


def test_generated_action_executes_through_environment():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)
    action = builder.build_action(observe_spec())

    result = action.execute(environment=FakeEnvironment())

    assert result["success"] is True
    assert result["observation"]["door"] == "locked"


def test_existing_action_is_reused_without_overwrite():
    agent = FakeAgent()
    existing = object()
    agent.actions["open_box"] = existing
    builder = DynamicWorkspaceActionBuilder(agent)

    report = builder.ensure_plan_actions(make_plan("open_box"), {})

    assert report.existing_actions == ("open_box",)
    assert report.generated_actions == ()
    assert agent.actions["open_box"] is existing


def test_missing_specification_is_rejected_without_registration():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)

    with pytest.raises(DynamicWorkspaceValidationError):
        builder.ensure_plan_actions(make_plan("unknown_action"), {})

    assert agent.actions == {}


def test_unsupported_capability_is_rejected():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)
    spec = DynamicActionSpec(
        name="run_unknown",
        description="Unknown operation.",
        capability_id="python.execute",
    )

    with pytest.raises(DynamicWorkspaceValidationError):
        builder.build_action(spec)

    assert agent.actions == {}


def test_invalid_action_name_is_rejected():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)
    spec = DynamicActionSpec(
        name="Run Python",
        description="Invalid action name.",
        capability_id="environment.observe",
    )

    with pytest.raises(DynamicWorkspaceValidationError):
        builder.build_action(spec)


def test_existing_action_is_never_overwritten():
    agent = FakeAgent()
    existing = object()
    agent.actions["inspect_room"] = existing
    builder = DynamicWorkspaceActionBuilder(agent)

    report = builder.ensure_plan_actions(
        make_plan("inspect_room"),
        {},
    )

    assert report.existing_actions == ("inspect_room",)
    assert report.generated_actions == ()
    assert report.registered_actions == ()
    assert agent.actions["inspect_room"] is existing


def test_all_missing_actions_are_validated_before_registration():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)
    specs = {
        "inspect_room": observe_spec(),
        "unsafe_action": DynamicActionSpec(
            name="unsafe_action",
            description="Unsupported operation.",
            capability_id="python.execute",
        ),
    }

    with pytest.raises(DynamicWorkspaceValidationError):
        builder.ensure_plan_actions(
            make_plan("inspect_room", "unsafe_action"),
            specs,
        )

    assert agent.actions == {}


def test_agent_rejects_duplicate_action_registration():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)
    action = builder.build_action(observe_spec())

    agent.register_action(action)

    with pytest.raises(ValueError):
        agent.register_action(action)

    assert agent.actions["inspect_room"] is action


def test_wrong_spec_type_raises_validation_error_not_attribute_error():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)

    with pytest.raises(DynamicWorkspaceValidationError):
        builder.ensure_plan_actions(
            make_plan("inspect_room"),
            {"inspect_room": object()},
        )

    assert agent.actions == {}


def test_invalid_plan_action_name_is_rejected():
    agent = FakeAgent()
    builder = DynamicWorkspaceActionBuilder(agent)

    with pytest.raises(DynamicWorkspaceValidationError):
        builder.ensure_plan_actions(make_plan(" inspect_room "), {})

    assert agent.actions == {}


def test_capability_result_must_contain_success():
    agent = FakeAgent()

    def bad_capability(*, environment, parameters, **kwargs):
        return {"message": "no success flag"}

    builder = DynamicWorkspaceActionBuilder(
        agent,
        capabilities={"test.bad": bad_capability},
    )
    action = builder.build_action(
        DynamicActionSpec(
            name="bad_action",
            description="Returns an invalid result.",
            capability_id="test.bad",
        )
    )

    with pytest.raises(DynamicWorkspaceActionError):
        action.execute(environment=FakeEnvironment())


def test_capability_may_return_any_mapping():
    agent = FakeAgent()

    class ResultMapping(dict):
        pass

    def capability(*, environment, parameters, **kwargs):
        return ResultMapping(success=True, value=parameters["mode"])

    builder = DynamicWorkspaceActionBuilder(
        agent,
        capabilities={"test.mapping": capability},
    )
    action = builder.build_action(
        DynamicActionSpec(
            name="configured_action",
            description="Uses trusted fixed parameters.",
            capability_id="test.mapping",
            parameters={"mode": "safe"},
        )
    )

    result = action.execute(environment=FakeEnvironment())

    assert result == {"success": True, "value": "safe"}


def test_reserved_parameters_argument_is_rejected():
    agent = FakeAgent()

    def capability(*, environment, parameters, **kwargs):
        return {"success": True}

    builder = DynamicWorkspaceActionBuilder(
        agent,
        capabilities={"test.capability": capability},
    )
    action = builder.build_action(
        DynamicActionSpec(
            name="reserved_arg_action",
            description="Tests reserved arguments.",
            capability_id="test.capability",
        )
    )

    with pytest.raises(DynamicWorkspaceValidationError):
        action.execute(
            environment=FakeEnvironment(),
            parameters={"override": True},
        )


def test_registration_failure_rolls_back_all_registry_changes():
    class FailingAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def register_action(self, action):
            self.calls += 1
            if self.calls == 2:
                # Simulate a registration method that mutates and then fails.
                self.actions[action.name] = action
                raise RuntimeError("simulated registration failure")
            super().register_action(action)

    agent = FailingAgent()
    original = object()
    agent.actions["preexisting"] = original
    builder = DynamicWorkspaceActionBuilder(agent)

    specs = {
        "inspect_room": observe_spec(),
        "inspect_hall": observe_spec("inspect_hall"),
    }

    with pytest.raises(DynamicWorkspaceRegistrationError):
        builder.ensure_plan_actions(
            make_plan("inspect_room", "inspect_hall"),
            specs,
        )

    assert agent.actions == {"preexisting": original}
    assert agent.actions["preexisting"] is original


def test_builder_with_real_agent_registers_action_and_updates_planner():
    from agent.core.agent import Agent

    agent = Agent(name="dynamic-workspace-builder-test")
    builder = DynamicWorkspaceActionBuilder(agent)

    report = builder.ensure_plan_actions(
        make_plan("inspect_room"),
        {"inspect_room": observe_spec()},
    )

    assert report.registered_actions == ("inspect_room",)
    assert isinstance(agent.actions["inspect_room"], Action)
    assert agent.planner is not None


def test_builder_with_real_agent_never_overwrites_existing_action():
    from agent.core.agent import Agent

    agent = Agent(name="dynamic-workspace-existing-action-test")
    existing = Action(
        name="inspect_room",
        description="Original trusted action.",
        handler=lambda **kwargs: {"success": True, "source": "original"},
    )
    agent.register_action(existing)
    builder = DynamicWorkspaceActionBuilder(agent)

    report = builder.ensure_plan_actions(
        make_plan("inspect_room"),
        {},
    )

    assert report.existing_actions == ("inspect_room",)
    assert report.registered_actions == ()
    assert agent.actions["inspect_room"] is existing
    assert agent.actions["inspect_room"].execute() == {
        "success": True,
        "source": "original",
    }
