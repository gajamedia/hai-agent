
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from agent.llm.llm_plan_agent_experience_integration import (
    LLMPlanAgentExperienceIntegrator,
)
from agent.llm.llm_plan_live_runtime import (
    LLMPlanLiveRuntimeExecutor,
)
from agent.llm.llm_plan_runtime_execution import (
    LLMPlanRuntimeExecutor,
)
from agent.llm.llm_plan_single_action_bridge import (
    LLMPlanSingleActionBridge,
)
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass(frozen=True)
class FakeExecutionResult:
    success: bool
    result: dict[str, Any]


class FakeEnvironment:
    """Small deterministic environment for the complete runtime pipeline."""

    def __init__(self) -> None:
        self.state = {
            "box": "closed",
            "key": "inside_box",
        }

    def observe(self) -> dict[str, str]:
        return deepcopy(self.state)


class FakeAgent:
    """Agent-shaped object used to verify experience integration."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(cycle=0)
        self.current_experience = None
        self.calls: list[tuple[Any, ...]] = []
        self.episodes: list[dict[str, Any]] = []

    def act(self, action: str) -> None:
        self.calls.append(("act", action))

    def observe(self, observation: Any) -> None:
        self.calls.append(("observe", deepcopy(observation)))

    def step(self) -> None:
        self.state.cycle += 1
        self.calls.append(("step",))

    def record_experience(
        self,
        action: str,
        observation_before: Any,
        observation_after: Any,
        result: Any,
    ) -> dict[str, Any]:
        episode = {
            "action": action,
            "observation_before": deepcopy(observation_before),
            "observation_after": deepcopy(observation_after),
            "result": result,
        }
        self.episodes.append(episode)
        self.calls.append(("record_experience", episode))
        return episode

    def decide_and_act(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "The LLM plan runtime must not call decide_and_act()."
        )


class FakeCanonicalAPI:
    """
    Deterministic API stand-in.

    It enforces action registration and executes only the requested action.
    The production runtime, bridge, live-state tracker, and experience
    integrator remain real implementations.
    """

    def __init__(self, environment: FakeEnvironment) -> None:
        self.environment = environment
        self.calls: list[Any] = []
        self.agent = SimpleNamespace(
            actions={
                "open_box": object(),
                "take_key": object(),
            }
        )

    def execute(self, request: Any) -> FakeExecutionResult:
        self.calls.append(request)

        if request.action not in self.agent.actions:
            return FakeExecutionResult(
                success=False,
                result={
                    "success": False,
                    "error": f"Unregistered action: {request.action}",
                },
            )

        if request.action == "open_box":
            if self.environment.state["box"] != "closed":
                return FakeExecutionResult(
                    success=False,
                    result={
                        "success": False,
                        "error": "Box is not closed.",
                    },
                )

            self.environment.state["box"] = "open"

        elif request.action == "take_key":
            if self.environment.state["box"] != "open":
                return FakeExecutionResult(
                    success=False,
                    result={
                        "success": False,
                        "error": "Box must be open before taking the key.",
                    },
                )

            if self.environment.state["key"] != "inside_box":
                return FakeExecutionResult(
                    success=False,
                    result={
                        "success": False,
                        "error": "Key is no longer inside the box.",
                    },
                )

            self.environment.state["key"] = "held"

        else:
            return FakeExecutionResult(
                success=False,
                result={
                    "success": False,
                    "error": f"Unsupported action: {request.action}",
                },
            )

        return FakeExecutionResult(
            success=True,
            result={
                "success": True,
                "action": request.action,
                "observation": self.environment.observe(),
            },
        )


def make_plan() -> LLMPlan:
    return LLMPlan(
        goal="take the key from the box",
        rationale="Open the box before taking the key.",
        steps=(
            LLMPlanStep(
                index=1,
                action="open_box",
                purpose="Open the closed box.",
                preconditions=("box is closed",),
                expected_observation="box is open",
                success_criteria="box is open",
                risk="low",
            ),
            LLMPlanStep(
                index=2,
                action="take_key",
                purpose="Take the key from the open box.",
                preconditions=("box is open",),
                expected_observation="key is held",
                success_criteria="key is held",
                risk="low",
            ),
        ),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.95,
        provider="test",
        model="deterministic-e2e",
    )


def build_pipeline():
    environment = FakeEnvironment()
    agent = FakeAgent()
    api = FakeCanonicalAPI(environment)

    bridge = LLMPlanSingleActionBridge.for_api(api)

    runtime = LLMPlanRuntimeExecutor.for_bridge(
        bridge,
        world_state_observer=lambda env: env.observe(),
    )

    live_runtime = LLMPlanLiveRuntimeExecutor.for_executor(runtime)

    integrator = LLMPlanAgentExperienceIntegrator.for_agent(
        agent,
        live_runtime,
    )

    return environment, agent, api, bridge, runtime, live_runtime, integrator


def test_end_to_end_executes_plan_and_records_agent_experiences():
    (
        environment,
        agent,
        api,
        bridge,
        runtime,
        live_runtime,
        integrator,
    ) = build_pipeline()

    plan = make_plan()

    result = integrator.execute(
        plan,
        environment=environment,
        world_state=environment.observe(),
    )

    assert result.success is True
    assert result.completed is True
    assert result.stopped is False

    assert [request.action for request in api.calls] == [
        "open_box",
        "take_key",
    ]

    assert [record.step.index for record in result.runtime_result.report.records] == [
        1,
        2,
    ]

    assert all(
        record.success
        for record in result.runtime_result.report.records
    )

    assert environment.observe() == {
        "box": "open",
        "key": "held",
    }

    assert result.final_state.plan_goal == plan.goal
    # assert result.progress == 1.0
    assert len(result.experiences) == len(plan.steps)
    assert tuple(item.step_index for item in result.experiences) == (1, 2)    

    assert [item.action for item in result.experiences] == [
        "open_box",
        "take_key",
    ]

    assert len(agent.episodes) == 2

    assert agent.episodes[0]["observation_before"] == {
        "box": "closed",
        "key": "inside_box",
    }
    assert agent.episodes[0]["observation_after"] == {
        "box": "open",
        "key": "inside_box",
    }

    assert agent.episodes[1]["observation_before"] == {
        "box": "open",
        "key": "inside_box",
    }
    assert agent.episodes[1]["observation_after"] == {
        "box": "open",
        "key": "held",
    }

    assert integrator.recorded_step_indices == (1, 2)
    assert integrator.active is False
    assert integrator.attached is False

    # The runtime must not replace the selected action or call the
    # Agent's legacy decision-and-action lifecycle.
    assert [call for call in agent.calls if call[0] == "act"] == [
        ("act", "open_box"),
        ("act", "take_key"),
    ]


def test_end_to_end_stops_when_an_action_fails():
    (
        environment,
        agent,
        api,
        bridge,
        runtime,
        live_runtime,
        integrator,
    ) = build_pipeline()

    # Start in a state where the first planned action cannot be applied.
    environment.state["box"] = "open"

    result = integrator.execute(
        make_plan(),
        environment=environment,
        world_state=environment.observe(),
    )

    assert result.success is False
    assert result.stopped is True
    assert result.runtime_result.report.stop_reason == "step_failed"

    # Failed first action stops the runtime before step 2.
    assert [request.action for request in api.calls] == ["open_box"]
    assert result.runtime_result.report.executed_step_indices == (1,)
    assert result.runtime_result.report.failed_step_indices == (1,)

    assert len(agent.episodes) == 1
    assert agent.episodes[0]["action"] == "open_box"

    assert environment.observe() == {
        "box": "open",
        "key": "inside_box",
    }

    assert integrator.active is False
    assert integrator.attached is False
