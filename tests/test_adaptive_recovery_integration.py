
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.llm.llm_plan_adaptive_replanning import LLMPlanAdaptiveReplanner
from agent.llm.llm_plan_adaptive_runtime import LLMPlanAdaptiveRuntime
from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedback,
)
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class ActionResult:
    success: bool
    result: dict[str, Any]


class RecoveryScenarioAPI:
    """Deterministic API: first action fails, recovery action succeeds."""

    def __init__(self) -> None:
        self.agent = type(
            "AgentStub",
            (),
            {
                "actions": {
                    "open_box": object(),
                    "recover_box": object(),
                }
            },
        )()
        self.calls: list[str] = []
        self.world_state: dict[str, Any] = {
            "box": "closed",
            "goal": "pending",
        }

    def execute(self, request: Any) -> ActionResult:
        action = str(request.action)
        self.calls.append(action)

        if action == "open_box":
            return ActionResult(
                success=False,
                result={
                    "error": "The box is jammed",
                    "world_state": dict(self.world_state),
                },
            )

        if action == "recover_box":
            self.world_state = {
                "box": "open",
                "goal": "achieved",
            }
            return ActionResult(
                success=True,
                result={
                    "observation": dict(self.world_state),
                    "world_state": dict(self.world_state),
                },
            )

        return ActionResult(
            success=False,
            result={"error": f"Unsupported action: {action}"},
        )


class StepFeedbackSource:
    """Provides feedback for the failed first step."""

    def __init__(self) -> None:
        self.by_step = {
            1: LLMPlanRuntimeLearningFeedback(
                step_index=1,
                action="open_box",
                success=False,
                progress_delta=0.0,
                evidence_quality=1.0,
                confidence=1.0,
                diagnosis="The box is jammed",
                lesson="Use the recovery action instead",
                next_step="recover_box",
                change_required=True,
                continue_recommended=False,
                exploration_recommended=False,
                risk_level="low",
                risk_score=0.25,
                information_gain=0.1,
            )
        }

    def get_feedback(
        self,
        step_index: int,
    ) -> LLMPlanRuntimeLearningFeedback | None:
        return self.by_step.get(step_index)


def make_plan(
    *,
    goal: str,
    action: str,
    index: int = 1,
    purpose: str | None = None,
) -> LLMPlan:
    return LLMPlan(
        goal=goal,
        rationale="Adaptive recovery integration test",
        steps=(
            LLMPlanStep(
                index=index,
                action=action,
                purpose=purpose or action,
                expected_observation=(
                    "goal achieved" if action == "recover_box"
                    else "box opened"
                ),
                success_criteria=(
                    "goal achieved" if action == "recover_box"
                    else "box opened"
                ),
                risk="low",
            ),
        ),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.9,
        provider="test",
        model="deterministic-test",
    )


def test_failed_action_triggers_replanning_and_executes_replacement_plan():
    api = RecoveryScenarioAPI()
    bridge = LLMPlanSingleActionBridge.for_api(api)

    runtime = LLMPlanRuntimeExecutor.for_bridge(
        bridge,
        goal_satisfaction_evaluator=(
            lambda goal, world_state, context:
            isinstance(world_state, dict)
            and world_state.get("goal") == "achieved"
        ),
    )

    initial_plan = make_plan(
        goal="Open the box",
        action="open_box",
        index=1,
    )
    replacement_calls: list[Any] = []

    def replacement_provider(request: Any) -> LLMPlan:
        replacement_calls.append(request)
        return make_plan(
            goal=request.goal,
            action="recover_box",
            index=2,
            purpose="Recover the jammed box",
        )

    replanner = LLMPlanAdaptiveReplanner(
        replacement_provider,
        max_replans=1,
    )
    controller = LLMPlanAdaptiveRuntime(
        runtime,
        replanner,
        feedback_source=StepFeedbackSource(),
        max_segments=2,
    )

    result = controller.execute(
        initial_plan,
        available_actions=("open_box", "recover_box"),
        world_state=dict(api.world_state),
    )

    # The initial action failed and the replacement action actually ran.
    assert api.calls == ["open_box", "recover_box"]

    # The provider was called and a replacement plan was accepted.
    assert len(replacement_calls) == 1
    assert result.replanned is True
    assert result.segment_count == 2
    assert result.generation_count == 2
    assert result.final_plan.steps[0].action == "recover_box"

    # The final state must reflect the recovery action.
    assert result.final_world_state["box"] == "open"
    assert result.final_world_state["goal"] == "achieved"

    # The report must not claim success unless its execution result says so.
    assert result.success is True
    assert result.failed_step_indices == (1,)
    assert result.completed_successful_step_indices == (2,)
