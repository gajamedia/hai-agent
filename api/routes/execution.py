
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import logging
import re
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from agent.llm.orchestrator import LLMOrchestrationStageError
from agent.llm.planning import (
    LLMPlan,
    LLMPlanStep,
    LLMPlanParser,
    LLMPlanningParseError,
    LLMPlanningSchemaError,
)

from agent.cognition.agent_decision import AgentDecision, DecisionMode
from agent.llm.agent_canonical_single_action_api import (
    AgentCanonicalSingleActionAPI,
)
from agent.llm.execution_security import (
    ExecutionSecurityAuditError,
    ExecutionSecurityError,
    ExecutionSecurityPolicy,
    SQLiteExecutionSecurityAuditSink,
)
from agent.llm.agent_execution_boundary import AgentExecutionRequest
from agent.llm.llm_plan_agent_experience_integration import (
    LLMPlanAgentExperienceIntegrator,
)
from agent.llm.llm_plan_live_runtime import LLMPlanLiveRuntimeExecutor
from agent.llm.llm_plan_runtime_execution import (
    LLMPlanRuntimeExecutionError,
    LLMPlanRuntimeExecutor,
    LLMPlanRuntimeValidationError,
)
from agent.llm.llm_plan_runtime_feedback_consumption import (
    LLMPlanRuntimeFeedbackAwareBridge,
    LLMPlanRuntimeFeedbackConsumer,
)
from agent.llm.llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedbackBridge,
)
from agent.llm.llm_plan_single_action_bridge import (
    LLMPlanSingleActionBridge,
)

from api.dependencies import get_runtime
from api.schemas import ExecutionRunRequest
from agent.llm.goal_feasibility import assess_goal_feasibility
from api.plan_run_idempotency import (
    PlanRunIdempotencyConflict,
    PlanRunIdempotencyInProgress,
    PlanRunIdempotencyStore,
)


router = APIRouter(prefix="/execution", tags=["execution"])
logger = logging.getLogger(__name__)


def _get_runtime(request: Request) -> Any:
    runtime = getattr(request.app.state, "execution_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "execution_runtime_unavailable",
                "message": "The simulation execution runtime is not available.",
            },
        )
    return runtime


def _decision_factory(request: Any) -> AgentDecision:
    """Create the existing bounded decision for a validated simulation step."""
    return AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason=(
            "LLM plan step passed validation and is executing "
            "inside the isolated RoomEnvironment simulation."
        ),
        uncertainty=0.1,
        expected_information_gain=0.0,
        risk=0.1,
    )



# This registry is intentionally limited to actions in the isolated RoomEnvironment
# simulation. It is application-owned configuration, not inferred from LLM metadata.
# Unknown or newly added actions remain high-risk under ExecutionSecurityPolicy and
# require a trusted approval checker before they can execute.
_SIMULATION_ACTION_RISKS: dict[str, float] = {
    "open_box": 0.10,
    "take_key": 0.10,
    "find_spare_key": 0.10,
    "unlock_door": 0.10,
}


def _simulation_action_catalog(runtime: Any) -> list[dict[str, Any]]:
    """Return action metadata from the trusted simulation registry only."""
    registered_actions = getattr(getattr(runtime, "agent", None), "actions", {})
    catalog: list[dict[str, Any]] = []
    for name in sorted(_registered_action_names(runtime)):
        action = (
            registered_actions.get(name)
            if isinstance(registered_actions, Mapping)
            else None
        )
        catalog.append({
            "id": name,
            "description": str(getattr(action, "description", "") or ""),
            "preconditions": dict(getattr(action, "preconditions", {}) or {}),
            "effects": dict(getattr(action, "effects", {}) or {}),
        })
    return catalog


def _registered_action_names(runtime: Any) -> set[str]:
    """Read action names from the runtime's trusted action allowlist."""
    actions = getattr(runtime, "allowed_actions", ())
    if isinstance(actions, Mapping):
        items = actions.keys()
    else:
        items = actions or ()

    names: set[str] = set()
    try:
        for item in items:
            name = str(getattr(item, "name", item)).strip()
            if name:
                names.add(name)
    except TypeError:
        return set()
    return names


def _build_simulation_security_policy(
    request: Request,
    runtime: Any,
) -> ExecutionSecurityPolicy:
    """Build an explicit low-risk policy for the isolated room simulation.

    The SQLite audit sink is app-scoped and persistent across requests. If the
    audit database cannot be initialized or written, authorization fails closed.
    """
    sink = getattr(request.app.state, "execution_security_audit_sink", None)
    if sink is None:
        configured_path = os.environ.get("HAI_EXECUTION_SECURITY_AUDIT_DB")
        audit_path = (
            Path(configured_path).expanduser()
            if configured_path and configured_path.strip()
            else Path(__file__).resolve().parents[2]
            / "data"
            / "execution_security_audit.sqlite3"
        )
        sink = SQLiteExecutionSecurityAuditSink(audit_path)
        request.app.state.execution_security_audit_sink = sink

    allowed_names = _registered_action_names(runtime)
    action_risks = {
        action: risk
        for action, risk in _SIMULATION_ACTION_RISKS.items()
        if action in allowed_names
    }
    return ExecutionSecurityPolicy(
        action_risks=action_risks,
        audit_sink=sink,
    )

def _goal_satisfied(
    state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> bool:
    return bool(target_state) and all(
        state.get(key) == expected
        for key, expected in target_state.items()
    )


# Explicit user assertions must not be silently overridden by a contradictory
# simulation snapshot. This guard is intentionally narrow: it recognizes clear
# statements that the spare key is unavailable, and prevents the registered
# search action from being used to contradict that assertion. It does not mutate
# the trusted world state or invent an alternative action.
def _explicitly_states_spare_key_unavailable(goal_description: str) -> bool:
    text = " ".join(goal_description.casefold().split())
    patterns = (
        r"\bkunci cadangan\s+(?:itu\s+)?(?:tidak|tak|nggak|gak)\s+(?:tersedia|ada|ditemukan|diketemukan)\b",
        r"\bkunci cadangan\s+(?:itu\s+)?(?:belum|sudah tidak|memang tidak)\s+(?:tersedia|ada)\b",
        r"\bkunci cadangan\s+(?:tidak ada|tidak tersedia|tidak ditemukan)\b",
        r"\b(?:spare key|backup key)\s+(?:is\s+)?(?:unavailable|not available|missing|does not exist|isn't available|is not available)\b",
        r"\b(?:there is no|there isn't a|there is not a)\s+(?:available\s+)?(?:spare|backup) key\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _reject_plan_that_violates_explicit_constraints(
    goal_description: str,
    actions: Any,
) -> None:
    if not _explicitly_states_spare_key_unavailable(goal_description):
        return

    if isinstance(actions, (str, bytes)) or actions is None:
        return

    try:
        action_names = [
            str(getattr(action, "action", action)).strip()
            for action in actions
        ]
    except TypeError:
        return

    if "find_spare_key" in action_names:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "explicit_constraint_conflict",
                "stage": "plan_validation",
                "message": (
                    "Instruksi menyatakan kunci cadangan tidak tersedia. "
                    "Rencana yang mencari kunci cadangan ditolak; agent tidak "
                    "boleh menganggap kunci tersedia hanya karena state simulasi "
                    "menyatakan sebaliknya."
                ),
                "conflicting_action": "find_spare_key",
                "goal_achieved": False,
                "execution_started": False,
            },
        )


def _read_value(source: Any, *names: str) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return None

    for name in names:
        if hasattr(source, name):
            return getattr(source, name)

    return None


def _state_progress(
    before: Any,
    after: Any,
    target_state: Mapping[str, Any],
) -> float | None:
    """Calculate target-state progress only from observed mappings."""
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None

    if not target_state:
        return None

    before_matches = sum(
        before.get(key) == expected
        for key, expected in target_state.items()
    )
    after_matches = sum(
        after.get(key) == expected
        for key, expected in target_state.items()
    )

    count = len(target_state)
    return (after_matches - before_matches) / count


def _build_reflection_provider(
    target_state: Mapping[str, Any],
):
    """
    Build deterministic reflection from recorded execution evidence.

    This is not an LLM-generated reflection. It only uses explicit execution
    results and observed before/after state. Unknown success remains unknown.
    """
    def reflection_provider(episode: Any) -> dict[str, Any]:
        before = _read_value(
            episode,
            "observation_before",
            "state_before",
        )
        after = _read_value(
            episode,
            "observation_after",
            "observation",
            "state_after",
        )
        result = _read_value(episode, "result")

        explicit_success = _read_value(result, "success")
        if explicit_success is None:
            explicit_success = _read_value(episode, "success")

        progress_delta = _state_progress(
            before,
            after,
            target_state,
        )

        if explicit_success is not None:
            success = bool(explicit_success)
            evidence_quality = 1.0
            confidence = 0.95
        elif progress_delta is not None and progress_delta > 0:
            success = True
            evidence_quality = 0.8
            confidence = 0.75
        else:
            success = None
            evidence_quality = 0.4
            confidence = 0.4

        failed = success is False
        no_progress = (
            progress_delta is not None and progress_delta <= 0
        )

        if failed:
            diagnosis = "observed_execution_failure"
            lesson = "The executed action reported failure."
            next_step = "Stop and reassess the plan before another action."
        elif no_progress:
            diagnosis = "no_observed_target_progress"
            lesson = "The observed state did not advance the target."
            next_step = "Reassess whether another action is appropriate."
        elif success is True:
            diagnosis = "successful_outcome_with_observed_progress"
            lesson = "The action produced observed target-state progress."
            next_step = "Continue with the next validated plan step."
        else:
            diagnosis = "insufficient_execution_evidence"
            lesson = "The available evidence does not establish success."
            next_step = "Avoid assuming success without stronger evidence."

        return {
            "success": success,
            "progress_delta": progress_delta,
            "evidence_quality": evidence_quality,
            "confidence": confidence,
            "diagnosis": diagnosis,
            "lesson": lesson,
            "next_step": next_step,
            "change_required": failed,
            "continue_recommended": not failed,
            "exploration_recommended": no_progress or failed,
            "risk_level": (
                "high" if failed
                else "medium" if no_progress or success is None
                else "low"
            ),
        }

    return reflection_provider


@router.get("/state")
def get_execution_state(request: Request) -> dict[str, Any]:
    """Read the current simulation state without executing actions."""
    runtime = _get_runtime(request)
    return {
        "status": "ok",
        "state": runtime.get_state(),
        "allowed_actions": list(runtime.allowed_actions),
        "action_catalog": _simulation_action_catalog(runtime),
    }


@router.put("/state")
def set_execution_state(
    payload: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    """Set a validated initial state for the isolated Room sandbox."""
    runtime = _get_runtime(request)
    try:
        state = runtime.set_state(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_world_state",
                "message": str(exc),
            },
        ) from exc

    return {
        "status": "ok",
        "message": "Sandbox world state updated. No agent action was executed.",
        "state": state,
        "allowed_actions": list(runtime.allowed_actions),
        "action_catalog": _simulation_action_catalog(runtime),
    }


@router.post("/run")
def run_execution(
    payload: ExecutionRunRequest,
    request: Request,
) -> dict[str, Any]:
    """Execute one allowlisted action or an explicit ordered plan."""
    runtime = _get_runtime(request)

    try:
        # Authorize every requested step before any action can have a side effect.
        # The policy and durable audit sink are shared with /plan-run.
        with runtime.lock:
            # Preserve the endpoint's existing validation contract: malformed
            # or unsupported requests are 422, not authorization denials.
            if not payload.actions:
                raise ValueError("At least one action is required.")
            if len(payload.actions) > len(runtime.allowed_actions):
                raise ValueError("Too many actions were supplied.")
            unknown_actions = [
                action for action in payload.actions
                if action not in runtime.allowed_actions
            ]
            if unknown_actions:
                raise ValueError(
                    f"Unsupported action(s): {', '.join(unknown_actions)}"
                )
            if not payload.target_state:
                raise ValueError("target_state must not be empty.")

            security_policy = _build_simulation_security_policy(request, runtime)
            security_requests = [
                AgentExecutionRequest(
                    action=action,
                    step={"action": action, "source": "execution_run"},
                    position=index,
                    metadata={
                        "source": "execution_run",
                        "execution_mode": "isolated_simulation",
                    },
                )
                for index, action in enumerate(payload.actions, start=1)
            ]
            for security_request in security_requests:
                security_policy.authorize(security_request)

            result = runtime.run(
                actions=payload.actions,
                goal_description=payload.goal_description,
                target_state=payload.target_state,
            )

            # Record outcomes only after runtime.run returns, using per-step
            # evidence when available. If result auditing fails, report that
            # execution may already have happened; clients must not blindly retry.
            step_results = result.get("steps", [])
            for index, security_request in enumerate(security_requests, start=1):
                step_success = None
                reason = "execution_outcome_unknown"

                if index <= len(step_results):
                    step = step_results[index - 1]
                    if (
                        isinstance(step, dict)
                        and step.get("action") == security_request.action
                        and isinstance(step.get("success"), bool)
                    ):
                        step_success = step["success"]
                        reason = "execution_run_result"

                elif (
                    len(security_requests) == 1
                    and not step_results
                    and isinstance(result.get("success"), bool)
                ):
                    step_success = result["success"]
                    reason = "execution_run_result"

                security_policy.record_execution_outcome(
                    security_request,
                    success=step_success,
                    reason=reason,
                )
            return result
    except ExecutionSecurityError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "execution_not_authorized",
                "message": str(exc),
            },
        ) from exc
    except ExecutionSecurityAuditError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "execution_audit_unavailable",
                "message": str(exc),
                "retry_safe": False,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_execution_request",
                "message": str(exc),
            },
        ) from exc
    except (AttributeError, RuntimeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "execution_failed",
                "message": "The simulation execution could not be completed.",
            },
        ) from exc


def _run_cognitive_plan_impl(
    request: Request,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a supplied Cognitive Pass plan, or generate a legacy plan when none is supplied.

    Only actions registered with the simulation Agent can be executed.
    Experience is recorded per step, and feedback is consumed before the
    next step. The target state is verified against the actual environment.
    """
    cognitive_runtime = get_runtime(request)
    execution_runtime = _get_runtime(request)

    observation = payload.get("observation")
    goal_description = payload.get("goal_description")
    target_state = payload.get("target_state")

    if observation is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "observation_required",
                "message": "observation is required.",
            },
        )

    if not isinstance(goal_description, str) or not goal_description.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "goal_description_required",
                "message": "goal_description must be a non-empty string.",
            },
        )

    if not isinstance(target_state, dict) or not target_state:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "target_state_required",
                "message": "target_state must be a non-empty object.",
            },
        )

    try:
        with execution_runtime.lock:
            state_before = dict(execution_runtime.environment.observe())

            # Build the planner catalog only from actions registered by the
            # trusted simulation runtime. Never accept this catalog from payload.
            registered_names = _registered_action_names(execution_runtime)
            registered_actions = getattr(
                getattr(execution_runtime, "agent", None),
                "actions",
                {},
            )
            action_catalog = _simulation_action_catalog(execution_runtime)

            # Assess feasibility using trusted runtime state and action catalog.
            # An inconclusive result must not be treated as unreachable.
            if action_catalog:
                feasibility = assess_goal_feasibility(
                    initial_state=state_before,
                    goal=target_state,
                    action_catalog=action_catalog,
                )

                if feasibility.status == "unreachable":
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "goal_unreachable",
                            "stage": "feasibility_assessment",
                            "message": (
                                "The goal is unreachable using the supplied "
                                "trusted action catalog."
                            ),
                            "assessment": feasibility.to_dict(),
                            "goal_achieved": False,
                            "execution_started": False,
                        },
                    )

            # Execution Workspace sends the user's explicitly selected simulation
            # action IDs. Cognitive Pass prose is not itself an executable action
            # plan: its steps may describe reasoning that is not registered in the
            # simulation runtime. Convert only selected, runtime-registered IDs
            # into a canonical plan; all metadata comes from trusted runtime objects.
            if "actions" in payload:
                requested_actions = payload.get("actions")
                if (
                    not isinstance(requested_actions, list)
                    or not requested_actions
                    or any(not isinstance(action, str) or not action.strip() for action in requested_actions)
                ):
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "invalid_action_selection",
                            "message": "actions must be a non-empty array of action IDs.",
                            "execution_started": False,
                        },
                    )

                normalized_actions = [action.strip() for action in requested_actions]
                if len(set(normalized_actions)) != len(normalized_actions):
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "invalid_action_selection",
                            "message": "Duplicate action IDs are not allowed.",
                            "execution_started": False,
                        },
                    )

                unknown_actions = [action for action in normalized_actions if action not in registered_names]
                if unknown_actions:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "action_not_available",
                            "message": "Selected actions are not registered: " + ", ".join(unknown_actions),
                            "execution_started": False,
                        },
                    )

                executable_steps = []
                for index, action_name in enumerate(normalized_actions, start=1):
                    action_object = (
                        registered_actions.get(action_name)
                        if isinstance(registered_actions, Mapping)
                        else None
                    )
                    description = str(getattr(action_object, "description", "") or action_name)
                    effects = getattr(action_object, "effects", {}) or {}
                    success_criteria = (
                        "; ".join(f"{key}={value}" for key, value in effects.items())
                        if isinstance(effects, Mapping) and effects
                        else f"Action {action_name} succeeds."
                    )
                    preconditions = getattr(action_object, "preconditions", {}) or {}
                    precondition_text = (
                        tuple(f"{key}={value}" for key, value in preconditions.items())
                        if isinstance(preconditions, Mapping)
                        else ()
                    )
                    executable_steps.append(
                        LLMPlanStep(
                            index=index,
                            action=action_name,
                            purpose=description,
                            preconditions=precondition_text,
                            expected_observation=success_criteria,
                            success_criteria=success_criteria,
                            risk="low",
                        )
                    )

                plan = LLMPlan(
                    goal=goal_description.strip(),
                    rationale=(
                        "Executable simulation plan built from action IDs explicitly "
                        "selected in Execution Workspace. Cognitive Pass reasoning "
                        "is contextual and is not treated as runtime actions."
                    ),
                    steps=tuple(executable_steps),
                    assumptions=(),
                    constraints=("Only runtime-registered actions may execute.",),
                    stop_conditions=("Stop when an action fails or the goal is satisfied.",),
                    confidence=1.0,
                    provider="execution_workspace",
                    model=None,
                )
                plan_source = "validated_action_selection"
            elif "plan" in payload:
                supplied_plan = payload.get("plan")
                if not isinstance(supplied_plan, Mapping):
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "invalid_supplied_plan",
                            "message": "plan must be a JSON object.",
                            "execution_started": False,
                        },
                    )

                try:
                    plan = LLMPlanParser.parse(
                        json.dumps(dict(supplied_plan)),
                        provider=supplied_plan.get("provider"),
                        model=supplied_plan.get("model"),
                    )
                except (LLMPlanningParseError, LLMPlanningSchemaError, TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "invalid_supplied_plan",
                            "message": str(exc),
                            "execution_started": False,
                        },
                    ) from exc

                plan_source = "cognitive_pass_handoff"
            else:
                # Backward compatibility for existing direct API callers.
                # The response explicitly identifies this path so replanning
                # cannot happen silently.
                cognitive_result = cognitive_runtime.run(
                    observation,
                    goal=goal_description,
                    state=state_before,
                    metadata={
                        "source": "execution_plan_run",
                        "execution_mode": "isolated_simulation",
                        "action_catalog": action_catalog,
                        "initial_state": dict(state_before),
                    },
                )
                plan = cognitive_result.plan
                plan_source = "runtime_generated_legacy"

            # Validate explicit negative facts from the user's instruction before
            # the execution boundary. A contradiction must stop before any action
            # executes, even if the simulation snapshot says the resource exists.
            _reject_plan_that_violates_explicit_constraints(
                goal_description,
                plan.steps,
            )

            # The feedback state is request-local; no stale feedback leaks
            # into a later plan execution.
            consumer = LLMPlanRuntimeFeedbackConsumer()
            learning_bridge = LLMPlanRuntimeLearningFeedbackBridge(
                reflection_provider=_build_reflection_provider(target_state),
                feedback_observer=consumer.consume,
            )

            security_policy = _build_simulation_security_policy(
                request,
                execution_runtime,
            )
            canonical_api = AgentCanonicalSingleActionAPI(
                execution_runtime.agent,
                decision_factory=_decision_factory,
                security_policy=security_policy,
            )
            single_action_bridge = LLMPlanSingleActionBridge(canonical_api)
            feedback_aware_bridge = LLMPlanRuntimeFeedbackAwareBridge(
                single_action_bridge,
                consumer,
            )

            runtime_executor = LLMPlanRuntimeExecutor(
                feedback_aware_bridge,
                stop_on_step_failure=True,
                raise_on_step_error=False,
                world_state_observer=lambda environment: dict(
                    environment.observe()
                ),
                step_success_evaluator=lambda step, result, world_state: bool(
                    getattr(result, "success", False)
                ),
                goal_satisfaction_evaluator=lambda goal, world_state, context: (
                    _goal_satisfied(world_state, target_state)
                ),
            )
            live_runtime = LLMPlanLiveRuntimeExecutor(runtime_executor)

            experience_integrator = LLMPlanAgentExperienceIntegrator(
                execution_runtime.agent,
                live_runtime,
                experience_observer=learning_bridge.observe_experience,
                project_agent_action=True,
                project_agent_observation=True,
                advance_agent_cycle=False,
            )

            # Plan execution, experience recording and feedback consumption
            # happen synchronously in the same runtime event sequence.
            integration_result = experience_integrator.execute(
                plan,
                goal=goal_description,
                world_state=state_before,
                environment=execution_runtime.environment,
                available_actions=execution_runtime.agent.actions,
                metadata={
                    "source": "execution_plan_run",
                    "execution_mode": "isolated_simulation",
                },
            )

            report = integration_result.runtime_result.report
            state_after = dict(execution_runtime.environment.observe())
            goal_achieved = _goal_satisfied(state_after, target_state)

            telemetry = getattr(
                execution_runtime.agent,
                "record_capability_telemetry",
                None,
            )
            if callable(telemetry):
                telemetry(
                    "execution",
                    bool(report.records) and all(
                        record.success and not record.skipped
                        for record in report.records
                    ),
                    source="cognitive_plan_execution_api",
                )
                telemetry(
                    "goal_achievement",
                    goal_achieved,
                    source="cognitive_plan_execution_api",
                )

            experiences = [
                {
                    "step_index": item.step_index,
                    "action": item.action,
                    "success": item.success,
                    "observation_before": item.observation_before,
                    "observation_after": item.observation_after,
                }
                for item in integration_result.experiences
            ]

            feedback = [
                {
                    "step_index": item.step_index,
                    "action": item.action,
                    "success": item.success,
                    "progress_delta": item.progress_delta,
                    "diagnosis": item.diagnosis,
                    "lesson": item.lesson,
                    "next_step": item.next_step,
                    "continue_recommended": item.continue_recommended,
                    "exploration_recommended": item.exploration_recommended,
                    "risk_level": item.risk_level,
                    "change_required": item.change_required,
                }
                for item in learning_bridge.feedback_history
            ]

            return {
                "status": "ok",
                "mode": "cognitive_plan_execution",
                "plan": plan.to_dict(),
                "plan_source": plan_source,
                "execution": {
                    # Keep action execution status distinct from goal
                    # achievement. A plan can execute its attempted actions
                    # successfully while the requested target remains unmet.
                    "success": bool(report.records) and all(
                        record.success and not record.skipped
                        for record in report.records
                    ),
                    "completed": report.completed,
                    "stopped": report.stopped,
                    "stop_reason": report.stop_reason,
                    "planned_step_count": len(plan.steps),
                    "executed_step_count": len(report.executed_step_indices),
                    "steps": [
                        {
                            "index": record.step.index,
                            "action": record.step.action,
                            "success": record.success,
                            "skipped": record.skipped,
                            "stop_requested": record.stop_requested,
                            "error": record.error,
                            "observation": record.observation,
                        }
                        for record in report.records
                    ],
                },
                "experience": {
                    "count": len(experiences),
                    "steps": experiences,
                },
                "runtime_feedback": {
                    "count": len(feedback),
                    "items": feedback,
                    "decisions": [
                        {
                            "source_step_index": item.source_step_index,
                            "target_step_index": item.target_step_index,
                            "should_continue": item.should_continue,
                            "should_stop": item.should_stop,
                            "reason": item.reason,
                        }
                        for item in consumer.decision_history
                    ],
                },
                "goal": {
                    "description": goal_description,
                    "target_state": target_state,
                    "satisfied": goal_achieved,
                },
                "state_before": state_before,
                "state_after": state_after,
                "success": bool(report.success and goal_achieved),
                "audit": {
                    "source": "execution_plan_run",
                    "execution_mode": "isolated_simulation",
                    "plan_source": plan_source,
                    "actions": list(plan.actions),
                    "experience_count": len(experiences),
                    "feedback_count": len(feedback),
                },
            }

    except LLMPlanRuntimeValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "plan_validation_failed",
                "message": str(exc),
            },
        ) from exc
    except LLMPlanRuntimeExecutionError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "plan_execution_rejected",
                "message": str(exc),
            },
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_plan_execution",
                "message": str(exc),
            },
        ) from exc
    except HTTPException:
        raise
    except LLMOrchestrationStageError as exc:
        if (
            exc.stage == "planning"
            and isinstance(exc.cause, LLMPlanningSchemaError)
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "planning_failed",
                    "stage": "planning",
                    "message": (
                        "No executable plan could be produced "
                        "for the current world state."
                    ),
                    "reason": str(exc.cause),
                    "goal_achieved": False,
                    "execution_started": False,
                },
            ) from exc
        raise
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception(
            "Unexpected error in POST /execution/plan-run"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "cognitive_plan_execution_failed",
                "message": (
                    "The cognitive plan execution could not be completed."
                ),
            },
        ) from exc


@router.post("/plan-run")
def run_cognitive_plan(
    request: Request,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Execute a plan, optionally with durable HTTP request idempotency.

    Clients opt in with the ``Idempotency-Key`` header. Completed responses are
    replayed from a persistent SQLite ledger. If a prior execution was interrupted
    after it may have started, the key is blocked rather than executing the plan
    again; that outcome requires reconciliation.
    """
    raw_key = request.headers.get("Idempotency-Key")
    if raw_key is None:
        return _run_cognitive_plan_impl(request, payload)

    key = raw_key.strip()
    if not key or len(key) > 255:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_idempotency_key",
                "message": "Idempotency-Key must contain 1 to 255 non-whitespace characters.",
            },
        )

    # Validate required request fields before reserving a key. Invalid requests
    # that cannot execute must not leave a durable reservation behind.
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail={"error": "invalid_request", "message": "Request body must be a JSON object."})
    if payload.get("observation") is None:
        raise HTTPException(status_code=422, detail={"error": "observation_required", "message": "observation is required."})
    goal_description = payload.get("goal_description")
    if not isinstance(goal_description, str) or not goal_description.strip():
        raise HTTPException(status_code=422, detail={"error": "goal_description_required", "message": "goal_description must be a non-empty string."})
    target_state = payload.get("target_state")
    if not isinstance(target_state, dict) or not target_state:
        raise HTTPException(status_code=422, detail={"error": "target_state_required", "message": "target_state must be a non-empty object."})

    try:
        canonical_request = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_request_json", "message": "Request body must contain JSON-compatible values."},
        ) from exc
    fingerprint = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()

    store = getattr(request.app.state, "plan_run_idempotency_store", None)
    if store is None:
        configured_path = os.environ.get("HAI_PLAN_RUN_IDEMPOTENCY_DB")
        database_path = (
            Path(configured_path).expanduser()
            if configured_path and configured_path.strip()
            else Path(__file__).resolve().parents[2] / "data" / "execution_plan_run_idempotency.sqlite3"
        )
        store = PlanRunIdempotencyStore(database_path)
        request.app.state.plan_run_idempotency_store = store

    try:
        claim = store.claim(idempotency_key=key, request_fingerprint=fingerprint)
    except PlanRunIdempotencyConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "idempotency_key_conflict", "message": str(exc), "retry_safe": False},
        ) from exc
    except PlanRunIdempotencyInProgress as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "idempotency_outcome_unknown", "message": str(exc), "retry_safe": False, "reconciliation_required": True},
        ) from exc

    if claim.status == "completed":
        result = dict(claim.response or {})
        result["idempotency"] = {"key": key, "replayed": True}
        return result
    if claim.status == "rejected":
        stored_error = dict(claim.response or {})
        stored_detail = stored_error.get("detail", {"error": "request_rejected"})
        if isinstance(stored_detail, dict):
            stored_detail = dict(stored_detail)
            stored_detail.setdefault("idempotency", {"key": key, "replayed": True})
        raise HTTPException(
            status_code=int(stored_error.get("status_code", 422)),
            detail=stored_detail,
        )
    if claim.status == "unknown":
        raise HTTPException(
            status_code=409,
            detail={"error": "idempotency_outcome_unknown", "message": "A previous attempt may have started execution; reconcile the simulation state before using a new key.", "retry_safe": False, "reconciliation_required": True},
        )

    def mark_unknown_without_masking_original_error() -> None:
        """Best-effort ledger update that must not replace the triggering error."""
        try:
            store.mark_unknown(idempotency_key=key, request_fingerprint=fingerprint)
        except Exception:
            # If the ledger itself is unavailable, the row may remain `running`.
            # That state is deliberately fail-closed: a retry will be blocked.
            logger.exception(
                "Unable to mark plan-run idempotency key %r as unknown; "
                "the original execution/storage error will be preserved.",
                key,
            )

    try:
        result = _run_cognitive_plan_impl(request, payload)
        response = dict(result)
        response["idempotency"] = {"key": key, "replayed": False}
        store.complete(
            idempotency_key=key,
            request_fingerprint=fingerprint,
            response=response,
        )
        return response
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        # Only explicitly pre-execution errors are safely replayable as rejected.
        try:
            if detail.get("execution_started") is False:
                store.reject(
                    idempotency_key=key,
                    request_fingerprint=fingerprint,
                    status_code=exc.status_code,
                    detail=detail,
                )
            else:
                mark_unknown_without_masking_original_error()
        except Exception:
            # A failed ledger transition must not hide the API's original HTTP error.
            logger.exception(
                "Unable to persist plan-run idempotency rejection for key %r.", key
            )
        raise
    except Exception:
        # A failure after reservation is conservatively treated as ambiguous.
        # Never automatically replay a plan that may have changed world state.
        mark_unknown_without_masking_original_error()
        raise


@router.post("/reset")
def reset_execution(request: Request) -> dict[str, Any]:
    """Reset simulation state while retaining historical telemetry."""
    runtime = _get_runtime(request)
    return {
        "status": "ok",
        "message": (
            "Simulation state reset. Telemetry history was retained."
        ),
        "state": runtime.reset(),
    }

