from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Mapping

from .llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedback,
)


class LLMPlanRuntimeFeedbackConsumptionError(RuntimeError):
    """Base error for runtime feedback consumption."""


class LLMPlanRuntimeFeedbackConsumptionConfigurationError(
    LLMPlanRuntimeFeedbackConsumptionError
):
    """Raised when feedback consumption is misconfigured."""


class LLMPlanRuntimeFeedbackConsumptionOrderError(
    LLMPlanRuntimeFeedbackConsumptionError
):
    """Raised when feedback is consumed out of runtime order."""


@dataclass(frozen=True)
class LLMPlanRuntimeFeedbackDecision:
    """Bounded runtime decision derived from the latest feedback."""

    source_step_index: int | None
    target_step_index: int
    action: str | None
    should_continue: bool
    should_stop: bool
    increase_exploration: bool
    reduce_risk: bool
    exploration_risk: float
    action_risk_multiplier: float
    information_gain_multiplier: float
    reason: str
    feedback: LLMPlanRuntimeLearningFeedback | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMPlanRuntimeFeedbackApplication:
    """Execution inputs after applying runtime feedback guidance."""

    target_step_index: int
    exploration_risk: float
    action_risk: Mapping[str, float]
    information_gain: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decision: LLMPlanRuntimeFeedbackDecision | None = None


def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, numeric))


class LLMPlanRuntimeFeedbackConsumer:
    """
    Consume Stage 24.11 runtime-learning feedback before the next step.

    The consumer is deliberately side-effect free with respect to Agent.
    It only turns immutable feedback into bounded runtime guidance.
    """

    def __init__(
        self,
        *,
        exploration_risk_increment: float = 0.20,
        exploration_information_gain_multiplier: float = 1.10,
        reduced_action_risk_multiplier: float = 0.75,
        stop_on_feedback: bool = True,
    ) -> None:
        if exploration_risk_increment < 0.0:
            raise LLMPlanRuntimeFeedbackConsumptionConfigurationError(
                "exploration_risk_increment cannot be negative."
            )
        if exploration_information_gain_multiplier <= 0.0:
            raise LLMPlanRuntimeFeedbackConsumptionConfigurationError(
                "exploration_information_gain_multiplier must be > 0."
            )
        if not 0.0 < reduced_action_risk_multiplier <= 1.0:
            raise LLMPlanRuntimeFeedbackConsumptionConfigurationError(
                "reduced_action_risk_multiplier must be in (0, 1]."
            )

        self.exploration_risk_increment = float(exploration_risk_increment)
        self.exploration_information_gain_multiplier = float(
            exploration_information_gain_multiplier
        )
        self.reduced_action_risk_multiplier = float(
            reduced_action_risk_multiplier
        )
        self.stop_on_feedback = bool(stop_on_feedback)

        self._lock = RLock()
        self._latest_feedback: LLMPlanRuntimeLearningFeedback | None = None
        self._decision_history: list[LLMPlanRuntimeFeedbackDecision] = []
        self._consumed_steps: set[int] = set()
        self._last_decision: LLMPlanRuntimeFeedbackDecision | None = None

    @property
    def latest_feedback(self) -> LLMPlanRuntimeLearningFeedback | None:
        return self._latest_feedback

    @property
    def last_decision(self) -> LLMPlanRuntimeFeedbackDecision | None:
        return self._last_decision

    @property
    def decision_history(self) -> tuple[LLMPlanRuntimeFeedbackDecision, ...]:
        return tuple(self._decision_history)

    @property
    def consumed_steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._consumed_steps))

    def consume(self, feedback: LLMPlanRuntimeLearningFeedback) -> LLMPlanRuntimeFeedbackDecision:
        if feedback is None:
            raise LLMPlanRuntimeFeedbackConsumptionError(
                "feedback cannot be None."
            )

        with self._lock:
            self._latest_feedback = feedback
            return self._decision_for_feedback(
                feedback,
                target_step_index=feedback.step_index + 1,
            )

    def _decision_for_feedback(
        self,
        feedback: LLMPlanRuntimeLearningFeedback,
        *,
        target_step_index: int,
    ) -> LLMPlanRuntimeFeedbackDecision:
        should_stop = bool(
            self.stop_on_feedback
            and (
                not feedback.continue_recommended
                or feedback.change_required
            )
        )
        should_continue = not should_stop
        increase_exploration = bool(
            should_continue and feedback.exploration_recommended
        )
        reduce_risk = bool(
            should_continue and feedback.risk_level in {"medium", "high"}
        )

        exploration_risk = 0.0
        action_risk_multiplier = 1.0
        information_gain_multiplier = 1.0

        if increase_exploration:
            exploration_risk = _clamp(
                self.exploration_risk_increment
            )
            information_gain_multiplier = (
                self.exploration_information_gain_multiplier
            )
        if reduce_risk:
            action_risk_multiplier = self.reduced_action_risk_multiplier

        if should_stop:
            reason = "runtime_feedback_stop"
        elif increase_exploration and reduce_risk:
            reason = "runtime_feedback_explore_with_reduced_risk"
        elif increase_exploration:
            reason = "runtime_feedback_increase_exploration"
        elif reduce_risk:
            reason = "runtime_feedback_reduce_risk"
        else:
            reason = "runtime_feedback_continue"

        decision = LLMPlanRuntimeFeedbackDecision(
            source_step_index=feedback.step_index,
            target_step_index=int(target_step_index),
            action=feedback.action,
            should_continue=should_continue,
            should_stop=should_stop,
            increase_exploration=increase_exploration,
            reduce_risk=reduce_risk,
            exploration_risk=exploration_risk,
            action_risk_multiplier=action_risk_multiplier,
            information_gain_multiplier=information_gain_multiplier,
            reason=reason,
            feedback=feedback,
            metadata={
                "feedback_step_index": feedback.step_index,
                "target_step_index": int(target_step_index),
                "feedback_risk_level": feedback.risk_level,
                "feedback_diagnosis": feedback.diagnosis,
            },
        )
        self._last_decision = decision
        self._decision_history.append(decision)
        return decision

    def before_step(self, step_index: int) -> LLMPlanRuntimeFeedbackDecision:
        """Consume the latest feedback strictly before the next step."""
        with self._lock:
            if step_index < 0:
                raise LLMPlanRuntimeFeedbackConsumptionOrderError(
                    "step_index cannot be negative."
                )

            feedback = self._latest_feedback
            if feedback is None:
                decision = LLMPlanRuntimeFeedbackDecision(
                    source_step_index=None,
                    target_step_index=step_index,
                    action=None,
                    should_continue=True,
                    should_stop=False,
                    increase_exploration=False,
                    reduce_risk=False,
                    exploration_risk=0.0,
                    action_risk_multiplier=1.0,
                    information_gain_multiplier=1.0,
                    reason="runtime_feedback_neutral",
                    metadata={"feedback_present": False},
                )
                self._last_decision = decision
                self._decision_history.append(decision)
                return decision

            if feedback.step_index >= step_index:
                raise LLMPlanRuntimeFeedbackConsumptionOrderError(
                    "Feedback must originate from an earlier runtime step."
                )

            if feedback.step_index != step_index - 1:
                raise LLMPlanRuntimeFeedbackConsumptionOrderError(
                    "Feedback consumption requires the immediately preceding step."
                )

            if step_index in self._consumed_steps:
                raise LLMPlanRuntimeFeedbackConsumptionOrderError(
                    f"Feedback for target step {step_index} has already been consumed."
                )

            self._consumed_steps.add(step_index)
            return self._decision_for_feedback(
                feedback,
                target_step_index=step_index,
            )

    def apply_to_execution_inputs(
        self,
        step_index: int,
        *,
        exploration_risk: float,
        action_risk: Mapping[str, float] | None = None,
        information_gain: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlanRuntimeFeedbackApplication:
        decision = self.before_step(step_index)

        effective_exploration_risk = _clamp(exploration_risk)
        effective_action_risk = dict(action_risk or {})
        effective_information_gain = dict(information_gain or {})

        if decision.should_stop:
            return LLMPlanRuntimeFeedbackApplication(
                target_step_index=step_index,
                exploration_risk=effective_exploration_risk,
                action_risk=effective_action_risk,
                information_gain=effective_information_gain,
                metadata={
                    **dict(metadata or {}),
                    "runtime_feedback_reason": decision.reason,
                    "runtime_feedback_stop": True,
                },
                decision=decision,
            )

        if decision.increase_exploration:
            effective_exploration_risk = _clamp(
                effective_exploration_risk + decision.exploration_risk
            )
            if effective_information_gain:
                effective_information_gain = {
                    key: round(
                        _clamp(
                            value * decision.information_gain_multiplier
                        ),
                        12,
                    )
                    for key, value in effective_information_gain.items()
                }

        if decision.reduce_risk and effective_action_risk:
            effective_action_risk = {
                key: round(
                    _clamp(
                        value * decision.action_risk_multiplier
                    ),
                    12,
                )
                for key, value in effective_action_risk.items()
            }

        merged_metadata = dict(metadata or {})
        merged_metadata.update(
            {
                "runtime_feedback_reason": decision.reason,
                "runtime_feedback_source_step": decision.source_step_index,
                "runtime_feedback_target_step": decision.target_step_index,
                "runtime_feedback_stop": decision.should_stop,
                "runtime_feedback_increase_exploration": decision.increase_exploration,
                "runtime_feedback_reduce_risk": decision.reduce_risk,
            }
        )

        return LLMPlanRuntimeFeedbackApplication(
            target_step_index=step_index,
            exploration_risk=effective_exploration_risk,
            action_risk=effective_action_risk,
            information_gain=effective_information_gain,
            metadata=merged_metadata,
            decision=decision,
        )

    def reset(self) -> None:
        with self._lock:
            self._latest_feedback = None
            self._decision_history = []
            self._consumed_steps = set()
            self._last_decision = None


class LLMPlanRuntimeFeedbackAwareBridge:
    """
    Decorate an existing single-action bridge with feedback guidance.

    The wrapped bridge remains responsible for actual action execution.
    This adapter only modifies bounded execution inputs before forwarding.
    """

    def __init__(
        self,
        bridge: Any,
        consumer: LLMPlanRuntimeFeedbackConsumer,
    ) -> None:
        if bridge is None or not callable(getattr(bridge, "execute_step", None)):
            raise LLMPlanRuntimeFeedbackConsumptionConfigurationError(
                "bridge must expose callable execute_step(...)."
            )
        if consumer is None or not callable(getattr(consumer, "before_step", None)):
            raise LLMPlanRuntimeFeedbackConsumptionConfigurationError(
                "consumer must expose callable before_step(...)."
            )

        self.bridge = bridge
        self.consumer = consumer
        self.last_application: LLMPlanRuntimeFeedbackApplication | None = None
        self.last_result: Any = None
        self.last_execution_request: Any = None

    @property
    def api(self) -> Any:
        return getattr(self.bridge, "api", None)

    def execute_step(
        self,
        plan: Any,
        step: Any,
        *,
        goal: Any = None,
        world_state: Any = None,
        environment: Any = None,
        strategies: Any = None,
        information_gain: Mapping[str, float] | None = None,
        action_risk: Mapping[str, float] | None = None,
        exploration_risk: float = 0.0,
        action_cost: Mapping[str, float] | None = None,
        experience_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        application = self.consumer.apply_to_execution_inputs(
            int(step.index),
            exploration_risk=exploration_risk,
            action_risk=action_risk,
            information_gain=information_gain,
            metadata=metadata,
        )
        self.last_application = application

        if application.decision is not None and application.decision.should_stop:
            result = _FeedbackBlockedResult(
                step_index=int(step.index),
                reason=application.decision.reason,
            )
            self.last_result = result
            return result

        result = self.bridge.execute_step(
            plan,
            step,
            goal=goal,
            world_state=world_state,
            environment=environment,
            strategies=strategies,
            information_gain=application.information_gain,
            action_risk=application.action_risk,
            exploration_risk=application.exploration_risk,
            action_cost=action_cost,
            experience_id=experience_id,
            metadata=application.metadata,
        )
        self.last_result = result
        self.last_execution_request = getattr(
            self.bridge,
            "last_execution_request",
            None,
        )
        return result

    def execute_step_index(self, plan: Any, step_index: int, **kwargs: Any) -> Any:
        for step in plan.steps:
            if step.index == step_index:
                return self.execute_step(plan, step, **kwargs)
        raise LLMPlanRuntimeFeedbackConsumptionOrderError(
            f"Step index {step_index} is not part of the plan."
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.bridge, name)


@dataclass(frozen=True)
class _FeedbackBlockedResult:
    step_index: int
    reason: str
    success: bool = False
    result: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result",
            {
                "success": False,
                "blocked_by_runtime_feedback": True,
                "step_index": self.step_index,
                "reason": self.reason,
            },
        )


__all__ = [
    "LLMPlanRuntimeFeedbackConsumptionError",
    "LLMPlanRuntimeFeedbackConsumptionConfigurationError",
    "LLMPlanRuntimeFeedbackConsumptionOrderError",
    "LLMPlanRuntimeFeedbackDecision",
    "LLMPlanRuntimeFeedbackApplication",
    "LLMPlanRuntimeFeedbackConsumer",
    "LLMPlanRuntimeFeedbackAwareBridge",
]
