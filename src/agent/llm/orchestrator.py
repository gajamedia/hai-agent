from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .context import AgentContext, AgentContextBuilder
from .interpretation import LLMInterpretation, LLMInterpreter
from .metacognition import LLMMetacognitionEngine, LLMMetacognitionResult
from .planning import LLMPlan, LLMPlanner
from .reasoning import LLMReasoner, LLMReasoning
from .service import LLMService
from ..learning.metacognition import MetacognitiveReflection


class LLMOrchestrationError(RuntimeError):
    """Base error for the cognitive LLM orchestration layer."""


class LLMOrchestrationStageError(LLMOrchestrationError):
    """Wrap a failure with the cognitive stage that failed."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"LLM cognitive stage '{stage}' failed: {cause}")


@dataclass(frozen=True)
class CognitiveCycleResult:
    """Immutable output of one orchestrated cognitive pass.

    The result contains interpretation, reasoning and planning outputs.
    Optional LLM-assisted metacognition is included only when explicitly
    requested with a completed/partially observed episode.
    """

    context: AgentContext
    interpretation: LLMInterpretation
    reasoning: LLMReasoning
    plan: LLMPlan
    metacognition: LLMMetacognitionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "interpretation": self.interpretation.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "plan": self.plan.to_dict(),
            "metacognition": (
                self.metacognition.to_dict()
                if self.metacognition is not None
                else None
            ),
        }


class CognitiveOrchestrator:
    """Coordinate the provider-neutral LLM cognitive pipeline.

    Pipeline:

        context -> interpretation -> reasoning -> planning

    Optional post-episode metacognition can be appended when both an episode
    and its deterministic reflection are supplied. The orchestrator does not
    execute actions, mutate the Agent, write memory, or update the world model.
    """

    def __init__(
        self,
        llm_service: LLMService,
        *,
        context_builder: AgentContextBuilder | None = None,
        interpreter: LLMInterpreter | None = None,
        reasoner: LLMReasoner | None = None,
        planner: LLMPlanner | None = None,
        metacognition_engine: LLMMetacognitionEngine | None = None,
    ) -> None:
        if llm_service is None:
            raise ValueError("llm_service is required")

        self.llm_service = llm_service
        self.context_builder = context_builder or AgentContextBuilder()
        self.interpreter = interpreter or LLMInterpreter(llm_service)
        self.reasoner = reasoner or LLMReasoner(llm_service)
        self.planner = planner or LLMPlanner(llm_service)
        self.metacognition_engine = metacognition_engine

        self.last_result: CognitiveCycleResult | None = None
        self.last_failed_stage: str | None = None

        self._validate_shared_service(
            "interpreter", self.interpreter
        )
        self._validate_shared_service("reasoner", self.reasoner)
        self._validate_shared_service("planner", self.planner)
        if self.metacognition_engine is not None:
            self._validate_shared_service(
                "metacognition_engine", self.metacognition_engine
            )

    def build_context(
        self,
        *,
        goal: Any = None,
        state: Any = None,
        world_model: Any = None,
        relevant_memory: Any = None,
        strategy: Any = None,
        self_awareness: Any = None,
        calibration: Any = None,
        metacognition: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentContext:
        """Build an AgentContext through the configured context builder."""
        return self.context_builder.build(
            goal=goal,
            state=state,
            world_model=world_model,
            relevant_memory=relevant_memory,
            strategy=strategy,
            self_awareness=self_awareness,
            calibration=calibration,
            metacognition=metacognition,
            metadata=metadata,
        )

    def run(
        self,
        observation: Any,
        context: AgentContext,
        *,
        episode: Any = None,
        deterministic_reflection: MetacognitiveReflection | None = None,
        include_metacognition: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> CognitiveCycleResult:
        """Run one side-effect-free cognitive pass."""
        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        if include_metacognition:
            if self.metacognition_engine is None:
                raise ValueError(
                    "metacognition_engine is required when include_metacognition=True"
                )
            if episode is None:
                raise ValueError(
                    "episode is required when include_metacognition=True"
                )
            if not isinstance(
                deterministic_reflection,
                MetacognitiveReflection,
            ):
                raise TypeError(
                    "deterministic_reflection must be a MetacognitiveReflection"
                )

        self.last_failed_stage = None
        call_metadata = {
            "orchestration": "cognitive_cycle",
            **dict(metadata or {}),
        }

        try:
            interpretation = self.interpreter.interpret(
                observation,
                context,
                metadata={"stage": "interpretation", **call_metadata},
            )
        except Exception as exc:
            self.last_failed_stage = "interpretation"
            raise LLMOrchestrationStageError("interpretation", exc) from exc

        try:
            reasoning = self.reasoner.reason(
                interpretation,
                context,
                observation=observation,
                metadata={"stage": "reasoning", **call_metadata},
            )
        except Exception as exc:
            self.last_failed_stage = "reasoning"
            raise LLMOrchestrationStageError("reasoning", exc) from exc

        try:
            plan = self.planner.plan(
                reasoning,
                context,
                metadata={"stage": "planning", **call_metadata},
            )
        except Exception as exc:
            self.last_failed_stage = "planning"
            raise LLMOrchestrationStageError("planning", exc) from exc

        metacognition = None
        if include_metacognition:
            assert self.metacognition_engine is not None
            assert deterministic_reflection is not None
            try:
                metacognition = self.metacognition_engine.reflect(
                    episode,
                    deterministic_reflection,
                    context,
                    metadata={
                        "stage": "metacognition",
                        **call_metadata,
                    },
                )
            except Exception as exc:
                self.last_failed_stage = "metacognition"
                raise LLMOrchestrationStageError(
                    "metacognition", exc
                ) from exc

        result = CognitiveCycleResult(
            context=context,
            interpretation=interpretation,
            reasoning=reasoning,
            plan=plan,
            metacognition=metacognition,
        )
        self.last_result = result
        return result

    def clear_trace(self) -> None:
        """Clear orchestration and component traces."""
        self.last_result = None
        self.last_failed_stage = None
        self.interpreter.clear_trace()
        self.reasoner.clear_trace()
        self.planner.clear_trace()
        if self.metacognition_engine is not None:
            self.metacognition_engine.clear_trace()

    def _validate_shared_service(self, name: str, component: Any) -> None:
        service = getattr(component, "llm_service", None)
        if service is not self.llm_service:
            raise ValueError(
                f"{name} must use the same LLMService instance as the orchestrator"
            )


__all__ = [
    "CognitiveCycleResult",
    "CognitiveOrchestrator",
    "LLMOrchestrationError",
    "LLMOrchestrationStageError",
]
