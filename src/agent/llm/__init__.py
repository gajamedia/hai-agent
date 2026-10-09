# from .contracts import (
#     LLMMessage,
#     LLMProvider,
#     LLMRequest,
#     LLMResponse,
# )

# from .service import (
#     LLMService,
# )

# from .context import (
#     AgentContext,
#     AgentContextBuilder,
# )

# from .interpretation import (
#     LLMInterpretation,
#     LLMInterpretationError,
#     LLMInterpretationParseError,
#     LLMInterpretationParser,
#     LLMInterpretationSchemaError,
#     LLMInterpreter,
# )

# from .reasoning import (
#     LLMReasoning,
#     LLMReasoningError,
#     LLMReasoningParseError,
#     LLMReasoningParser,
#     LLMReasoningSchemaError,
#     LLMReasoner,
# )

# from .planning import (
#     LLMPlan,
#     LLMPlanParser,
#     LLMPlanStep,
#     LLMPlanner,
#     LLMPlanningError,
#     LLMPlanningParseError,
#     LLMPlanningSchemaError,
# )

# from .metacognition import (
#     LLMMetacognitiveReflection,
#     LLMMetacognitionEngine,
#     LLMMetacognitionError,
#     LLMMetacognitionParseError,
#     LLMMetacognitionParser,
#     LLMMetacognitionResult,
#     LLMMetacognitionSchemaError,
# )

# from .orchestrator import (
#     CognitiveCycleResult,
#     CognitiveOrchestrator,
#     LLMOrchestrationError,
#     LLMOrchestrationStageError,
# )

# from .agent_runtime import (
#     AgentCognitiveRuntime,
#     AgentCognitiveRuntimeConfigurationError,
#     AgentCognitiveRuntimeError,
#     AgentCognitiveRuntimeTrace,
# )

# from .plan_validation import (
#     LLMPlanValidationError,
#     LLMPlanValidationResult,
#     LLMPlanValidator,
#     PlanValidationIssue,
# )

# from .controlled_execution import (
#     ControlledExecutionError,
#     ControlledExecutionReport,
#     ControlledPlanExecutor,
#     ControlledStepResult,
# )

# from .agent_execution_boundary import (
#     AgentExecutionBoundary,
#     AgentExecutionBoundaryError,
#     AgentExecutionRequest,
#     AgentPlanExecutionResult,
#     CallbackAgentExecutionBoundary,
#     LLMPlanAgentExecutionAdapter,
# )

# from .unified_executor_binding import (
#     AgentUnifiedExecutionReport,
#     AgentUnifiedExecutorBinding,
#     AgentUnifiedExecutorBindingConfigurationError,
#     AgentUnifiedExecutorBindingError,
#     DecisionFactory,
# )

# from .canonical_single_action import (
#     CanonicalSingleActionConfigurationError,
#     CanonicalSingleActionExecutionError,
#     CanonicalSingleActionExecutor,
#     CanonicalSingleActionRequest,
#     CanonicalSingleActionResult,
# )

# from .agent_canonical_single_action_api import (
#     AgentCanonicalSingleActionAPI,
#     AgentCanonicalSingleActionAPIConfigurationError,
#     AgentCanonicalSingleActionAPIError,
# )

# from .llm_plan_single_action_bridge import (
#     LLMPlanSingleActionBridge,
#     LLMPlanSingleActionBridgeConfigurationError,
#     LLMPlanSingleActionBridgeError,
#     LLMPlanSingleActionBridgeRequestError,
#     LLMPlanSingleActionExecutionRequest,
# )

# from .llm_plan_runtime_execution import (
#     LLMPlanRuntimeConfigurationError,
#     LLMPlanRuntimeExecutionError,
#     LLMPlanRuntimeExecutionReport,
#     LLMPlanRuntimeExecutor,
#     LLMPlanRuntimeValidationError,
#     LLMPlanStepExecutionRecord,
# )

# from .llm_plan_runtime_state import (
#     LLMPlanRuntimeState,
#     LLMPlanRuntimeStateConfigurationError,
#     LLMPlanRuntimeStateError,
#     LLMPlanRuntimeStateTracker,
#     LLMPlanRuntimeStateTransitionError,
#     LLMPlanRuntimeStatus,
# )

# from .llm_plan_runtime_execution import (
#     LLMPlanRuntimeEvent,
#     RuntimeEventObserver,
# )

# from .llm_plan_live_runtime import (
#     LLMPlanLiveRuntimeError,
#     LLMPlanLiveRuntimeConfigurationError,
#     LLMPlanLiveRuntimeStateError,
#     LLMPlanLiveRuntimeExecutionResult,
#     LLMPlanLiveRuntimeExecutor,
# )

# from .llm_plan_agent_state_integration import (
#     LLMPlanAgentStateIntegrationError,
#     LLMPlanAgentStateIntegrationConfigurationError,
#     LLMPlanAgentStateIntegrationProjectionError,
#     LLMPlanAgentStateIntegrationResult,
#     LLMPlanAgentStateIntegrator,
# )

# from .llm_plan_agent_experience_integration import (
#     LLMPlanAgentExperienceIntegrationError,
#     LLMPlanAgentExperienceIntegrationConfigurationError,
#     LLMPlanAgentExperienceRecordingError,
#     LLMPlanAgentExperienceRecord,
#     LLMPlanAgentExperienceIntegrationResult,
#     LLMPlanAgentExperienceIntegrator,
# )

# from .llm_plan_runtime_learning_feedback import (
#     LLMPlanRuntimeLearningFeedbackError,
#     LLMPlanRuntimeLearningFeedbackConfigurationError,
#     LLMPlanRuntimeLearningFeedbackInputError,
#     LLMPlanRuntimeLearningFeedback,
#     LLMPlanRuntimeLearningFeedbackEngine,
#     LLMPlanRuntimeLearningFeedbackBridge,
# )

# from .llm_plan_runtime_feedback_consumption import (
#     LLMPlanRuntimeFeedbackApplication,
#     LLMPlanRuntimeFeedbackAwareBridge,
#     LLMPlanRuntimeFeedbackConsumptionConfigurationError,
#     LLMPlanRuntimeFeedbackConsumptionError,
#     LLMPlanRuntimeFeedbackConsumptionOrderError,
#     LLMPlanRuntimeFeedbackConsumer,
#     LLMPlanRuntimeFeedbackDecision,
# )

# from .llm_plan_adaptive_replanning import (
#     LLMPlanAdaptiveReplanner,
#     LLMPlanAdaptiveReplanningConfigurationError,
#     LLMPlanAdaptiveReplanningError,
#     LLMPlanAdaptiveReplanningRequestError,
#     LLMPlanAdaptiveReplanningSafetyError,
#     LLMPlanAdaptiveReplanningValidationError,
#     LLMPlanAdaptiveReplanningResult,
#     LLMPlanReplannerProvider,
#     LLMPlanReplanningRequest,
#     LLMPlanReplanningTrigger,
# )

# from .llm_plan_adaptive_runtime import (
#     AdaptiveRuntimeEventObserver,
#     AdaptiveRuntimeFeedbackSource,
#     LLMPlanAdaptiveRuntime,
#     LLMPlanAdaptiveRuntimeConfigurationError,
#     LLMPlanAdaptiveRuntimeError,
#     LLMPlanAdaptiveRuntimeEvent,
#     LLMPlanAdaptiveRuntimeExecutionError,
#     LLMPlanAdaptiveRuntimeExecutionReport,
#     LLMPlanAdaptiveRuntimeReplanningError,
#     LLMPlanAdaptiveRuntimeSafetyError,
# )

# from .llm_planner_adaptive_replanning import (
#     AdaptiveReplanningContextBuilder,
#     AdaptiveReplanningReasoningBuilder,
#     AdaptiveReplanningRiskResolver,
#     LLMPlannerAdaptiveReplanningBuildError,
#     LLMPlannerAdaptiveReplanningConfigurationError,
#     LLMPlannerAdaptiveReplanningContext,
#     LLMPlannerAdaptiveReplanningError,
#     LLMPlannerAdaptiveReplanningProvider,
#     LLMPlannerAdaptiveReplanningRequestError,
#     LLMPlannerAdaptiveReplanningResult,
#     build_llm_adaptive_replanner,
#     default_risk_resolver,
# )

# from .goal_management import (
#     GoalHierarchyError,
#     GoalLifecycleError,
#     GoalLifecycleManager,
#     GoalManagementConfigurationError,
#     GoalManagementError,
#     GoalNotFoundError,
#     GoalSelection,
#     GoalSelectionPolicy,
#     GoalManagerSnapshot,
#     GoalStatus,
#     ManagedGoal,
# )

# from .agent_goal_management_integration import (
#     AgentGoalManagementIntegrationConfigurationError,
#     AgentGoalManagementIntegrationError,
#     AgentGoalManagementIntegrator,
#     AgentGoalManagementSnapshot,
#     AgentGoalProjectionError,
# )

# from .goal_state_persistence import (
#     PersistentGoalStateConfigurationError,
#     PersistentGoalStateError,
#     PersistentGoalStateLoadResult,
#     PersistentGoalStateNotFoundError,
#     PersistentGoalStateSerializationError,
#     PersistentGoalStateSnapshot,
#     PersistentGoalStateStore,
#     PersistentGoalStateVersionError,
# )

# from .goal_progress_evaluation import (
#     GoalAchievementEvidence,
#     GoalAchievementEvaluation,
#     GoalEvaluationConfigurationError,
#     GoalEvaluationError,
#     GoalEvaluationEvidenceError,
#     GoalEvaluationInputError,
#     GoalEvaluationThresholds,
#     GoalProgressAchievementEvaluator,
# )

# from .multi_goal_arbitration import (
#     MultiGoalArbitrationApplier,
#     MultiGoalArbitrationApplicationError,
#     MultiGoalArbitrationCandidate,
#     MultiGoalArbitrationConfigurationError,
#     MultiGoalArbitrationDecision,
#     MultiGoalArbitrationError,
#     MultiGoalArbitrationInputError,
#     MultiGoalArbitrationThresholds,
#     MultiGoalArbitrationWeights,
#     MultiGoalArbiter,
# )

# from .goal_conflict_resolution import (
#     GoalConflictCandidate,
#     GoalConflictPolicy,
#     GoalConflictRelation,
#     GoalConflictResolutionApplier,
#     GoalConflictResolutionApplicationError,
#     GoalConflictResolutionConfigurationError,
#     GoalConflictResolutionDecision,
#     GoalConflictResolutionError,
#     GoalConflictResolutionInputError,
#     GoalConflictResolutionResult,
#     GoalConflictResolver,
# )

# from .goal_aware_llm_planning import (
#     GoalAwareLLMPlanningConfigurationError,
#     GoalAwareLLMPlanningConsistencyError,
#     GoalAwarePlanningContextBuilder,
#     GoalAwareLLMPlanningError,
#     GoalAwareLLMPlanningInputError,
#     GoalAwareLLMPlanningRequest,
#     GoalAwareLLMPlanner,
#     GoalAwarePlanningContext,
# )

# from .goal_aware_adaptive_replanning import (
#     GoalAwareAdaptiveReplanningConfigurationError,
#     GoalAwareAdaptiveReplanningConsistencyError,
#     GoalAwareAdaptiveReplanningDecision,
#     GoalAwareAdaptiveReplanningError,
#     GoalAwareAdaptiveReplanningInputError,
#     GoalAwareAdaptiveReplanningProvider,
#     GoalAwareAdaptiveReplanningResult,
#     GoalAwareAdaptiveReplanningController,
# )

# from .closed_loop_multi_goal_runtime import (
#     ClosedLoopMultiGoalRuntime,
#     ClosedLoopMultiGoalRuntimeConfigurationError,
#     ClosedLoopMultiGoalRuntimeCoordinationError,
#     ClosedLoopMultiGoalRuntimeError,
#     ClosedLoopMultiGoalRuntimeEvent,
#     ClosedLoopMultiGoalRuntimeExecutionError,
#     ClosedLoopMultiGoalRuntimeExecutionReport,
#     ClosedLoopMultiGoalRuntimeInputError,
#     ClosedLoopMultiGoalRuntimeReplanningError,
#     ClosedLoopMultiGoalRuntimeSegment,
# )

# from .goal_proposal import (
#     GoalCandidate,
#     GoalOrigin,
#     GoalProposalBatch,
#     GoalProposalConfigurationError,
#     GoalProposalEngine,
#     GoalProposalError,
#     GoalProposalInputError,
#     GoalSignal,
# )

# from .goal_acceptance import (
#     AgentGoalAcceptanceIntegrator,
#     GoalAcceptanceApplicationError,
#     GoalAcceptanceConfigurationError,
#     GoalAcceptanceDecision,
#     GoalAcceptanceEngine,
#     GoalAcceptanceError,
#     GoalAcceptanceInputError,
#     GoalAcceptancePolicy,
#     GoalAcceptanceReason,
#     GoalAcceptanceResult,
# )

# from .goal_utility_evaluation import (
#     GoalUtilityEvaluation,
#     GoalUtilityEvaluationConfigurationError,
#     GoalUtilityEvaluationError,
#     GoalUtilityEvaluationInputError,
#     GoalUtilityEvaluator,
#     GoalUtilityInput,
#     GoalUtilityWeights,
# )

# from .intrinsic_motivation import (
#     IntrinsicMotivationError,
#     IntrinsicMotivationConfigurationError,
#     IntrinsicMotivationInputError,
#     IntrinsicMotivationWeights,
#     IntrinsicMotivationInput,
#     IntrinsicMotivationEvaluation,
#     IntrinsicMotivationEvaluator,
# )

# from .memory_goal_generation import (
#     MemoryGoalGenerationError,
#     MemoryGoalGenerationConfigurationError,
#     MemoryGoalGenerationInputError,
#     MemoryGapType,
#     MemoryGap,
#     MemoryGapGoalGenerationWeights,
#     MemoryGapGoalGenerationResult,
#     MemoryGapGoalGenerationBatch,
#     MemoryGapGoalGenerator,
# )

# from .goal_generation_pipeline import (
#     GoalGenerationPipeline,
#     GoalGenerationPipelineConfigurationError,
#     GoalGenerationPipelineError,
#     GoalGenerationPipelineInputError,
#     GoalGenerationPipelineResult,
# )

# from .goal_evaluation import (
#     GoalEvaluationBatch,
#     GoalEvaluationConfigurationError,
#     GoalEvaluationEngine,
#     GoalEvaluationError,
#     GoalEvaluationInputError,
#     GoalEvaluationResult,
#     GoalEvaluationWeights,
# )

# from .goal_evaluation_acceptance import (
#     GoalEvaluationAcceptanceBatch,
#     GoalEvaluationAcceptanceConfigurationError,
#     GoalEvaluationAcceptanceDecision,
#     GoalEvaluationAcceptanceEngine,
#     GoalEvaluationAcceptanceError,
#     GoalEvaluationAcceptanceInputError,
#     GoalEvaluationAcceptancePolicy,
#     GoalEvaluationAcceptanceResult,
# )

# __all__ = [
#     "LLMMessage",
#     "LLMProvider",
#     "LLMRequest",
#     "LLMResponse",
#     "LLMService",
#     "AgentContext",
#     "AgentContextBuilder",
#     "LLMInterpretation",
#     "LLMInterpretationError",
#     "LLMInterpretationParseError",
#     "LLMInterpretationParser",
#     "LLMInterpretationSchemaError",
#     "LLMInterpreter",
#     "LLMReasoning",
#     "LLMReasoningError",
#     "LLMReasoningParseError",
#     "LLMReasoningParser",
#     "LLMReasoningSchemaError",
#     "LLMReasoner",
#     "LLMPlan",
#     "LLMPlanParser",
#     "LLMPlanStep",
#     "LLMPlanner",
#     "LLMPlanningError",
#     "LLMPlanningParseError",
#     "LLMPlanningSchemaError",
#     "LLMMetacognitiveReflection",
#     "LLMMetacognitionEngine",
#     "LLMMetacognitionError",
#     "LLMMetacognitionParseError",
#     "LLMMetacognitionParser",
#     "LLMMetacognitionResult",
#     "LLMMetacognitionSchemaError",
#     "CognitiveCycleResult",
#     "CognitiveOrchestrator",
#     "LLMOrchestrationError",
#     "LLMOrchestrationStageError",
#     "AgentCognitiveRuntime",
#     "AgentCognitiveRuntimeConfigurationError",
#     "AgentCognitiveRuntimeError",
#     "AgentCognitiveRuntimeTrace",
# ]


#=======================================
"""
Public API for the ``agent.llm`` package.

Exports are resolved lazily to avoid importing the entire cognitive runtime when
a caller only needs lightweight contracts. This also helps prevent circular
imports between the agent, runtime, planning, and learning packages.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


# Explicit mappings for the most frequently used contracts and runtime types.
# These are loaded directly and do not require scanning the other modules.
_EXPORT_MODULES: dict[str, str] = {
    "LLMMessage": "contracts",
    "LLMProvider": "contracts",
    "LLMRequest": "contracts",
    "LLMResponse": "contracts",
    "LLMService": "service",
    "AgentContext": "context",
    "AgentContextBuilder": "context",
    "AgentCognitiveRuntime": "agent_runtime",
    "AgentCognitiveRuntimeConfigurationError": "agent_runtime",
    "AgentCognitiveRuntimeError": "agent_runtime",
    "AgentCognitiveRuntimeTrace": "agent_runtime",
}

# Modules that historically contributed public names to this package.
# They are not imported at package-import time.
_LAZY_MODULES: tuple[str, ...] = (
    "contracts",
    "service",
    "context",
    "interpretation",
    "reasoning",
    "planning",
    "metacognition",
    "orchestrator",
    "agent_runtime",
    "plan_validation",
    "controlled_execution",
    "agent_execution_boundary",
    "unified_executor_binding",
    "canonical_single_action",
    "agent_canonical_single_action_api",
    "llm_plan_single_action_bridge",
    "llm_plan_runtime_execution",
    "llm_plan_runtime_state",
    "llm_plan_live_runtime",
    "llm_plan_agent_state_integration",
    "llm_plan_agent_experience_integration",
    "llm_plan_runtime_learning_feedback",
    "llm_plan_runtime_feedback_consumption",
    "llm_plan_adaptive_replanning",
    "llm_plan_adaptive_runtime",
    "llm_planner_adaptive_replanning",
    "goal_management",
    "agent_goal_management_integration",
    "goal_state_persistence",
    "goal_progress_evaluation",
    "multi_goal_arbitration",
    "goal_conflict_resolution",
    "goal_aware_llm_planning",
    "goal_aware_adaptive_replanning",
    "closed_loop_multi_goal_runtime",
    "goal_proposal",
    "goal_acceptance",
    "goal_utility_evaluation",
    "intrinsic_motivation",
    "memory_goal_generation",
    "goal_generation_pipeline",
    "goal_evaluation",
    "goal_evaluation_acceptance",
)


def __getattr__(name: str) -> Any:
    """Resolve public exports only when requested."""
    module_name = _EXPORT_MODULES.get(name)

    if module_name is not None:
        module = import_module(f"{__name__}.{module_name}")
        try:
            value = getattr(module, name)
        except AttributeError as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        globals()[name] = value
        return value

    # Compatibility fallback for the package's wider historical public API.
    # Importing these modules lazily avoids eager circular-import chains.
    for candidate_module in _LAZY_MODULES:
        try:
            module = import_module(f"{__name__}.{candidate_module}")
        except ImportError:
            # A module may depend on an optional integration or another module
            # currently being initialized. Continue looking for this export.
            continue

        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose known lazy names in interactive completion."""
    return sorted(set(globals()) | set(_EXPORT_MODULES))


__all__ = sorted(set(_EXPORT_MODULES))


