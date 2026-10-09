from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional, List

from agent.core.goal import Goal
from agent.core.state import AgentState
from agent.core.action import Action

from agent.core.experience import Experience
from agent.core.experience_store import ExperienceStore
from agent.core.experience_recorder import ExperienceRecorder

from agent.memory.working import WorkingMemory
from agent.memory.episodic_store import EpisodicMemoryStore
from agent.memory.episode_recorder import EpisodeRecorder
from agent.memory.semantic_store import SemanticMemoryStore
from agent.memory.causal_store import CausalMemoryStore
from agent.memory.reflection_store import ReflectionMemoryStore
from agent.memory.procedural_store import ProceduralMemoryStore
from agent.memory.persistence import PersistentMemory

from agent.memory.strategy_repository import (
    StrategyRepository,
)
from agent.memory.strategy_store import (
    StrategyMemoryStore,
)

from agent.learning.semantic_learning import (
    SemanticLearning,
)
from agent.learning.causal_learning import (
    CausalLearning,
)
from agent.learning.online_causal_learning import (
    OnlineCausalLearning,
)
from agent.learning.procedural_learning import (
    ProceduralLearning,
)
from agent.learning.reflection import (
    ReflectionEngine,
)
from agent.learning.reflection_strategy_updater import (
    ReflectionStrategyUpdater,
)
from agent.learning.metacognition import (
    MetacognitionEngine,
    MetacognitiveReflection,
)
from agent.llm import (
    LLMMessage,
    LLMService,
    LLMResponse,
)
from agent.learning.strategy import Strategy
from agent.learning.strategy_attribution import (
    StrategyAttributor,
    StrategyAttributionResult,
)
from agent.learning.strategy_consolidation import (
    StrategyConsolidator,
    StrategyLifecycleManager,
)
from agent.learning.strategy_recovery import (
    StrategyRecoveryManager,
)
from agent.learning.progress_learning import (
    ProgressAwareLearning,
    ProgressLearningResult,
)

from agent.learning.strategy_evolution import (
    StrategyEvolution,
)
from agent.learning.strategy_candidate_validator import (
    StrategyCandidateValidator,
)
from agent.learning.strategy_evolution_selector import (
    StrategyEvolutionPipeline,
    StrategyEvolutionSelectionResult,
)

from agent.learning.strategy_evolution_evidence import (
    StrategyEvolutionEvidence,
    StrategyEvolutionEvidenceStore,
)

from agent.learning.meta_learning import (
    MetaLearningStore,
)
from agent.learning.meta_learning_repository import (
    MetaLearningRepository,
)
from agent.learning.persistent_meta_learning_store import (
    PersistentMetaLearningStore,
)
from agent.learning.meta_learning_attribution import (
    MetaLearningEventAttributor,
)
from agent.learning.meta_learning_feedback import (
    MetaLearningFeedbackEngine,
)
from agent.learning.meta_learning_feedback_policy import (
    MetaLearningFeedbackPolicy,
)
from agent.learning.meta_learning_runtime import (
    MetaLearningRuntime,
)

from agent.learning.agent_behavioral_adaptation import (
    AgentBehavioralAdaptation,
    BehavioralAdaptationResult,
)
from agent.learning.end_to_end_behavioral_decision import (
    EndToEndBehavioralDecisionEvaluator,
    BehavioralDecisionResult,
)
from agent.learning.agent_self_awareness import (
    AgentSelfAwareness,
    AgentSelfAwarenessSnapshot,
)
from agent.learning.self_aware_decision import (
    SelfAwareDecisionArbiter,
)
from agent.learning.self_awareness_feedback import (
    SelfAwarenessFeedbackTrace,
    SelfAwarenessFeedbackRecord,
    SelfAwarenessFeedbackSnapshot,
)
from agent.learning.self_awareness_calibration import (
    SelfAwarenessCalibrator,
    SelfAwarenessCalibrationResult,
)
from agent.learning.experience_self_awareness import (
    ExperienceSelfAwarenessUpdater,
    ExperienceSelfAwarenessUpdateResult,
)
from agent.learning.capability_telemetry import (
    CapabilityTelemetry,
    CapabilityTelemetryEvent,
    CapabilityTelemetrySnapshot,
)
from agent.learning.unified_capability_evidence import (
    UnifiedCapabilityEvidenceBuilder,
    UnifiedCapabilityEvidenceResult,
)
from agent.learning.calibrated_self_awareness import (
    CalibratedSelfAwareInfluenceApplier,
    CalibratedSelfAwareInfluenceResult,
)

from agent.self.store import SelfModelStore
from agent.self.learning import SelfLearning

from agent.cognition.planner import Planner
from agent.cognition.adaptive_planner import (
    AdaptivePlanner,
)
from agent.cognition.goal_decomposer import (
    GoalDecomposer,
)
from agent.cognition.hierarchical_planner import (
    HierarchicalPlanner,
)
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)
from agent.cognition.active_explorer import (
    ActiveExplorer,
)
from agent.cognition.agent_decision import (
    AgentDecisionMaker,
    DecisionMode,
)
from agent.cognition.unified_executor import (
    UnifiedDecisionExecutor,
)
from agent.learning.strategy_selector import (
    StrategySelector,
)
from agent.learning.calibrated_self_awareness_persistence import (
    PersistentCalibratedSelfAwarenessStore,
    CalibratedSelfAwarenessPersistenceSnapshot,
)
from agent.learning.learned_state_restoration import (
    UnifiedLearnedStateRestorationResult,
)

class Agent:

    def __init__(
        self,
        name: str,
        memory_path: Optional[str | Path] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.name = name

        # =================================================
        # Core state
        # =================================================

        self.state = AgentState()

        self.working_memory = WorkingMemory()

        # Keep the original path so additional persistent
        # subsystems can use the same database location.
        self.memory_path: Optional[Path] = (
            Path(memory_path)
            if memory_path is not None
            else None
        )

        # =================================================
        # Persistent long-term memory
        # =================================================

        self.persistent_memory: Optional[
            PersistentMemory
        ] = None

        if self.memory_path is not None:
            self.persistent_memory = (
                PersistentMemory(
                    self.memory_path
                )
            )

        # =================================================
        # Self-awareness
        # =================================================
        #
        # High-level self-awareness facade.
        #
        # Pipeline:
        #
        #   Capability Confidence
        #          ↓
        #   Self Uncertainty
        #          ↓
        #   Decision Influence
        #          ↓
        #   Optional Persistence
        #
        # The facade keeps self-awareness orchestration
        # outside the Agent implementation.
        # =================================================

        self.self_awareness = (
            AgentSelfAwareness(
                agent_name=self.name,
                memory_path=self.memory_path,
            )
        )

        self.experience_self_awareness_updater = (
            ExperienceSelfAwarenessUpdater(
                self.self_awareness
            )
        )
        self.last_experience_self_awareness_update = None

        # =================================================
        # Self-aware decision arbitration
        # =================================================
        #
        # Self-awareness does not replace the existing
        # decision engine.
        #
        # Pipeline:
        #
        #   Existing AgentDecision
        #          ↓
        #   SelfAwareDecisionArbiter
        #          ↓
        #   Final AgentDecision
        #
        # The arbiter only applies bounded mode influence.
        # It never executes actions.
        # =================================================

        self.self_aware_decision_arbiter = (
            SelfAwareDecisionArbiter()
        )
        self.last_self_aware_decision_result = None

        self.self_awareness_feedback = (
            SelfAwarenessFeedbackTrace()
        )

        self.self_awareness_calibrator = (
            SelfAwarenessCalibrator()
        )
        self.last_self_awareness_calibration = None

        self.calibrated_self_aware_influence_applier = (
            CalibratedSelfAwareInfluenceApplier()
        )
        self.last_calibrated_self_aware_influence = None
        self.last_calibrated_self_aware_result = None
        self.last_unified_learned_state_restoration = None
        self.calibrated_self_awareness_store = None
        if self.memory_path is not None:
            calibrated_path = (
                self.memory_path
                if self.memory_path.suffix
                else (
                    self.memory_path
                    / "calibrated_self_awareness.sqlite3"
                )
            )
            if calibrated_path == self.memory_path:
                calibrated_path = (
                    self.memory_path.parent
                    / "calibrated_self_awareness.sqlite3"
                )
            self.calibrated_self_awareness_store = (
                PersistentCalibratedSelfAwarenessStore(
                    calibrated_path
                )
            )

        self.capability_telemetry = (
            CapabilityTelemetry()
        )
        self.last_capability_telemetry_event = None

        self.unified_capability_evidence_builder = (
            UnifiedCapabilityEvidenceBuilder(
                telemetry_priority=True
            )
        )
        self.last_unified_capability_evidence = None        

        # =================================================
        # Episodic memory
        # =================================================

        self.episodic_memory = (
            EpisodicMemoryStore(
                repository=(
                    self.persistent_memory.episodic
                    if self.persistent_memory is not None
                    else None
                )
            )
        )

        # =================================================
        # Semantic memory
        # =================================================

        self.semantic_memory = (
            SemanticMemoryStore(
                repository=(
                    self.persistent_memory.semantic
                    if self.persistent_memory is not None
                    else None
                )
            )
        )

        # =================================================
        # Causal memory
        # =================================================

        self.causal_memory = (
            CausalMemoryStore(
                repository=(
                    self.persistent_memory.causal
                    if self.persistent_memory is not None
                    else None
                )
            )
        )

        # =================================================
        # Reflection memory
        # =================================================

        self.reflection_memory = (
            ReflectionMemoryStore(
                repository=(
                    self.persistent_memory.reflection
                    if self.persistent_memory is not None
                    else None
                )
            )
        )

        self.reflection_engine = (
            ReflectionEngine(
                self.reflection_memory
            )
        )

        # =================================================
        # Metacognition
        # =================================================
        #
        # Converts an observed episode into a structured
        # meta-level interpretation. The engine itself does
        # not mutate Agent state; Agent only keeps the latest
        # reflection for inspection and downstream learning.
        # =================================================

        self.metacognition_engine = (
            MetacognitionEngine()
        )
        self.last_metacognitive_reflection = None

        # =================================================
        # LLM integration
        # =================================================
        #
        # The Agent depends only on the provider-neutral LLMService.
        # A concrete provider is injected from the application layer.
        # No vendor SDK or network call is required by the Agent core.
        # =================================================

        self.llm_service: Optional[LLMService] = (
            llm_service
        )
        self.last_llm_request = None
        self.last_llm_response: Optional[LLMResponse] = None

        # =================================================
        # Episodic recording / learning
        # =================================================

        self.episode_recorder = (
            EpisodeRecorder(
                self.episodic_memory
            )
        )

        self.semantic_learning = (
            SemanticLearning(
                self.semantic_memory
            )
        )

        self.causal_learning = (
            CausalLearning(
                self.causal_memory
            )
        )

        self.online_causal_learning = (
            OnlineCausalLearning(
                self.causal_learning
            )
        )

        # =================================================
        # Actions
        # =================================================

        self.actions: dict[str, Action] = {}

        # =================================================
        # Strategy
        # =================================================

        self.reflection_strategy_updater = (
            ReflectionStrategyUpdater()
        )

        self.active_strategy: Optional[
            Strategy
        ] = None

        self.strategy_selector = (
            StrategySelector()
        )

        self.risk_strategy_selector = (
            RiskAwareStrategySelector()
        )

        self.strategy_attributor = (
            StrategyAttributor()
        )

        self.strategy_consolidator = (
            StrategyConsolidator()
        )

        self.strategy_lifecycle = (
            StrategyLifecycleManager()
        )

        # -------------------------------------------------
        # Strategy recovery
        # -------------------------------------------------

        self.strategy_recovery = (
            StrategyRecoveryManager()
        )

        # -------------------------------------------------
        # Strategy evolution evidence
        # -------------------------------------------------

        self.strategy_evolution_evidence = (
            StrategyEvolutionEvidenceStore()
        )

        # =================================================
        # Persistent strategy memory
        # =================================================

        if self.memory_path is not None:
            self.strategy_store = (
                StrategyMemoryStore(
                    repository=StrategyRepository(
                        self.memory_path
                    )
                )
            )
        else:
            self.strategy_store = (
                StrategyMemoryStore()
            )

        # Keep compatibility with the existing public
        # strategy_registry concept.
        self.strategy_registry: list[
            Strategy
        ] = self.strategy_store.all()

        # =================================================
        # Meta-learning
        # =================================================
        #
        # Meta-learning learns about the agent's own
        # decision/adaptation behavior:
        #
        #   EXPLORE
        #   EXPLOIT
        #   EVOLUTION
        #   RECOVERY
        #   PROGRESS
        #
        # When memory_path is available, meta-learning
        # uses the same SQLite database as the rest of the
        # persistent memory subsystem.
        #
        # Without memory_path, behavior remains exactly
        # compatible with the original in-memory store.
        # =================================================

        if self.memory_path is not None:

            self.meta_learning_repository = (
                MetaLearningRepository(
                    self.memory_path
                )
            )

            self.meta_learning_store = (
                PersistentMetaLearningStore(
                    repository=(
                        self.meta_learning_repository
                    )
                )
            )

        else:

            self.meta_learning_repository = None

            self.meta_learning_store = (
                MetaLearningStore()
            )

        self.meta_learning_profile = (
            self.meta_learning_store.get_or_create(
                self.name
            )
        )

        # =================================================
        # Behavioral adaptation
        # =================================================
        #
        # Converts meta-learning history into stabilized
        # behavioral signals for future decision influence.
        #
        # The subsystem does not modify the current
        # AgentDecision yet.
        # =================================================

        self.behavioral_adaptation = (
            AgentBehavioralAdaptation()
        )

        self.behavioral_decision_evaluator = (
            EndToEndBehavioralDecisionEvaluator(
                behavioral_adaptation=(
                    self.behavioral_adaptation
                )
            )
        )

        self.meta_learning_attributor = (
            MetaLearningEventAttributor()
        )

        # =================================================
        # Planning
        # =================================================

        self.planner: Optional[
            Planner
        ] = None

        self.adaptive_planner: Optional[
            AdaptivePlanner
        ] = None

        self.goal_decomposer = (
            GoalDecomposer()
        )

        self.hierarchical_planner: Optional[
            HierarchicalPlanner
        ] = None

        self.risk_aware_hierarchical_planner: Optional[
            RiskAwareHierarchicalPlanner
        ] = None

        # =================================================
        # Decision / exploration
        # =================================================

        self.decision_maker = (
            AgentDecisionMaker()
        )

        self.active_explorer = (
            ActiveExplorer()
        )

        # =================================================
        # Meta-learning runtime
        # =================================================
        #
        # Connects accumulated meta-learning evidence
        # to runtime decision making.
        #
        # Meta-learning supplies bounded adaptive
        # influence; it does not directly execute actions.
        # =================================================

        self.meta_learning_feedback_engine = (
            MetaLearningFeedbackEngine()
        )

        self.meta_learning_feedback_policy = (
            MetaLearningFeedbackPolicy()
        )

        self.meta_learning_runtime = (
            MetaLearningRuntime(
                profile=self.meta_learning_profile,
                base_decision_maker=(
                    self.decision_maker
                ),
                feedback_engine=(
                    self.meta_learning_feedback_engine
                ),
                feedback_policy=(
                    self.meta_learning_feedback_policy
                ),
            )
        )

        self.unified_executor: Optional[
            UnifiedDecisionExecutor
        ] = None

        # =================================================
        # Procedural memory / learning
        # =================================================

        self.procedural_memory = (
            ProceduralMemoryStore(
                repository=(
                    self.persistent_memory.procedural
                    if self.persistent_memory is not None
                    else None
                )
            )
        )

        self.procedural_learning = (
            ProceduralLearning(
                self.procedural_memory
            )
        )

        # =================================================
        # Progress-aware learning
        # =================================================

        self.progress_learning = (
            ProgressAwareLearning()
        )

        self.last_progress_learning: Optional[
            ProgressLearningResult
        ] = None

        # =================================================
        # Experience
        # =================================================

        self.experience_store = (
            ExperienceStore()
        )

        self.experience_recorder = (
            ExperienceRecorder(
                self.experience_store
            )
        )

        self.current_experience: Optional[
            Experience
        ] = None

        # =================================================
        # Self model
        # =================================================

        self.self_model_store = (
            SelfModelStore(
                repository=(
                    self.persistent_memory.self_model
                    if self.persistent_memory is not None
                    else None
                )
            )
        )

        self.self_learning = (
            SelfLearning(
                self.self_model_store.get()
            )
        )

        # =================================================
        # Initial planner configuration
        # =================================================

        self.configure_planner()

    # =====================================================
    # Goal
    # =====================================================

    def set_goal(
        self,
        goal: Goal,
    ) -> None:

        if goal is None:
            raise ValueError(
                "goal cannot be None."
            )

        self.state.current_goal = goal

        self.working_memory.set_goal(
            goal.description
        )

    # =====================================================
    # Observation
    # =====================================================

    def observe(
        self,
        observation: Any,
    ) -> None:

        self.state.last_observation = (
            observation
        )

        if isinstance(
            observation,
            dict,
        ):
            self.working_memory.set_observation(
                observation
            )

    # =====================================================
    # Working memory
    # =====================================================

    def remember(
        self,
        item: Any,
    ) -> None:

        self.state.working_memory.append(
            item
        )

    def act(
        self,
        action: str,
    ) -> None:

        self.state.previous_action = (
            action
        )

        self.working_memory.set_current_action(
            action
        )

    def step(self) -> None:
        self.state.cycle += 1

    def set_plan(
        self,
        plan: list[str],
    ) -> None:

        self.working_memory.set_plan(
            plan
        )

    def set_reasoning(
        self,
        reasoning: str,
    ) -> None:

        self.working_memory.set_reasoning(
            reasoning
        )

    # =====================================================
    # Experience
    # =====================================================

    def record_experience(
        self,
        action: str,
        observation_before: dict,
        observation_after: dict,
        result: dict,
    ):

        goal = None

        if self.current_experience is not None:
            goal = self.current_experience.goal

        elif self.current_goal is not None:
            goal = self.current_goal.description

        episode = (
            self.episode_recorder.record(
                goal=goal,
                action=action,
                observation_before=(
                    observation_before
                ),
                observation_after=(
                    observation_after
                ),
                result=result,
                cycle=self.state.cycle,
            )
        )

        if self.current_experience is not None:
            self.experience_recorder.add_episode(
                episode
            )

        # -------------------------------------------------
        # Semantic learning
        # -------------------------------------------------

        self.semantic_learning.learn_from_episode(
            episode
        )

        # -------------------------------------------------
        # Online causal learning
        # -------------------------------------------------

        self.online_causal_learning.observe_episode(
            episode=episode,
            actions=self.actions,
        )

        # -------------------------------------------------
        # Reflection on failed action
        # -------------------------------------------------

        action_object = self.actions.get(
            action
        )

        if (
            action_object is not None
            and not result.get(
                "success",
                False,
            )
        ):

            reflection = (
                self.reflection_engine.reflect(
                    episode=episode,
                    action=action_object,
                )
            )

            if (
                reflection is not None
                and self.active_strategy is not None
            ):

                self.reflection_strategy_updater.apply(
                    strategy=self.active_strategy,
                    reflection=reflection,
                )

                self.reflection_memory.save_all()

                # Persist updated strategy too.
                self.strategy_store.save(
                    self.active_strategy
                )

        # -------------------------------------------------
        # Metacognitive reflection
        # -------------------------------------------------
        #
        # Run after the episode has been recorded and the
        # outcome is available. Goal progress is supplied
        # explicitly when a current goal exists; otherwise
        # the metacognition layer correctly treats progress
        # evidence as unavailable.
        # -------------------------------------------------

        progress_before = None
        progress_after = None

        if self.current_goal is not None:

            progress_before = (
                self.goal_progress(
                    world_state=(
                        observation_before
                    ),
                    goal=self.current_goal,
                )
            )

            progress_after = (
                self.goal_progress(
                    world_state=(
                        observation_after
                    ),
                    goal=self.current_goal,
                )
            )

        self.reflect_on_episode(
            episode,
            progress_before=progress_before,
            progress_after=progress_after,
        )

        return episode

    def reflect_on_episode(
        self,
        episode: Any,
        *,
        progress_before: float | None = None,
        progress_after: float | None = None,
    ) -> MetacognitiveReflection:
        """
        Generate and retain a structured metacognitive
        reflection for an observed episode.

        The metacognition engine does not mutate the episode
        or Agent state. Agent only exposes the latest result
        through ``last_metacognitive_reflection``.
        """

        reflection = (
            self.metacognition_engine.reflect_on_episode(
                episode,
                progress_before=progress_before,
                progress_after=progress_after,
            )
        )

        self.last_metacognitive_reflection = (
            reflection
        )

        return reflection

    def get_metacognitive_reflection(
        self,
    ) -> Optional[MetacognitiveReflection]:
        """
        Return the latest metacognitive reflection.
        """

        return self.last_metacognitive_reflection

    # =====================================================
    # LLM integration
    # =====================================================

    @property
    def has_llm(
        self,
    ) -> bool:
        """
        Return whether an LLM service is configured.
        """

        return self.llm_service is not None

    def set_llm_service(
        self,
        llm_service: Optional[LLMService],
    ) -> None:
        """
        Attach or detach the provider-neutral LLM service.

        Replacing the service clears the Agent's previous response
        trace so it cannot be mistaken for a response from the new
        provider.
        """

        self.llm_service = llm_service
        self.last_llm_request = None
        self.last_llm_response = None

    def llm_complete(
        self,
        messages: Iterable[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Complete an LLM request through the configured service.

        The Agent core remains provider-neutral. Concrete network/API
        behavior belongs to the injected LLM provider.
        """

        if self.llm_service is None:
            raise RuntimeError(
                "LLM service is not configured"
            )

        response = self.llm_service.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )

        self.last_llm_request = (
            self.llm_service.last_request
        )
        self.last_llm_response = response

        return response

    def llm_complete_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Convenience wrapper for a single text prompt.
        """

        if self.llm_service is None:
            raise RuntimeError(
                "LLM service is not configured"
            )

        response = self.llm_service.complete_text(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )

        self.last_llm_request = (
            self.llm_service.last_request
        )
        self.last_llm_response = response

        return response

    def get_last_llm_response(
        self,
    ) -> Optional[LLMResponse]:
        """
        Return the latest LLM response produced by this Agent.
        """

        return self.last_llm_response

    def clear_llm_trace(
        self,
    ) -> None:
        """
        Clear the Agent-level LLM request/response trace.
        """

        self.last_llm_request = None
        self.last_llm_response = None

        if self.llm_service is not None:
            self.llm_service.clear_trace()

    # =====================================================
    # Persistence
    # =====================================================

    @property
    def is_persistent(
        self,
    ) -> bool:

        return (
            self.persistent_memory is not None
        )

    def save_meta_learning(
        self,
    ) -> None:
        """
        Explicitly persist the current meta-learning
        profile.

        Persistent agents save automatically after each
        meta-learning event, but this method provides an
        explicit lifecycle flush point for callers.
        """

        self._persist_meta_learning_profile()

    def memory_counts(
        self,
    ) -> dict[str, int]:

        counts = {
            "episodic": (
                self.episodic_memory.count()
            ),
            "semantic": (
                self.semantic_memory.count()
            ),
            "causal": (
                self.causal_memory.count()
            ),
            "reflection": (
                self.reflection_memory.count()
            ),
            "procedural": (
                self.procedural_memory.count()
            ),
        }

        counts["self_model"] = (
            1
            if (
                self.persistent_memory is not None
                and self.persistent_memory.self_model.exists()
            )
            else 0
        )

        return counts

    def clear_memory(
        self,
    ) -> None:

        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.causal_memory.clear()
        self.reflection_memory.clear()
        self.last_metacognitive_reflection = None

    # =====================================================
    # Current goal
    # =====================================================

    @property
    def current_goal(
        self,
    ) -> Optional[Goal]:

        return self.state.current_goal

    # =====================================================
    # Goal completion / progress
    # =====================================================

    def is_goal_satisfied(
        self,
        world_state: Optional[
            dict[str, Any]
        ] = None,
        goal: Optional[Goal] = None,
    ) -> bool:

        if goal is None:
            goal = self.current_goal

        if goal is None:
            return False

        if world_state is None:

            if isinstance(
                self.state.last_observation,
                dict,
            ):
                world_state = (
                    self.state.last_observation
                )

            else:
                world_state = {}

        return goal.is_satisfied(
            world_state
        )

    def goal_progress(
        self,
        world_state: Optional[
            dict[str, Any]
        ] = None,
        goal: Optional[Goal] = None,
    ) -> float:

        if goal is None:
            goal = self.current_goal

        if goal is None:
            return 0.0

        if world_state is None:

            if isinstance(
                self.state.last_observation,
                dict,
            ):
                world_state = (
                    self.state.last_observation
                )

            else:
                world_state = {}

        target_state = (
            goal.target_state
        )

        if not target_state:
            return 0.0

        satisfied_count = 0

        for key, expected_value in (
            target_state.items()
        ):

            if (
                world_state.get(key)
                == expected_value
            ):
                satisfied_count += 1

        progress = (
            satisfied_count
            / len(target_state)
        )

        return round(
            progress,
            6,
        )

    def update_goal_status(
        self,
        world_state: Optional[
            dict[str, Any]
        ] = None,
        goal: Optional[Goal] = None,
    ) -> bool:

        if goal is None:
            goal = self.current_goal

        if goal is None:
            return False

        satisfied = (
            self.is_goal_satisfied(
                world_state=world_state,
                goal=goal,
            )
        )

        if (
            satisfied
            and not goal.completed
        ):
            goal.complete()

        return satisfied

    # =====================================================
    # Knowledge uncertainty
    # =====================================================

    def _knowledge_uncertainty(
        self,
        goal: Optional[Goal] = None,
    ) -> float:
        """
        Estimate uncertainty for a specific goal.

        Procedural knowledge belonging to another goal
        does not reduce uncertainty for this goal.
        """

        if goal is None:
            goal = self.current_goal

        if goal is None:
            return 1.0

        skill = (
            goal.description
            .strip()
            .lower()
            .replace(" ", "_")
        )

        procedural = None

        for memory in (
            self.procedural_memory.all()
        ):

            if memory.skill == skill:
                procedural = memory
                break

        if procedural is None:
            return 1.0

        evidence = (
            procedural.success_count
            + procedural.failure_count
        )

        if evidence <= 0:
            return 1.0

        uncertainty = (
            1.0
            / ((evidence + 1) ** 0.5)
        )

        failure_rate = (
            procedural.failure_count
            / evidence
        )

        uncertainty = min(
            1.0,
            uncertainty
            + (
                0.5
                * failure_rate
            ),
        )

        return round(
            uncertainty,
            6,
        )

    # =====================================================
    # Actions
    # =====================================================

    def register_action(
        self,
        action: Action,
    ) -> None:

        if action is None:
            raise ValueError(
                "action cannot be None."
            )

        self.actions[
            action.name
        ] = action

        self.configure_planner()

    def register_actions(
        self,
        actions: list[Action],
    ) -> None:

        for action in actions:
            self.register_action(
                action
            )

    # =====================================================
    # Strategy registry synchronization
    # =====================================================

    def _refresh_strategy_registry(
        self,
    ) -> None:
        """
        Refresh the compatibility strategy_registry
        from the canonical StrategyMemoryStore.

        StrategyMemoryStore remains the single source
        of truth for registered strategies.
        """

        self.strategy_registry = (
            self.strategy_store.all()
        )

    # =====================================================
    # Strategy registration
    # =====================================================

    def register_strategy(
        self,
        strategy: Strategy,
    ) -> None:

        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        existing = (
            self.strategy_store.find_matching(
                strategy
            )
        )

        if existing is not None:
            self._refresh_strategy_registry()
            return

        self.strategy_store.add(
            strategy
        )

        self._refresh_strategy_registry()

    def register_strategies(
        self,
        strategies: list[Strategy],
    ) -> None:

        if strategies is None:
            raise ValueError(
                "strategies cannot be None."
            )

        for strategy in strategies:
            self.register_strategy(
                strategy
            )

    def get_registered_strategies(
        self,
    ) -> list[Strategy]:

        return list(
            self.strategy_registry
        )

    # =====================================================
    # Active strategy
    # =====================================================

    def set_active_strategy(
        self,
        strategy: Optional[Strategy],
    ) -> None:

        self.active_strategy = strategy

        if strategy is not None:
            self.strategy_store.add(
                strategy
            )

            self._refresh_strategy_registry()

    def get_active_strategy(
        self,
    ) -> Optional[Strategy]:

        return self.active_strategy

    # =====================================================
    # Strategy attribution
    # =====================================================

    def attribute_strategy(
        self,
        experience: Experience,
        strategies: Optional[
            list[Strategy]
        ] = None,
    ) -> StrategyAttributionResult:

        candidates = (
            strategies
            if strategies is not None
            else self.strategy_registry
        )

        result = (
            self.strategy_attributor.attribute(
                experience=experience,
                strategies=candidates,
            )
        )

        # Explicit strategy always has precedence.
        if self.active_strategy is not None:

            return StrategyAttributionResult(
                strategy=self.active_strategy,
                matched=True,
                reason=(
                    "explicit active strategy"
                ),
            )

        if (
            result.matched
            and result.strategy is not None
        ):
            self.active_strategy = (
                result.strategy
            )

        return result

    # =====================================================
    # Strategy lifecycle / consolidation
    # =====================================================

    def consolidate_strategies(
        self,
    ):

        result = (
            self.strategy_consolidator.consolidate(
                self.strategy_registry
            )
        )

        self.strategy_registry = (
            result.strategies
        )

        self.strategy_store.clear()

        for strategy in (
            self.strategy_registry
        ):
            self.strategy_store.add(
                strategy
            )

        self._refresh_strategy_registry()

        return result

    def evaluate_strategy_lifecycle(
        self,
    ):
        """
        Evaluate strategy lifecycle and attempt recovery
        of paused strategies.
        """

        paused_before = {
            strategy.name
            for strategy in self.strategy_registry
            if strategy.is_paused
        }

        recovery_results = (
            self.strategy_recovery.recover_all(
                self.strategy_registry
            )
        )

        for recovery_result in (
            recovery_results
        ):
            strategy = (
                recovery_result.strategy
            )

            if (
                strategy.name
                in paused_before
            ):
                self._record_meta_recovery(
                    success=(
                        recovery_result.recovered
                    )
                )

        lifecycle = (
            self.strategy_lifecycle.evaluate_all(
                self.strategy_registry
            )
        )

        self._refresh_strategy_registry()

        self.strategy_store.save_all()

        return lifecycle

    def consolidate_and_evaluate_strategies(
        self,
    ):

        consolidation = (
            self.consolidate_strategies()
        )

        lifecycle = (
            self.evaluate_strategy_lifecycle()
        )

        return (
            consolidation,
            lifecycle,
        )

    # =====================================================
    # Strategy progress attribution
    # =====================================================

    def _update_strategy_from_experience(
        self,
        experience: Experience,
        success: bool,
    ) -> None:

        attribution = (
            self.attribute_strategy(
                experience=experience
            )
        )

        strategy = attribution.strategy

        if strategy is None:
            return

        progress_delta = 0.0

        if (
            self.last_progress_learning
            is not None
        ):
            progress_delta = (
                self.last_progress_learning
                .progress_delta
            )

        evidence = (
            self.strategy_evolution_evidence.get(
                strategy.name
            )
        )

        if evidence is not None:

            evidence.record_trial(
                success=success,
                progress_delta=progress_delta,
                experience_id=(
                    experience.experience_id
                ),
            )

        if success:

            strategy.record_success(
                experience_id=(
                    experience.experience_id
                ),
                action_count=(
                    experience.action_count
                ),
                progress_delta=(
                    progress_delta
                ),
            )

        else:

            strategy.record_failure(
                experience_id=(
                    experience.experience_id
                ),
                action_count=(
                    experience.action_count
                ),
                progress_delta=(
                    progress_delta
                ),
            )

        self.strategy_store.save(
            strategy
        )

    # =====================================================
    # Strategy evolution
    # =====================================================

    def evolve_strategy_from_experience(
        self,
        experience: Optional[Experience] = None,
        world_state: Optional[
            dict[str, Any]
        ] = None,
        goal: Optional[Goal] = None,
    ) -> Optional[
        StrategyEvolutionSelectionResult
    ]:

        if experience is None:
            experience = (
                self.current_experience
            )

        if experience is None:
            return None

        if not experience.failed:
            return None

        if goal is None:
            goal = self.current_goal

        if goal is None:
            return None

        attribution = (
            self.attribute_strategy(
                experience=experience
            )
        )

        source_strategy = (
            attribution.strategy
        )

        if source_strategy is None:
            return None

        if source_strategy.is_retired:
            return None

        if world_state is None:

            if experience.episodes:

                first_episode = (
                    experience.episodes[0]
                )

                candidate_world_state = getattr(
                    first_episode,
                    "observation_before",
                    None,
                )

                if isinstance(
                    candidate_world_state,
                    dict,
                ):
                    world_state = (
                        candidate_world_state
                    )

            if world_state is None:

                if isinstance(
                    self.state.last_observation,
                    dict,
                ):
                    world_state = (
                        self.state.last_observation
                    )

                else:
                    world_state = {}

        goal_state = dict(
            goal.target_state
        )

        if not goal_state:
            return None

        evolution = StrategyEvolution(
            actions=self.actions.values(),
        )

        candidates = (
            evolution.evolve_candidates(
                strategy=source_strategy,
                experience=experience,
            )
        )

        if not candidates:

            result = (
                StrategyEvolutionPipeline(
                    validator=(
                        StrategyCandidateValidator(
                            actions=(
                                self.actions.values()
                            ),
                            strategies=(
                                self.strategy_registry
                            ),
                        )
                    )
                ).select(
                    source_strategy=source_strategy,
                    candidates=[],
                    world_state=world_state,
                    goal_state=goal_state,
                )
            )

            self._record_meta_evolution(
                success=False
            )

            return result

        validator = (
            StrategyCandidateValidator(
                actions=self.actions.values(),
                strategies=self.strategy_registry,
            )
        )

        pipeline = (
            StrategyEvolutionPipeline(
                validator=validator,
            )
        )

        result = pipeline.select(
            source_strategy=source_strategy,
            candidates=candidates,
            world_state=world_state,
            goal_state=goal_state,
        )

        selected_strategy = (
            result.selection.strategy
        )

        if selected_strategy is not None:

            existing = (
                self.strategy_store.find_matching(
                    selected_strategy
                )
            )

            if existing is not None:
                selected_strategy = existing

            else:
                self.strategy_store.add(
                    selected_strategy
                )

                self._refresh_strategy_registry()

            self._create_evolution_evidence(
                strategy=selected_strategy,
                source_strategy=source_strategy,
                experience=experience,
            )

            self._record_meta_evolution(
                success=True
            )

        else:

            self._record_meta_evolution(
                success=False
            )

        return result

    # =====================================================
    # Planner configuration
    # =====================================================

    def configure_planner(
        self,
    ) -> None:

        self.planner = Planner(
            actions=list(
                self.actions.values()
            )
        )

        self.adaptive_planner = (
            AdaptivePlanner(
                planner=self.planner,
                strategy_selector=(
                    self.strategy_selector
                ),
                risk_strategy_selector=(
                    self.risk_strategy_selector
                ),
            )
        )

        self.hierarchical_planner = (
            HierarchicalPlanner(
                planner=self.planner,
                decomposer=self.goal_decomposer,
                procedural_store=(
                    self.procedural_memory
                ),
                causal_store=(
                    self.causal_memory
                ),
            )
        )

        self.risk_aware_hierarchical_planner = (
            RiskAwareHierarchicalPlanner(
                self.hierarchical_planner,
                self.risk_strategy_selector,
            )
        )

        self.unified_executor = (
            UnifiedDecisionExecutor(
                explorer=self.active_explorer,
                hierarchical_planner=(
                    self.risk_aware_hierarchical_planner
                ),
            )
        )

    # =====================================================
    # Existing planning API
    # =====================================================

    def choose_plan(
        self,
        world_state: dict[str, Any],
        goal_state: dict[str, Any],
        strategies: list[Strategy],
        goal: Optional[str] = None,
    ):

        if self.adaptive_planner is None:
            self.configure_planner()

        return self.adaptive_planner.plan(
            world_state=world_state,
            goal_state=goal_state,
            goal=goal,
            strategies=strategies,
        )

    # =====================================================
    # Agent Self-Awareness
    # =====================================================
    #
    # The Agent exposes self-awareness through a facade.
    #
    # The facade owns:
    #
    #   Capability Confidence
    #          ↓
    #   Self Uncertainty
    #          ↓
    #   Decision Influence
    #          ↓
    #   Persistence
    #
    # =====================================================

    def evaluate_self_awareness(
        self,
        capabilities,
        *,
        persist: bool = True,
    ) -> AgentSelfAwarenessSnapshot:
        """
        Evaluate the Agent's self-awareness from observed
        capability evidence.

        Manual self-awareness evaluation invalidates any previously
        calibrated decision influence because the underlying
        self-model has changed.
        """

        snapshot = (
            self.self_awareness.evaluate(
                capabilities,
                persist=persist,
            )
        )

        self.last_self_awareness_calibration = None
        self.last_calibrated_self_aware_influence = None
        self.last_calibrated_self_aware_result = None

        return snapshot


    def restore_self_awareness(
        self,
    ) -> AgentSelfAwarenessSnapshot | None:
        """
        Restore persisted self-awareness.

        Any previously calibrated runtime influence is invalidated
        because the self-model has been restored.
        """

        snapshot = (
            self.self_awareness.restore()
        )

        self.last_self_awareness_calibration = None
        self.last_calibrated_self_aware_influence = None
        self.last_calibrated_self_aware_result = None

        return snapshot

    def save_self_awareness(
        self,
    ):
        """
        Explicitly persist the current self-awareness state.

        Raises ValueError when persistence is not configured
        or no self-awareness state has been evaluated.
        """

        return self.self_awareness.save()

    def get_self_awareness_snapshot(
        self,
    ) -> AgentSelfAwarenessSnapshot:
        """
        Return the current self-awareness snapshot.
        """

        return (
            self.self_awareness.snapshot()
        )

    def reset_self_awareness(
        self,
        *,
        clear_persistence: bool = False,
    ) -> None:
        """
        Reset self-awareness and invalidate calibrated decision
        influence.
        """

        self.self_awareness.reset(
            clear_persistence=clear_persistence
        )

        self.last_self_awareness_calibration = None
        self.last_calibrated_self_aware_influence = None
        self.last_calibrated_self_aware_result = None

    def restore_learned_state(
        self,
    ) -> UnifiedLearnedStateRestorationResult:
        """
        Restore the durable learned state of the Agent.

        Restoration order:

            1. Self-awareness
            2. Calibrated self-aware influence
            3. Consistency validation

        A calibrated influence is only restored when the base
        self-awareness state is also present and consistent.

        Runtime telemetry and feedback traces are intentionally
        not restored here.
        """

        # -------------------------------------------------
        # Always clear transient calibration cache first.
        # -------------------------------------------------

        self.last_self_awareness_calibration = None
        self.last_calibrated_self_aware_influence = None
        self.last_calibrated_self_aware_result = None

        # -------------------------------------------------
        # Restore base self-awareness.
        # -------------------------------------------------

        self_awareness_snapshot = (
            self.self_awareness.restore()
        )

        self_awareness_restored = (
            self_awareness_snapshot is not None
        )

        self_awareness_version = None

        if self_awareness_snapshot is not None:

            self_awareness_version = getattr(
                self_awareness_snapshot,
                "persisted_version",
                None,
            )

            if self_awareness_version is None:

                self_awareness_version = getattr(
                    self_awareness_snapshot,
                    "version",
                    None,
                )

        # -------------------------------------------------
        # Restore calibrated state from independent store.
        # -------------------------------------------------

        calibrated_snapshot = None

        if (
            self.calibrated_self_awareness_store
            is not None
        ):

            calibrated_snapshot = (
                self.calibrated_self_awareness_store
                .load(self.name)
            )

        calibrated_version = None

        if calibrated_snapshot is not None:

            calibrated_version = (
                calibrated_snapshot.version
            )

        # -------------------------------------------------
        # No self-awareness at all.
        # -------------------------------------------------

        if not self_awareness_restored:

            if calibrated_snapshot is None:

                result = (
                    UnifiedLearnedStateRestorationResult(
                        agent_name=self.name,
                        self_awareness_restored=False,
                        calibrated_influence_restored=False,
                        self_awareness_version=None,
                        calibrated_influence_version=None,
                        consistent=True,
                        reason=(
                            "No persisted learned state found."
                        ),
                    )
                )

            else:

                # We deliberately refuse to restore a calibration
                # whose base self-awareness is unavailable.
                result = (
                    UnifiedLearnedStateRestorationResult(
                        agent_name=self.name,
                        self_awareness_restored=False,
                        calibrated_influence_restored=False,
                        self_awareness_version=None,
                        calibrated_influence_version=(
                            calibrated_version
                        ),
                        consistent=False,
                        reason=(
                            "Persisted calibrated influence exists "
                            "but base self-awareness is unavailable; "
                            "calibrated influence was not restored."
                        ),
                    )
                )

            self.last_unified_learned_state_restoration = (
                result
            )

            return result

        # -------------------------------------------------
        # Base self-awareness exists, but calibration does not.
        # -------------------------------------------------

        if calibrated_snapshot is None:

            result = (
                UnifiedLearnedStateRestorationResult(
                    agent_name=self.name,
                    self_awareness_restored=True,
                    calibrated_influence_restored=False,
                    self_awareness_version=(
                        self_awareness_version
                    ),
                    calibrated_influence_version=None,
                    consistent=True,
                    reason=(
                        "Self-awareness restored; "
                        "no persisted calibrated influence found."
                    ),
                )
            )

            self.last_unified_learned_state_restoration = (
                result
            )

            return result

        # -------------------------------------------------
        # Validate calibration against restored self-awareness.
        # -------------------------------------------------

        calibration_result = (
            calibrated_snapshot.calibration_result
        )

        restored_self_uncertainty = getattr(
            self_awareness_snapshot,
            "self_uncertainty",
            None,
        )

        restored_confidence = None

        if restored_self_uncertainty is not None:

            restored_confidence = getattr(
                restored_self_uncertainty,
                "confidence",
                None,
            )

        persisted_raw_confidence = getattr(
            calibration_result,
            "raw_confidence",
            None,
        )

        confidence_consistent = True

        if (
            restored_confidence is not None
            and persisted_raw_confidence is not None
        ):

            confidence_consistent = (
                abs(
                    float(restored_confidence)
                    - float(
                        persisted_raw_confidence
                    )
                )
                <= 1e-6
            )

        if not confidence_consistent:

            result = (
                UnifiedLearnedStateRestorationResult(
                    agent_name=self.name,
                    self_awareness_restored=True,
                    calibrated_influence_restored=False,
                    self_awareness_version=(
                        self_awareness_version
                    ),
                    calibrated_influence_version=(
                        calibrated_version
                    ),
                    consistent=False,
                    reason=(
                        "Persisted calibration does not match "
                        "the restored self-awareness confidence; "
                        "calibrated influence was not restored."
                    ),
                )
            )

            self.last_unified_learned_state_restoration = (
                result
            )

            return result

        # -------------------------------------------------
        # Restore calibrated influence.
        # -------------------------------------------------

        self.last_calibrated_self_aware_influence = (
            calibrated_snapshot.influence
        )

        self.last_calibrated_self_aware_result = (
            calibrated_snapshot.calibration_result
        )

        result = (
            UnifiedLearnedStateRestorationResult(
                agent_name=self.name,
                self_awareness_restored=True,
                calibrated_influence_restored=True,
                self_awareness_version=(
                    self_awareness_version
                ),
                calibrated_influence_version=(
                    calibrated_version
                ),
                consistent=True,
                reason=(
                    "Self-awareness and calibrated "
                    "self-aware influence restored."
                ),
            )
        )

        self.last_unified_learned_state_restoration = (
            result
        )

        return result

    def reset_learned_state_runtime(
        self,
    ) -> None:
        """
        Clear transient calibrated learned-state caches without
        deleting persisted learned state.
        """

        self.last_self_awareness_calibration = None
        self.last_calibrated_self_aware_influence = None
        self.last_calibrated_self_aware_result = None
        self.last_unified_learned_state_restoration = None

    def get_learned_state_restoration(
        self,
    ) -> UnifiedLearnedStateRestorationResult | None:
        """
        Return the latest unified learned-state restoration result.
        """

        return (
            self.last_unified_learned_state_restoration
        )

    def record_capability_telemetry(
        self,
        capability,
        success: bool | None = None,
        *,
        source: str = "manual",
    ) -> CapabilityTelemetryEvent:
        """
        Record explicit evidence for an Agent capability.

        This layer records telemetry only. It does not
        automatically modify self-awareness yet.

        Example:

            agent.record_capability_telemetry(
                "planning",
                success=True,
                source="planner",
            )
        """

        event = (
            self.capability_telemetry.record(
                capability,
                success,
                source=source,
            )
        )

        self.last_capability_telemetry_event = (
            event
        )

        return event


    def get_capability_telemetry_evidence(
        self,
    ) -> dict[str, dict[str, int]]:
        """
        Return aggregated capability evidence.
        """

        return (
            self.capability_telemetry.evidence()
        )


    def get_capability_telemetry_snapshot(
        self,
    ) -> CapabilityTelemetrySnapshot:
        """
        Return the current capability telemetry snapshot.
        """

        return (
            self.capability_telemetry.snapshot()
        )


    def get_capability_telemetry_events(
        self,
    ) -> list[CapabilityTelemetryEvent]:
        """
        Return all observed capability telemetry events.
        """

        return (
            self.capability_telemetry.events()
        )


    def reset_capability_telemetry(
        self,
    ) -> None:
        """
        Clear all capability telemetry.
        """

        self.capability_telemetry.clear()

        self.last_capability_telemetry_event = (
            None
        )

    def _record_decision_capability_telemetry(
        self,
        decision,
        success: bool,
    ) -> None:
        """
        Record explicit capability evidence produced by a
        runtime decision.

        The following capabilities are currently supported:

            decision_making
            execution
            goal_achievement

        Additionally, one mode-specific capability is recorded:

            exploration
            exploitation

        Planning and reasoning are intentionally not inferred
        here because the current runtime does not expose enough
        direct evidence to make those measurements trustworthy.
        """

        source = "decide_and_act"

        self.record_capability_telemetry(
            "decision_making",
            success,
            source=source,
        )

        self.record_capability_telemetry(
            "execution",
            success,
            source=source,
        )

        self.record_capability_telemetry(
            "goal_achievement",
            success,
            source=source,
        )

        if decision.mode == DecisionMode.EXPLORE:

            self.record_capability_telemetry(
                "exploration",
                success,
                source=source,
            )

        elif decision.mode == DecisionMode.EXPLOIT:

            self.record_capability_telemetry(
                "exploitation",
                success,
                source=source,
            )

    def update_self_awareness_from_experience(
        self,
        *,
        persist: bool = True,
    ) -> ExperienceSelfAwarenessUpdateResult:
        """
        Refresh self-awareness from the complete episodic
        experience history.

        This converts previous experiences into capability
        evidence and updates the self-awareness pipeline:

            Experience
                ↓
            Capability Evidence
                ↓
            Capability Confidence
                ↓
            Self Uncertainty
                ↓
            Decision Influence
        """

        result = (
            self.experience_self_awareness_updater.update(
                self.episodic_memory.all(),
                persist=persist,
            )
        )

        self.last_experience_self_awareness_update = (
            result
        )

        return result

    def get_last_experience_self_awareness_update(
        self,
    ) -> ExperienceSelfAwarenessUpdateResult | None:
        """
        Return the latest experience-driven
        self-awareness update.
        """

        return (
            self.last_experience_self_awareness_update
        )


    def build_unified_capability_evidence(
        self,
    ) -> UnifiedCapabilityEvidenceResult:
        """
        Build unified capability evidence from:

            1. explicit runtime telemetry
            2. experience-derived evidence

        Explicit telemetry takes priority when the same
        capability exists in both sources.
        """

        experience_evidence = (
            self.experience_self_awareness_updater
            .build_capability_evidence(
                self.episodic_memory.all()
            )
        )

        telemetry_evidence = (
            self.capability_telemetry.evidence()
        )

        result = (
            self.unified_capability_evidence_builder
            .build(
                telemetry_evidence=(
                    telemetry_evidence
                ),
                experience_evidence=(
                    experience_evidence
                ),
            )
        )

        self.last_unified_capability_evidence = (
            result
        )

        return result


    def update_self_awareness_from_all_evidence(
        self,
        *,
        persist: bool = True,
    ):
        """
        Update self-awareness from the unified evidence model.

        Evidence flow:

            Experience
                +
            Explicit Telemetry
                ↓
            Unified Capability Evidence
                ↓
            Capability Confidence
                ↓
            Self Uncertainty
                ↓
            Decision Influence
        """

        unified = (
            self.build_unified_capability_evidence()
        )

        if not unified.has_evidence:
            return None

        return (
            self.self_awareness.evaluate(
                unified.capabilities,
                persist=persist,
            )
        )


    def get_unified_capability_evidence(
        self,
    ) -> UnifiedCapabilityEvidenceResult | None:
        """
        Return the most recently built unified capability
        evidence result.
        """

        return (
            self.last_unified_capability_evidence
        )

    def record_self_awareness_feedback(
        self,
    ) -> SelfAwarenessFeedbackRecord | None:
        """
        Record the current self-awareness state together with
        the latest self-aware decision arbitration result.

        This is an observability operation only.
        """

        uncertainty = getattr(
            self.self_awareness,
            "self_uncertainty",
            None,
        )

        if uncertainty is None:
            return None

        # influence = getattr(
        #     self.self_awareness,
        #     "decision_influence",
        #     None,
        # )
        influence = (
            self.last_calibrated_self_aware_influence
        )

        if influence is None:

            influence = getattr(
                self.self_awareness,
                "decision_influence",
                None,
            )

        decision_result = (
            self.last_self_aware_decision_result
        )

        preferred_mode = None
        selected_mode = None
        mode_changed = False

        if influence is not None:
            preferred_mode = (
                influence.preferred_mode
            )

        if decision_result is not None:
            selected_mode = (
                decision_result.selected_mode
            )

            mode_changed = bool(
                decision_result.mode_changed
            )

        return (
            self.self_awareness_feedback.record(
                uncertainty=uncertainty.uncertainty,
                confidence=uncertainty.confidence,
                preferred_mode=preferred_mode,
                selected_mode=selected_mode,
                evidence_count=uncertainty.capability_count,
                mode_changed=mode_changed,
            )
        )


    def get_self_awareness_feedback_snapshot(
        self,
    ) -> SelfAwarenessFeedbackSnapshot:
        """
        Return the current self-awareness feedback trace.
        """

        return (
            self.self_awareness_feedback.snapshot()
        )


    def reset_self_awareness_feedback(
        self,
    ) -> None:
        """
        Clear the self-awareness feedback trace.
        """

        self.self_awareness_feedback.clear()

    # def calibrate_self_awareness(
    #     self,
    # ) -> SelfAwarenessCalibrationResult | None:
    #     """
    #     Calibrate the current self-confidence against unified
    #     observed capability outcomes.

    #     This operation does not modify the current decision
    #     influence yet. It only measures calibration quality.
    #     """

    #     self_uncertainty = getattr(
    #         self.self_awareness,
    #         "self_uncertainty",
    #         None,
    #     )

    #     if self_uncertainty is None:
    #         return None

    #     unified = (
    #         self.build_unified_capability_evidence()
    #     )

    #     if not unified.has_evidence:
    #         return None

    #     result = (
    #         self.self_awareness_calibrator.calibrate(
    #             raw_confidence=(
    #                 self_uncertainty.confidence
    #             ),
    #             evidence=(
    #                 unified.capabilities
    #             ),
    #         )
    #     )

    #     self.last_self_awareness_calibration = (
    #         result
    #     )

    #     return result
    def calibrate_self_awareness(
        self,
    ) -> SelfAwarenessCalibrationResult | None:
        """
        Calibrate current self-confidence against observed
        capability outcomes and prepare a corrected self-aware
        decision influence for the NEXT decision cycle.
        """

        self_uncertainty = getattr(
            self.self_awareness,
            "self_uncertainty",
            None,
        )

        if self_uncertainty is None:
            self.last_self_awareness_calibration = None
            self.last_calibrated_self_aware_influence = None
            self.last_calibrated_self_aware_result = None
            return None

        unified = (
            self.build_unified_capability_evidence()
        )

        if not unified.has_evidence:
            self.last_self_awareness_calibration = None
            self.last_calibrated_self_aware_influence = None
            self.last_calibrated_self_aware_result = None
            return None

        result = (
            self.self_awareness_calibrator.calibrate(
                raw_confidence=(
                    self_uncertainty.confidence
                ),
                evidence=(
                    unified.capabilities
                ),
            )
        )

        self.last_self_awareness_calibration = (
            result
        )

        raw_influence = getattr(
            self.self_awareness,
            "decision_influence",
            None,
        )

        if raw_influence is None:

            self.last_calibrated_self_aware_influence = (
                None
            )

            self.last_calibrated_self_aware_result = (
                None
            )

            return result

        calibrated_result = (
            self.calibrated_self_aware_influence_applier
            .apply(
                influence=raw_influence,
                calibration=result,
            )
        )

        self.last_calibrated_self_aware_result = (
            calibrated_result
        )

        self.last_calibrated_self_aware_influence = (
            calibrated_result.influence
        )

        if (
            self.calibrated_self_awareness_store
            is not None
        ):
            self.calibrated_self_awareness_store.save(
                agent_name=self.name,
                influence=(
                    calibrated_result.influence
                ),
                calibration_result=(
                    calibrated_result
                ),
            )

        return result

    def get_calibrated_self_aware_influence(
        self,
    ):
        """
        Return the corrected decision influence prepared for
        the next decision cycle.
        """

        return (
            self.last_calibrated_self_aware_influence
        )


    def get_calibrated_self_aware_result(
        self,
    ) -> CalibratedSelfAwareInfluenceResult | None:
        """
        Return the latest calibration-to-influence result.
        """

        return (
            self.last_calibrated_self_aware_result
        )


    def get_self_awareness_calibration(
        self,
    ) -> SelfAwarenessCalibrationResult | None:
        """
        Return the latest self-awareness calibration result.
        """

        return (
            self.last_self_awareness_calibration
        )

    def restore_calibrated_self_awareness(
        self,
    ) -> CalibratedSelfAwarenessPersistenceSnapshot | None:
        """
        Restore the latest persisted calibrated self-awareness
        influence.

        Returns None when persistence is disabled or there is
        no previously persisted calibration.
        """

        if (
            self.calibrated_self_awareness_store
            is None
        ):
            return None

        persisted = (
            self.calibrated_self_awareness_store
            .load(self.name)
        )

        if persisted is None:
            return None

        self.last_calibrated_self_aware_influence = (
            persisted.influence
        )

        self.last_calibrated_self_aware_result = (
            persisted.calibration_result
        )

        return persisted

    def get_calibrated_self_awareness_persistence(
        self,
    ):
        """
        Return the latest persisted calibrated
        self-awareness snapshot.
        """

        if (
            self.calibrated_self_awareness_store
            is None
        ):
            return None

        return (
            self.calibrated_self_awareness_store
            .load(self.name)
        )

    # =====================================================
    # Self-aware decision inspection
    # =====================================================

    def get_self_aware_decision_result(
        self,
    ):
        """
        Return the latest self-aware decision arbitration
        result from decide_and_act().

        Returns None when the current/last decision did not
        use an available self-awareness influence.
        """

        return self.last_self_aware_decision_result

    # =====================================================
    # Meta-learning runtime
    # =====================================================

    def evaluate_meta_learning_runtime(
        self,
        uncertainty: float,
        information_gain: float,
        risk: float,
        explore_allowed: bool = True,
        exploit_allowed: bool = True,
    ):
        """
        Evaluate runtime decision signals through the
        meta-learning feedback pipeline.

        This method does not execute an action and does
        not record an outcome.
        """

        return (
            self.meta_learning_runtime.evaluate_from_values(
                uncertainty=uncertainty,
                information_gain=information_gain,
                risk=risk,
                explore_allowed=explore_allowed,
                exploit_allowed=exploit_allowed,
            )
        )

    # =====================================================
    # Unified decide + act
    # =====================================================

    def decide_and_act(
        self,
        goal: Optional[Goal] = None,
        world_state: Optional[
            dict[str, Any]
        ] = None,
        strategies: Optional[
            list[Strategy]
        ] = None,
        environment: Any = None,
        information_gain: Optional[
            dict[str, float]
        ] = None,
        action_risk: Optional[
            dict[str, float]
        ] = None,
        risk: float = 0.0,
        action_cost: Optional[
            dict[str, float]
        ] = None,
        experience_id: Optional[str] = None,
    ):
        """
        Top-level Agent decision and execution.

        Flow:

            Goal
              ↓
            Knowledge
              ↓
            Meta-learning feedback
              ↓
            Adaptive decision
              ↓
            Self-aware arbitration
              ↓
            EXPLORE / EXPLOIT
              ↓
            Execute
              ↓
            Observe
              ↓
            Goal status
              ↓
            Meta-learning outcome

        Self-aware arbitration only applies when a
        self-awareness DecisionInfluence is currently
        available. Otherwise the pre-existing decision path
        remains unchanged.
        """

        # -------------------------------------------------
        # Resolve goal
        # -------------------------------------------------

        if goal is None:
            goal = self.current_goal

        if goal is None:
            raise ValueError(
                "A goal is required for "
                "decide_and_act()."
            )

        self.set_goal(
            goal
        )

        # -------------------------------------------------
        # Resolve world state
        # -------------------------------------------------

        if world_state is None:

            if environment is not None:
                world_state = (
                    environment.observe()
                )

            else:
                world_state = {}

        # -------------------------------------------------
        # Available strategies
        # -------------------------------------------------

        available_strategies = (
            self.get_available_strategies()
        )

        # -------------------------------------------------
        # Defaults
        # -------------------------------------------------

        if strategies is None:
            strategies = (
                available_strategies
            )

        else:
            strategies = [
                strategy
                for strategy in strategies
                if strategy.is_active
            ]

        information_gain = (
            information_gain
            if information_gain is not None
            else {}
        )

        action_risk = (
            action_risk
            if action_risk is not None
            else {}
        )

        action_cost = (
            action_cost
            if action_cost is not None
            else {}
        )

        # -------------------------------------------------
        # Ensure executor
        # -------------------------------------------------

        if self.unified_executor is None:
            self.configure_planner()

        # -------------------------------------------------
        # Goal-specific uncertainty
        # -------------------------------------------------

        uncertainty = (
            self._knowledge_uncertainty(
                goal=goal
            )
        )

        # -------------------------------------------------
        # Expected information gain
        # -------------------------------------------------

        expected_information_gain = max(
            information_gain.values(),
            default=uncertainty,
        )

        # -------------------------------------------------
        # Meta-learning aware decision
        # -------------------------------------------------

        meta_runtime = (
            self.evaluate_meta_learning_runtime(
                uncertainty=uncertainty,
                information_gain=(
                    expected_information_gain
                ),
                risk=risk,
                explore_allowed=True,
                exploit_allowed=True,
            )
        )

        meta_decision = (
            meta_runtime.decision
        )

        # -------------------------------------------------
        # Preserve the original AgentDecision contract.
        # -------------------------------------------------

        decision = (
            meta_decision.base_decision
        )

        if decision is None:
            raise RuntimeError(
                "Meta-learning decision did not "
                "preserve the base AgentDecision."
            )

        # -------------------------------------------------
        # Adaptive mode integration.
        #
        # Only mode and reason are changed. All other
        # AgentDecision fields remain untouched.
        # -------------------------------------------------

        if (
            meta_decision.mode
            != meta_decision.base_mode
        ):

            decision = replace(
                decision,
                mode=DecisionMode(
                    meta_decision.mode
                ),
                reason=(
                    meta_decision.reason
                ),
            )

        # -------------------------------------------------
        # Self-aware decision arbitration
        # -------------------------------------------------
        #
        # This is intentionally placed AFTER the existing
        # meta-learning decision layer.
        #
        # Therefore:
        #
        #   Meta-learning
        #       ↓
        #   Self-awareness
        #       ↓
        #   Executor
        #
        # Self-awareness does not bypass or replace
        # meta-learning. It provides a bounded final
        # influence based on the agent's current
        # self-model.
        # -------------------------------------------------

        self.last_self_aware_decision_result = None

        self_awareness_influence = (
            self.last_calibrated_self_aware_influence
        )

        if self_awareness_influence is None:

            self_awareness_influence = getattr(
                self.self_awareness,
                "decision_influence",
                None,
            )

        if self_awareness_influence is not None:

            self.last_self_aware_decision_result = (
                self.self_aware_decision_arbiter.evaluate(
                    decision,
                    self_awareness_influence,
                )
            )

            decision = (
                self.last_self_aware_decision_result
                .decision
            )

        # -------------------------------------------------
        # Execute
        # -------------------------------------------------

        result = self.unified_executor.execute(
            decision=decision,
            agent=self,
            environment=environment,
            actions=list(
                self.actions.values()
            ),
            information_gain=(
                information_gain
            ),
            action_risk=(
                action_risk
            ),
            exploration_risk=risk,
            experience_id=(
                experience_id
                if experience_id is not None
                else (
                    f"{self.name}-exploration-"
                    f"{self.state.cycle + 1}"
                )
            ),
            goal=goal,
            world_state=world_state,
            strategies=strategies,
            action_cost=action_cost,
        )

        # -------------------------------------------------
        # Update goal status after execution
        # -------------------------------------------------

        if environment is not None:

            final_world_state = (
                environment.observe()
            )

        else:

            final_world_state = (
                world_state
            )

        self.observe(
            final_world_state
        )

        goal_satisfied = (
            self.update_goal_status(
                world_state=final_world_state,
                goal=goal,
            )
        )

        # -------------------------------------------------
        # Meta-learning: decision result
        # -------------------------------------------------

        decision_success = (
            goal_satisfied
        )

        if decision.mode == DecisionMode.EXPLORE:

            exploration_result = getattr(
                result,
                "exploration_result",
                None,
            )

            if exploration_result is not None:

                execution_result = getattr(
                    exploration_result,
                    "result",
                    None,
                )

                if isinstance(
                    execution_result,
                    dict,
                ):
                    if "success" in execution_result:
                        decision_success = bool(
                            execution_result[
                                "success"
                            ]
                        )

        # -------------------------------------------------
        # Capability telemetry
        # -------------------------------------------------
        #
        # Record capability evidence only after the final
        # decision outcome has been resolved.
        # -------------------------------------------------

        self._record_decision_capability_telemetry(
            decision=decision,
            success=decision_success,
        )

        # -------------------------------------------------
        # Unified self-awareness update
        # -------------------------------------------------
        self.update_self_awareness_from_all_evidence(
            persist=True,
        )

        self.calibrate_self_awareness()

        # -------------------------------------------------
        # Preserve calibrated influence object identity when
        # the post-decision calibration produces the same
        # influence values that were already used for the
        # current decision.
        #
        # This is important because the calibrated influence
        # restored from persistence may be the exact runtime
        # object consumed by the next arbitration cycle.
        # Replacing it with an equivalent newly-created object
        # would preserve value equality but lose object identity.
        # -------------------------------------------------
        if (
            self_awareness_influence is not None
            and self.last_calibrated_self_aware_influence is not None
            and self.last_calibrated_self_aware_influence
            == self_awareness_influence
        ):
            self.last_calibrated_self_aware_influence = (
                self_awareness_influence
            )

        # -------------------------------------------------
        # Self-awareness feedback observation
        # -------------------------------------------------
        self.record_self_awareness_feedback()

        if decision.mode == DecisionMode.EXPLORE:

            self._record_meta_exploration(
                success=decision_success
            )

        else:

            self._record_meta_exploitation(
                success=decision_success
            )

        return result

    # =====================================================
    # Experience lifecycle
    # =====================================================

    def start_experience(
        self,
        experience_id: str,
        goal: str,
    ) -> Experience:

        experience = (
            self.experience_recorder.start(
                experience_id=experience_id,
                goal=goal,
            )
        )

        self.current_experience = (
            experience
        )

        return experience

    def complete_experience(
        self,
    ) -> None:

        if self.current_experience is None:
            raise RuntimeError(
                "No active experience."
            )

        self.experience_recorder.complete()

        experience = (
            self.current_experience
        )

        # -------------------------------------------------
        # Procedural learning
        # -------------------------------------------------

        self.procedural_learning.learn_from_episode_sequence(
            goal=experience.goal,
            episodes=experience.episodes,
        )

        # -------------------------------------------------
        # Goal progress learning
        # -------------------------------------------------

        if self.current_goal is not None:

            self.last_progress_learning = (
                self.progress_learning.learn_from_experience(
                    goal=self.current_goal,
                    episodes=experience.episodes,
                )
            )

        # -------------------------------------------------
        # Meta-learning: progress
        # -------------------------------------------------

        if self.last_progress_learning is not None:

            self._record_meta_progress(
                progress_delta=(
                    self.last_progress_learning
                    .progress_delta
                ),
                event_id=(
                    f"{self.name}:meta:progress:"
                    f"{experience.experience_id}"
                ),
            )

        # -------------------------------------------------
        # Strategy attribution + learning
        # -------------------------------------------------

        self._update_strategy_from_experience(
            experience=experience,
            success=True,
        )

        # -------------------------------------------------
        # Self learning
        # -------------------------------------------------

        self.self_learning.learn_from_experience(
            experience
        )

        self.self_model_store.save()

        # -------------------------------------------------
        # Experience-driven self-awareness
        # -------------------------------------------------
        # self.update_self_awareness_from_experience(
        #     persist=True,
        # )        
        self.update_self_awareness_from_all_evidence(
            persist=True,
        )

    def fail_experience(
        self,
    ) -> None:

        if self.current_experience is None:
            raise RuntimeError(
                "No active experience."
            )

        self.experience_recorder.fail()

        experience = (
            self.current_experience
        )

        # -------------------------------------------------
        # Progress learning
        # -------------------------------------------------

        if self.current_goal is not None:

            self.last_progress_learning = (
                self.progress_learning.learn_from_experience(
                    goal=self.current_goal,
                    episodes=experience.episodes,
                )
            )

        # -------------------------------------------------
        # Meta-learning: progress
        # -------------------------------------------------

        if self.last_progress_learning is not None:

            self._record_meta_progress(
                progress_delta=(
                    self.last_progress_learning
                    .progress_delta
                ),
                event_id=(
                    f"{self.name}:meta:progress:"
                    f"{experience.experience_id}"
                ),
            )

        # -------------------------------------------------
        # Strategy attribution + learning
        # -------------------------------------------------

        self._update_strategy_from_experience(
            experience=experience,
            success=False,
        )

        # -------------------------------------------------
        # Self learning
        # -------------------------------------------------

        self.self_learning.learn_from_experience(
            experience
        )

        self.self_model_store.save()

        # -------------------------------------------------
        # Experience-driven self-awareness
        # -------------------------------------------------
        # self.update_self_awareness_from_experience(
        #     persist=True,
        # )        
        self.update_self_awareness_from_all_evidence(
            persist=True,
        )

    def close_experience(
        self,
    ) -> None:

        self.experience_recorder.close()

        self.current_experience = (
            None
        )

    # =====================================================
    # Available strategies
    # =====================================================

    def get_available_strategies(
        self,
    ) -> List[Strategy]:
        """
        Return strategies that are currently executable.

        Only ACTIVE strategies are eligible.
        """

        return [
            strategy
            for strategy in self.strategy_registry
            if strategy.is_active
        ]

    # =====================================================
    # Strategy Evolution Evidence
    # =====================================================

    def get_strategy_evolution_evidence(
        self,
        strategy_name: str,
    ) -> Optional[
        StrategyEvolutionEvidence
    ]:

        return self.strategy_evolution_evidence.get(
            strategy_name
        )

    def get_all_strategy_evolution_evidence(
        self,
    ) -> list[StrategyEvolutionEvidence]:

        return self.strategy_evolution_evidence.all()

    def _create_evolution_evidence(
        self,
        strategy: Strategy,
        source_strategy: Strategy,
        experience: Experience,
    ) -> StrategyEvolutionEvidence:

        existing = (
            self.strategy_evolution_evidence.get(
                strategy.name
            )
        )

        if existing is not None:
            return existing

        evidence = (
            StrategyEvolutionEvidence(
                strategy_name=(
                    strategy.name
                ),
                source_strategy_name=(
                    source_strategy.name
                ),
                source_experience_id=(
                    experience.experience_id
                ),
            )
        )

        self.strategy_evolution_evidence.add(
            evidence
        )

        return evidence

    # =====================================================
    # Meta-learning
    # =====================================================

    def _persist_meta_learning_profile(
        self,
    ) -> None:
        """
        Persist the current meta-learning profile when
        persistent meta-learning storage is enabled.

        In-memory agents intentionally do nothing here.
        """

        if (
            self.memory_path is None
            or self.meta_learning_repository is None
        ):
            return

        self.meta_learning_store.save(
            self.name,
            self.meta_learning_profile,
        )

    def evaluate_behavioral_adaptation(
        self,
    ) -> BehavioralAdaptationResult:
        """
        Evaluate the current behavioral adaptation state
        from the agent's accumulated meta-learning profile.

        This method does not execute an action and does not
        modify the meta-learning profile itself.
        """

        return (
            self.behavioral_adaptation.evaluate(
                self.meta_learning_profile
            )
        )

    def reset_behavioral_adaptation(
        self,
    ) -> None:
        """
        Reset the behavioral adaptation state without
        deleting meta-learning history.
        """

        self.behavioral_adaptation.reset()

    def get_behavioral_adaptation_snapshot(
        self,
    ) -> dict:
        """
        Return the current stabilized behavioral state.
        """

        return (
            self.behavioral_adaptation.snapshot()
        )

    def _next_meta_learning_event_id(
        self,
        event_type: str,
    ) -> str:
        """
        Generate a unique event id for meta-learning.
        """

        counter = (
            self.meta_learning_profile
            .total_learning_events
            + 1
        )

        return (
            f"{self.name}:meta:"
            f"{event_type}:{counter}"
        )

    def _record_meta_exploration(
        self,
        success: bool,
    ) -> None:

        self.meta_learning_attributor.record_exploration(
            profile=self.meta_learning_profile,
            success=success,
            event_id=(
                self._next_meta_learning_event_id(
                    "exploration"
                )
            ),
        )

        self._persist_meta_learning_profile()

    def _record_meta_exploitation(
        self,
        success: bool,
    ) -> None:

        self.meta_learning_attributor.record_exploitation(
            profile=self.meta_learning_profile,
            success=success,
            event_id=(
                self._next_meta_learning_event_id(
                    "exploitation"
                )
            ),
        )

        self._persist_meta_learning_profile()

    def _record_meta_evolution(
        self,
        success: bool,
    ) -> None:

        self.meta_learning_attributor.record_evolution(
            profile=self.meta_learning_profile,
            success=success,
            event_id=(
                self._next_meta_learning_event_id(
                    "evolution"
                )
            ),
        )

        self._persist_meta_learning_profile()

    def _record_meta_recovery(
        self,
        success: bool,
    ) -> None:

        self.meta_learning_attributor.record_recovery(
            profile=self.meta_learning_profile,
            success=success,
            event_id=(
                self._next_meta_learning_event_id(
                    "recovery"
                )
            ),
        )

        self._persist_meta_learning_profile()

    def _record_meta_progress(
        self,
        progress_delta: float,
        event_id: str | None = None,
    ) -> None:

        self.meta_learning_attributor.record_progress(
            profile=self.meta_learning_profile,
            progress_delta=progress_delta,
            event_id=(
                event_id
                if event_id is not None
                else self._next_meta_learning_event_id(
                    "progress"
                )
            ),
        )

        self._persist_meta_learning_profile()

    def get_meta_learning_profile(
        self,
    ):
        """
        Return the current meta-learning profile.
        """

        return self.meta_learning_profile

    def get_meta_learning_snapshot(
        self,
    ):
        """
        Return a serializable snapshot of the current
        meta-learning state.
        """

        return self.meta_learning_profile.snapshot()

    def evaluate_end_to_end_behavior(
        self,
        uncertainty: float,
        information_gain: float,
        risk: float,
    ) -> BehavioralDecisionResult:
        """
        Evaluate the agent's behavioral mode by combining
        current decision signals with learned behavioral
        history.

        This method evaluates behavior only.
        It does not execute an action.
        """

        return (
            self.behavioral_decision_evaluator.evaluate(
                profile=self.meta_learning_profile,
                uncertainty=uncertainty,
                information_gain=information_gain,
                risk=risk,
            )
        )
