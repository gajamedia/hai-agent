"""
Meta-learning runtime integration.

Connects the meta-learning feedback pipeline to runtime
decision making without replacing the existing Agent
execution loop.

Pipeline:

    MetaLearningProfile
        -> FeedbackEngine
        -> FeedbackPolicy
        -> MetaLearningAwareDecisionMaker
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agent.learning.meta_learning import (
    MetaLearningProfile,
)
from agent.learning.meta_learning_feedback import (
    MetaLearningFeedbackEngine,
)
from agent.learning.meta_learning_feedback_policy import (
    FeedbackDecision,
    MetaLearningFeedbackPolicy,
)
from agent.learning.meta_learning_aware_decision import (
    MetaLearningAwareDecision,
    MetaLearningAwareDecisionMaker,
)


@dataclass(frozen=True)
class RuntimeDecisionContext:
    """
    Runtime signals supplied to the meta-learning layer.
    """

    uncertainty: float
    information_gain: float
    risk: float

    explore_allowed: bool = True
    exploit_allowed: bool = True


@dataclass(frozen=True)
class MetaLearningRuntimeResult:
    """
    Complete result of one runtime meta-learning pass.
    """

    feedback: Any
    feedback_decision: FeedbackDecision
    decision: MetaLearningAwareDecision

    @property
    def mode(self) -> str:
        return self.decision.mode

    @property
    def adaptation_requested(self) -> bool:
        return self.decision.adaptation_requested

    @property
    def feedback_applied(self) -> bool:
        return self.decision.feedback_applied


class MetaLearningRuntime:
    """
    Runtime bridge for meta-learning.

    Responsibilities:

    1. Read the current meta-learning profile.
    2. Generate feedback.
    3. Apply the feedback policy.
    4. Ask MetaLearningAwareDecisionMaker for the
       adaptive runtime decision.

    It does not execute actions, modify goals, or bypass
    hard runtime constraints.
    """

    def __init__(
        self,
        profile: MetaLearningProfile,
        base_decision_maker,
        feedback_engine: Optional[
            MetaLearningFeedbackEngine
        ] = None,
        feedback_policy: Optional[
            MetaLearningFeedbackPolicy
        ] = None,
        decision_maker: Optional[
            MetaLearningAwareDecisionMaker
        ] = None,
    ) -> None:

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        if base_decision_maker is None:
            raise ValueError(
                "base_decision_maker cannot be None."
            )

        self.profile = profile

        self.feedback_engine = (
            feedback_engine
            or MetaLearningFeedbackEngine()
        )

        self.feedback_policy = (
            feedback_policy
            or MetaLearningFeedbackPolicy()
        )

        self.decision_maker = (
            decision_maker
            or MetaLearningAwareDecisionMaker(
                base_decision_maker=base_decision_maker,
            )
        )

    def evaluate(
        self,
        context: RuntimeDecisionContext,
    ) -> MetaLearningRuntimeResult:

        if context is None:
            raise ValueError(
                "context cannot be None."
            )

        feedback = (
            self.feedback_engine.analyze(
                self.profile
            )
        )

        feedback_decision = (
            self.feedback_policy.evaluate(
                feedback
            )
        )

        decision = (
            self.decision_maker.decide(
                uncertainty=context.uncertainty,
                information_gain=context.information_gain,
                risk=context.risk,
                feedback_decision=feedback_decision,
                explore_allowed=context.explore_allowed,
                exploit_allowed=context.exploit_allowed,
            )
        )

        return MetaLearningRuntimeResult(
            feedback=feedback,
            feedback_decision=feedback_decision,
            decision=decision,
        )

    def evaluate_from_values(
        self,
        uncertainty: float,
        information_gain: float,
        risk: float,
        explore_allowed: bool = True,
        exploit_allowed: bool = True,
    ) -> MetaLearningRuntimeResult:

        context = RuntimeDecisionContext(
            uncertainty=uncertainty,
            information_gain=information_gain,
            risk=risk,
            explore_allowed=explore_allowed,
            exploit_allowed=exploit_allowed,
        )

        return self.evaluate(
            context
        )
