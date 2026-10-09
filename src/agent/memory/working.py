from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkingMemory:
    """
    Short-term memory used during the current cognitive process.
    """

    goal: Optional[str] = None
    plan: List[str] = field(default_factory=list)
    current_action: Optional[str] = None
    recent_observation: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None

    def set_goal(self, goal: str) -> None:
        self.goal = goal

    def set_plan(self, plan: List[str]) -> None:
        self.plan = list(plan)

    def set_current_action(self, action: Optional[str]) -> None:
        self.current_action = action

    def set_observation(self, observation: Dict[str, Any]) -> None:
        self.recent_observation = observation.copy()

    def update_context(self, context: Dict[str, Any]) -> None:
        self.context.update(context)

    def set_reasoning(self, reasoning: Optional[str]) -> None:
        self.reasoning = reasoning

    def clear_current_action(self) -> None:
        self.current_action = None

    def clear_plan(self) -> None:
        self.plan.clear()

    def clear(self) -> None:
        self.goal = None
        self.plan.clear()
        self.current_action = None
        self.recent_observation = None
        self.context.clear()
        self.reasoning = None

    @property
    def has_active_plan(self) -> bool:
        return len(self.plan) > 0
