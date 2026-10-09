from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .goal import Goal


@dataclass
class AgentState:
    current_goal: Optional[Goal] = None

    current_context: Dict[str, Any] = field(default_factory=dict)

    beliefs: Dict[str, Any] = field(default_factory=dict)

    working_memory: List[Any] = field(default_factory=list)

    previous_action: Optional[str] = None

    last_observation: Optional[Any] = None

    uncertainty: float = 0.0

    cycle: int = 0
