from dataclasses import dataclass
from typing import Any, Dict, Optional

from .models import Memory


@dataclass
class EpisodicMemory(Memory):
    """
    Represents a concrete experience that happened to the agent.
    """

    goal: Optional[str] = None
    action: Optional[str] = None

    observation_before: Optional[Dict[str, Any]] = None
    observation_after: Optional[Dict[str, Any]] = None

    result: Optional[Dict[str, Any]] = None
    cycle: Optional[int] = None
