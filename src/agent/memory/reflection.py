from dataclasses import dataclass, field
from typing import Any, List, Optional

from .models import Memory


@dataclass
class ReflectionMemory(Memory):
    """
    Represents a reflection produced from an experience,
    especially a failed action or strategy.
    """

    goal: Optional[str] = None
    action: Optional[str] = None

    problem: Optional[str] = None
    cause: Optional[str] = None
    lesson: Optional[str] = None

    condition: Any = None

    source_episodes: List[str] = field(
        default_factory=list
    )

    applied: bool = False
