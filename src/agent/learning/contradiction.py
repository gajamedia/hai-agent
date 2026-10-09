from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Contradiction:
    """
    Represents evidence that conflicts with an
    existing semantic or causal belief.
    """

    subject: str
    predicate: str
    expected_object: Any
    observed_object: Any

    source_episode_id: Optional[str] = None

    reason: str = ""
