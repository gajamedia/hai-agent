from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    memory: Any
    score: float
