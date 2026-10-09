from .strategy import Strategy
from .strategy_learning import StrategyLearning
from .strategy_selector import StrategySelector
from .contradiction import Contradiction
from .contradiction_detector import (
    ContradictionDetector,
)
from .contradiction_learning import (
    ContradictionLearning,
)
from .context_extractor import (
    ContextExtractor,
)


def __getattr__(name: str):
    """Load StrategyStore lazily to avoid circular imports."""
    if name == "StrategyStore":
        from .strategy_store import StrategyStore
        return StrategyStore

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "Strategy",
    "StrategyStore",
    "StrategyLearning",
    "StrategySelector",
    "Contradiction",
    "ContradictionDetector",
    "ContradictionLearning",
    "ContextExtractor",
]
