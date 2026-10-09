
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DeterministicReflectionInput(BaseModel):
    """Validated deterministic reflection supplied to metacognition."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str | None = None
    goal: str | None = None
    action: str | None = None
    success: bool | None = None

    progress_before: float | None = None
    progress_after: float | None = None
    progress_delta: float | None = None

    diagnosis: str
    lesson: str
    next_step: str

    evidence_quality: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    change_required: bool


class CognitiveRunRequest(BaseModel):
    """Input for one side-effect-free cognitive pass."""

    model_config = ConfigDict(extra="forbid")

    observation: Any
    goal: Any = None
    state: Any = None
    world_model: Any = None
    relevant_memory: list[Any] | None = None
    strategy: Any = None
    self_awareness: Any = None
    calibration: Any = None
    metacognition: Any = None

    # Optional inputs for LLM-assisted metacognition.
    include_metacognition: bool = False
    episode: Any = None
    deterministic_reflection: DeterministicReflectionInput | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    include_trace: bool = False


class CognitiveRunResponse(BaseModel):
    cycle: int | None = None
    result: dict[str, Any]
    trace: dict[str, Any] | None = None
