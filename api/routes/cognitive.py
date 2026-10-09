
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from agent.learning.metacognition import MetacognitiveReflection
from agent.llm.agent_runtime import AgentCognitiveRuntime
from agent.llm.orchestrator import LLMOrchestrationStageError

from api.dependencies import get_runtime
from api.schemas import CognitiveRunRequest, CognitiveRunResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/cognitive",
    tags=["cognitive"],
)


@router.post(
    "/run",
    response_model=CognitiveRunResponse,
)
def run_cognitive_cycle(
    payload: CognitiveRunRequest,
    runtime: AgentCognitiveRuntime = Depends(get_runtime),
) -> CognitiveRunResponse:
    """Run a cognitive pass without executing the generated plan."""

    if payload.include_metacognition:
        if payload.episode is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "episode_required",
                    "message": (
                        "episode is required when "
                        "include_metacognition is true."
                    ),
                },
            )

        if payload.deterministic_reflection is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "deterministic_reflection_required",
                    "message": (
                        "deterministic_reflection is required when "
                        "include_metacognition is true."
                    ),
                },
            )

    deterministic_reflection = None

    if payload.deterministic_reflection is not None:
        deterministic_reflection = MetacognitiveReflection(
            **payload.deterministic_reflection.model_dump()
        )

    try:
        result = runtime.run(
            payload.observation,
            goal=payload.goal,
            state=payload.state,
            world_model=payload.world_model,
            relevant_memory=payload.relevant_memory,
            strategy=payload.strategy,
            self_awareness=payload.self_awareness,
            calibration=payload.calibration,
            metacognition=payload.metacognition,
            episode=payload.episode,
            deterministic_reflection=deterministic_reflection,
            include_metacognition=payload.include_metacognition,
            metadata=payload.metadata,
        )

    except LLMOrchestrationStageError:
        # Preserve the existing global handler for stage failures.
        raise

    except (TypeError, ValueError) as exc:
        logger.warning(
            "Invalid cognitive runtime input: %s",
            exc,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_cognitive_input",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected cognitive runtime failure"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "cognitive_runtime_failed",
                "message": (
                    "The cognitive runtime failed. "
                    "Check server logs."
                ),
            },
        ) from exc

    trace = runtime.last_trace

    return CognitiveRunResponse(
        cycle=trace.cycle if trace is not None else None,
        result=result.to_dict(),
        trace=(
            trace.to_dict()
            if payload.include_trace and trace is not None
            else None
        ),
    )
