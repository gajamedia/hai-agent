
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from api.schemas import ExecutionRunRequest


router = APIRouter(
    prefix="/execution",
    tags=["execution"],
)


def _get_runtime(request: Request) -> Any:
    runtime = getattr(
        request.app.state,
        "execution_runtime",
        None,
    )

    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "execution_runtime_unavailable",
                "message": (
                    "The simulation execution runtime "
                    "is not available."
                ),
            },
        )

    return runtime


@router.get("/state")
def get_execution_state(
    request: Request,
) -> dict[str, Any]:
    """Read the current simulation state without executing actions."""

    runtime = _get_runtime(request)

    return {
        "status": "ok",
        "state": runtime.get_state(),
        "allowed_actions": list(runtime.allowed_actions),
    }


@router.post("/run")
def run_execution(
    payload: ExecutionRunRequest,
    request: Request,
) -> dict[str, Any]:
    """Execute one allowlisted action or an explicit ordered plan."""

    runtime = _get_runtime(request)

    try:
        return runtime.run(
            actions=payload.actions,
            goal_description=payload.goal_description,
            target_state=payload.target_state,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_execution_request",
                "message": str(exc),
            },
        ) from exc

    except (AttributeError, RuntimeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "execution_failed",
                "message": (
                    "The simulation execution could not "
                    "be completed."
                ),
            },
        ) from exc


@router.post("/reset")
def reset_execution(
    request: Request,
) -> dict[str, Any]:
    """
    Reset the simulation state.

    Historical telemetry is intentionally retained.
    """

    runtime = _get_runtime(request)

    return {
        "status": "ok",
        "message": (
            "Simulation state reset. "
            "Telemetry history was retained."
        ),
        "state": runtime.reset(),
    }
