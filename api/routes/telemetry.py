
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(
    prefix="/telemetry",
    tags=["telemetry"],
)


def _to_dict(value: Any) -> dict[str, Any]:
    """Convert a telemetry dataclass or dictionary to JSON-ready data."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    if isinstance(value, dict):
        return dict(value)

    raise TypeError(
        "Telemetry value must be a dataclass or dictionary."
    )


def _supports_telemetry(agent: Any) -> bool:
    return (
        callable(
            getattr(
                agent,
                "get_capability_telemetry_snapshot",
                None,
            )
        )
        and callable(
            getattr(
                agent,
                "get_capability_telemetry_events",
                None,
            )
        )
    )


def _has_telemetry_evidence(agent: Any) -> bool:
    try:
        snapshot = _to_dict(
            agent.get_capability_telemetry_snapshot()
        )
        return bool(snapshot.get("event_count", 0))
    except (TypeError, ValueError, AttributeError):
        return False


def _get_agent(request: Request) -> tuple[Any, str]:
    """Select an available telemetry source with evidence-aware fallback."""
    execution_runtime = getattr(
        request.app.state,
        "execution_runtime",
        None,
    )
    execution_agent = getattr(
        execution_runtime,
        "agent",
        None,
    )

    cognitive_runtime = getattr(
        request.app.state,
        "cognitive_runtime",
        None,
    )
    cognitive_agent = getattr(
        cognitive_runtime,
        "agent",
        None,
    )

    cognitive_available = (
        cognitive_agent is not None
        and _supports_telemetry(cognitive_agent)
    )

    execution_available = (
        execution_agent is not None
        and _supports_telemetry(execution_agent)
    )

    # Prefer actual simulation evidence.
    if (
        execution_available
        and _has_telemetry_evidence(execution_agent)
    ):
        return execution_agent, "execution_runtime"

    # Preserve the existing cognitive telemetry contract.
    if cognitive_available:
        return cognitive_agent, "cognitive_runtime"

    # Without a valid cognitive agent, only expose the simulation
    # agent if it has real telemetry evidence.
    if (
        execution_available
        and _has_telemetry_evidence(execution_agent)
    ):
        return execution_agent, "execution_runtime"

    if cognitive_runtime is None and execution_runtime is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "runtime_unavailable",
                "message": "No runtime is available.",
            },
        )

    raise HTTPException(
        status_code=503,
        detail={
            "error": "agent_unavailable",
            "message": (
                "No available runtime exposes capability telemetry "
                "with the required evidence."
            ),
        },
    )


@router.get("/capabilities")
def get_capability_telemetry(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Read capability telemetry without mutating runtime state."""
    agent, source = _get_agent(request)

    try:
        snapshot_data = _to_dict(
            agent.get_capability_telemetry_snapshot()
        )
        event_data = [
            _to_dict(event)
            for event in agent.get_capability_telemetry_events()
        ]
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "telemetry_read_failed",
                "message": (
                    "Capability telemetry could not be serialized."
                ),
            },
        ) from exc

    latest_events = list(reversed(event_data))[:limit]
    capabilities = snapshot_data.get("capabilities", {})

    return {
        "status": "ok",
        "source": source,
        "snapshot": {
            **snapshot_data,
            "capability_count": len(capabilities),
            "has_evidence": bool(capabilities),
            "observed_event_count": snapshot_data.get(
                "event_count",
                0,
            ),
        },
        "events": latest_events,
        "events_returned": len(latest_events),
        "events_limit": limit,
    }
