from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


def _to_dict(value: Any) -> dict[str, Any]:
    """Convert a telemetry dataclass or dictionary to JSON-ready data."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("Telemetry value must be a dataclass or dictionary.")


def _get_agent(request: Request) -> Any:
    runtime = getattr(request.app.state, "cognitive_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "runtime_unavailable",
                "message": "The cognitive runtime is not available.",
            },
        )

    agent = getattr(runtime, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "agent_unavailable",
                "message": "The runtime does not expose its Agent instance.",
            },
        )
    return agent


@router.get("/capabilities")
def get_capability_telemetry(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Read live Agent capability telemetry without mutating runtime state."""
    agent = _get_agent(request)

    snapshot_getter = getattr(agent, "get_capability_telemetry_snapshot", None)
    events_getter = getattr(agent, "get_capability_telemetry_events", None)
    if not callable(snapshot_getter) or not callable(events_getter):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "telemetry_unavailable",
                "message": (
                    "The Agent does not expose the required capability "
                    "telemetry read methods."
                ),
            },
        )

    try:
        snapshot_data = _to_dict(snapshot_getter())
        event_data = [_to_dict(event) for event in events_getter()]
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "telemetry_read_failed",
                "message": "Capability telemetry could not be serialized.",
            },
        ) from exc

    latest_events = list(reversed(event_data))[:limit]
    capabilities = snapshot_data.get("capabilities", {})

    return {
        "status": "ok",
        "snapshot": {
            **snapshot_data,
            "capability_count": len(capabilities),
            "has_evidence": bool(capabilities),
            "observed_event_count": snapshot_data.get("event_count", 0),
        },
        "events": latest_events,
        "events_returned": len(latest_events),
        "events_limit": limit,
    }
