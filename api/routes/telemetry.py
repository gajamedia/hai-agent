from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/telemetry", tags=["telemetry"])
TelemetrySource = Literal["cognitive_runtime", "execution_runtime"]


def _to_dict(value: Any) -> dict[str, Any]:
    """Convert a telemetry dataclass or dictionary to JSON-ready data."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("Telemetry value must be a dataclass or dictionary.")


def _supports_telemetry(agent: Any) -> bool:
    return (
        callable(getattr(agent, "get_capability_telemetry_snapshot", None))
        and callable(getattr(agent, "get_capability_telemetry_events", None))
    )


def _has_telemetry_evidence(agent: Any) -> bool:
    try:
        snapshot = _to_dict(agent.get_capability_telemetry_snapshot())
        return bool(snapshot.get("event_count", 0))
    except (TypeError, ValueError, AttributeError):
        return False


def _runtime_agents(request: Request) -> dict[str, Any]:
    """Return the two independent telemetry agents without merging them."""
    execution_runtime = getattr(request.app.state, "execution_runtime", None)
    cognitive_runtime = getattr(request.app.state, "cognitive_runtime", None)
    return {
        "cognitive_runtime": getattr(cognitive_runtime, "agent", None),
        "execution_runtime": getattr(execution_runtime, "agent", None),
    }


def _get_agent(
    request: Request,
    source: TelemetrySource | None = None,
) -> tuple[Any, str]:
    """Resolve one telemetry source; preserve legacy evidence-aware selection."""
    agents = _runtime_agents(request)

    if source is not None:
        agent = agents[source]
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "runtime_unavailable",
                    "source": source,
                    "message": f"The {source} is not available.",
                },
            )
        if not _supports_telemetry(agent):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "telemetry_unavailable",
                    "source": source,
                    "message": f"The {source} does not expose capability telemetry.",
                },
            )
        return agent, source

    # Backward-compatible behavior for existing callers without `source`.
    execution_agent = agents["execution_runtime"]
    cognitive_agent = agents["cognitive_runtime"]
    if (
        execution_agent is not None
        and _supports_telemetry(execution_agent)
        and _has_telemetry_evidence(execution_agent)
    ):
        return execution_agent, "execution_runtime"
    if cognitive_agent is not None and _supports_telemetry(cognitive_agent):
        return cognitive_agent, "cognitive_runtime"
    if (
        execution_agent is not None
        and _supports_telemetry(execution_agent)
        and _has_telemetry_evidence(execution_agent)
    ):
        return execution_agent, "execution_runtime"

    if all(getattr(request.app.state, key, None) is None for key in (
        "cognitive_runtime", "execution_runtime"
    )):
        raise HTTPException(
            status_code=503,
            detail={"error": "runtime_unavailable", "message": "No runtime is available."},
        )
    raise HTTPException(
        status_code=503,
        detail={
            "error": "agent_unavailable",
            "message": "No available runtime exposes capability telemetry with the required evidence.",
        },
    )


def _read_agent_telemetry(agent: Any, source: str, limit: int) -> dict[str, Any]:
    try:
        snapshot_data = _to_dict(agent.get_capability_telemetry_snapshot())
        event_data = [_to_dict(event) for event in agent.get_capability_telemetry_events()]
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "telemetry_read_failed",
                "source": source,
                "message": "Capability telemetry could not be serialized.",
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
            "observed_event_count": snapshot_data.get("event_count", 0),
        },
        "events": latest_events,
        "events_returned": len(latest_events),
        "events_limit": limit,
    }


@router.get("/capabilities")
def get_capability_telemetry(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    source: TelemetrySource | None = Query(
        default=None,
        description=(
            "Optional explicit source: cognitive_runtime or execution_runtime. "
            "Omit to preserve legacy evidence-aware source selection."
        ),
    ),
) -> dict[str, Any]:
    """Read one source's telemetry without mutating runtime state."""
    agent, resolved_source = _get_agent(request, source)
    return _read_agent_telemetry(agent, resolved_source, limit)


@router.get("/capabilities/sources")
def get_capability_telemetry_sources(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Return independently labeled telemetry for both runtimes.

    A missing/unavailable runtime is reported in its own entry and does not
    prevent the other runtime's telemetry from being returned.
    """
    results: dict[str, Any] = {"status": "ok", "sources": {}}
    for source, agent in _runtime_agents(request).items():
        if agent is None:
            results["sources"][source] = {
                "status": "unavailable",
                "source": source,
                "error": "runtime_unavailable",
            }
            continue
        if not _supports_telemetry(agent):
            results["sources"][source] = {
                "status": "unavailable",
                "source": source,
                "error": "telemetry_unavailable",
            }
            continue
        try:
            results["sources"][source] = _read_agent_telemetry(agent, source, limit)
        except HTTPException as exc:
            results["sources"][source] = {
                "status": "error",
                "source": source,
                "error": (exc.detail.get("error") if isinstance(exc.detail, dict) else "telemetry_read_failed"),
                "message": (exc.detail.get("message") if isinstance(exc.detail, dict) else str(exc.detail)),
            }
    return results
