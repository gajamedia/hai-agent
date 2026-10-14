
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from api.dependencies import get_runtime


router = APIRouter(
    prefix="/strategies",
    tags=["strategies"],
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return (
        result
        if result == result and abs(result) != float("inf")
        else default
    )


def _strategy_to_dict(
    strategy: Any,
    active_strategy: Any,
) -> dict[str, Any]:
    attempts = int(getattr(strategy, "attempts", 0) or 0)
    success_count = int(
        getattr(strategy, "success_count", 0) or 0
    )
    failure_count = int(
        getattr(strategy, "failure_count", 0) or 0
    )

    success_rate = getattr(strategy, "success_rate", None)
    if success_rate is None:
        success_rate = (
            success_count / attempts
            if attempts
            else 0.0
        )

    average_actions = getattr(
        strategy,
        "average_actions",
        None,
    )
    if average_actions is None:
        average_actions = (
            _number(getattr(strategy, "total_actions", 0))
            / attempts
            if attempts
            else 0.0
        )

    return {
        "name": str(getattr(strategy, "name", "")),
        "goal": str(getattr(strategy, "goal", "")),
        "steps": list(getattr(strategy, "steps", []) or []),
        "status": str(getattr(strategy, "status", "active")),
        "is_active": bool(getattr(strategy, "is_active", False)),
        "is_currently_selected": (
            active_strategy is not None
            and getattr(active_strategy, "name", None)
            == getattr(strategy, "name", None)
        ),
        "success_count": success_count,
        "failure_count": failure_count,
        "attempts": attempts,
        "success_rate": _number(success_rate),
        "total_actions": int(
            getattr(strategy, "total_actions", 0) or 0
        ),
        "average_actions": _number(average_actions),
        "reflection_penalty": _number(
            getattr(strategy, "reflection_penalty", 0.0)
        ),
        "total_progress": _number(
            getattr(strategy, "total_progress", 0.0)
        ),
        "progress_count": int(
            getattr(strategy, "progress_count", 0) or 0
        ),
        "average_progress": _number(
            getattr(strategy, "average_progress", 0.0)
        ),
        "positive_progress_count": int(
            getattr(strategy, "positive_progress_count", 0) or 0
        ),
        "negative_progress_count": int(
            getattr(strategy, "negative_progress_count", 0) or 0
        ),
        "source_experiences": list(
            getattr(strategy, "source_experiences", []) or []
        ),
    }


def _evidence_to_dict(evidence: Any) -> dict[str, Any]:
    trial_count = int(
        getattr(evidence, "trial_count", 0) or 0
    )
    success_count = int(
        getattr(evidence, "success_count", 0) or 0
    )
    failure_count = int(
        getattr(evidence, "failure_count", 0) or 0
    )

    success_rate = getattr(evidence, "success_rate", None)
    if success_rate is None:
        success_rate = (
            success_count / trial_count
            if trial_count
            else 0.0
        )

    return {
        "strategy_name": str(
            getattr(evidence, "strategy_name", "")
        ),
        "source_strategy_name": str(
            getattr(evidence, "source_strategy_name", "")
        ),
        "source_experience_id": str(
            getattr(evidence, "source_experience_id", "")
        ),
        "generation": int(
            getattr(evidence, "generation", 1) or 1
        ),
        "status": str(
            getattr(evidence, "status", "experimental")
        ),
        "trial_count": trial_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": _number(success_rate),
        "failure_rate": _number(
            getattr(
                evidence,
                "failure_rate",
                (
                    failure_count / trial_count
                    if trial_count
                    else 0.0
                ),
            )
        ),
        "total_progress": _number(
            getattr(evidence, "total_progress", 0.0)
        ),
        "average_progress": _number(
            getattr(evidence, "average_progress", 0.0)
        ),
        "source_experience_ids": list(
            getattr(evidence, "source_experience_ids", []) or []
        ),
    }


def _serialize_agent_strategies(
    agent: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Serialize strategies and evidence from one Agent."""
    strategy_store = getattr(
        agent,
        "strategy_store",
        None,
    )

    if strategy_store is None or not callable(
        getattr(strategy_store, "all", None)
    ):
        return [], []

    strategies = strategy_store.all()
    active_strategy = getattr(
        agent,
        "active_strategy",
        None,
    )

    serialized_strategies = [
        _strategy_to_dict(item, active_strategy)
        for item in strategies
    ]

    evidence_getter = getattr(
        agent,
        "get_all_strategy_evolution_evidence",
        None,
    )
    evidence_items = (
        evidence_getter()
        if callable(evidence_getter)
        else []
    )

    serialized_evidence = [
        _evidence_to_dict(item)
        for item in evidence_items
    ]

    return serialized_strategies, serialized_evidence


def _evidence_persistence(agent: Any) -> str:
    evidence_store = getattr(
        agent,
        "strategy_evolution_evidence",
        None,
    )
    repository = getattr(
        evidence_store,
        "repository",
        None,
    )

    return (
        "persistent_sqlite"
        if getattr(repository, "database_path", None) is not None
        else "in_memory"
    )


@router.get("")
def list_strategies(
    request: Request,
    runtime=Depends(get_runtime),
) -> dict[str, Any]:
    """
    Return cognitive strategies and a separate simulation snapshot.

    The original root-level fields remain cognitive-runtime data
    for backward compatibility.
    """
    agent = getattr(runtime, "agent", None)

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "agent_unavailable",
                "message": (
                    "The cognitive runtime has no accessible agent."
                ),
            },
        )

    strategy_store = getattr(
        agent,
        "strategy_store",
        None,
    )

    if strategy_store is None or not callable(
        getattr(strategy_store, "all", None)
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "strategy_registry_unavailable",
                "message": (
                    "The agent does not expose a strategy registry."
                ),
            },
        )

    try:
        cognitive_strategies, cognitive_evidence = (
            _serialize_agent_strategies(agent)
        )

        execution_runtime = getattr(
            request.app.state,
            "execution_runtime",
            None,
        )
        simulation_agent = getattr(
            execution_runtime,
            "agent",
            None,
        )

        if simulation_agent is not None:
            (
                simulation_strategies,
                simulation_evidence,
            ) = _serialize_agent_strategies(
                simulation_agent
            )
        else:
            simulation_strategies = []
            simulation_evidence = []

    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "strategy_data_invalid",
                "message": (
                    "Strategy data could not be serialized."
                ),
            },
        ) from exc

    active_strategy = getattr(
        agent,
        "active_strategy",
        None,
    )

    return {
        "status": "ok",

        # Backward-compatible cognitive-runtime fields.
        "source": "cognitive_runtime",
        "agent_name": getattr(agent, "name", None),
        "strategy_count": len(cognitive_strategies),
        "active_strategy_name": getattr(
            active_strategy,
            "name",
            None,
        ),
        "strategies": cognitive_strategies,
        "evolution_evidence_count": len(cognitive_evidence),
        "evolution_evidence": cognitive_evidence,
        "evolution_evidence_persistence": (
            _evidence_persistence(agent)
        ),

        # New, explicitly separate simulation-runtime fields.
        "simulation": {
            "source": "simulation_runtime",
            "agent_name": getattr(
                simulation_agent,
                "name",
                None,
            ),
            "strategy_count": len(simulation_strategies),
            "strategies": simulation_strategies,
            "evolution_evidence_count": len(
                simulation_evidence
            ),
            "evolution_evidence": simulation_evidence,
            "strategy_persistence": (
                "persistent_sqlite"
                if getattr(
                    getattr(simulation_agent, "strategy_store", None),
                    "repository",
                    None,
                ) is not None
                else "in_memory"
            ),
            "evolution_evidence_persistence": (
                _evidence_persistence(simulation_agent)
                if simulation_agent is not None
                else "unavailable"
            ),
        },
    }
