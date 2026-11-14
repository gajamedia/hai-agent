"""Server settings for changing data, read from the environment each time they are asked for.

Read live (not once at startup) on purpose: an administrator who tightens a rule, or turns
self-approval off, sees it take effect without a restart.
"""

from __future__ import annotations

import os
from typing import Any

from agent.workflow.data_change import DEFAULT_MAX_ROWS_CEILING

PLAN, EXECUTE, APPROVE = "data:change:plan", "data:change:execute", "data:change:approve"
ENTRY = "data:entry:record"


def _int_env(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


def settings() -> dict[str, Any]:
    mode = os.getenv("HAI_DATA_CHANGE_APPROVAL", "delete").strip().lower()
    if mode not in {"delete", "all", "none"}:
        mode = "delete"
    return {
        "max_rows_ceiling": _int_env("HAI_DATA_CHANGE_MAX_ROWS", DEFAULT_MAX_ROWS_CEILING, 1, 1_000_000),
        "default_max_rows": 100,
        "approval": mode,
        # Off by default. On a one-person setup the owner may approve their own plan; the approval is still a
        # separate step that needs a fresh sign-in, and both decisions are in the audit trail.
        "self_approval": os.getenv("HAI_DATA_CHANGE_SELF_APPROVAL", "").strip().lower() in {"1", "true", "yes"},
        # "minimal" (default): audit files hold no business data, only row keys and counts.
        # "full": they also hold the data as it was before a change and the values written.
        "audit_detail": "full" if os.getenv("HAI_AUDIT_DETAIL", "").strip().lower() == "full" else "minimal",
        # How long after recording its author may take an entry back (minutes).
        "entry_undo_minutes": _int_env("HAI_ENTRY_UNDO_MINUTES", 1440, 1, 43200),
        "step_up_max_age_seconds": _int_env("HAI_STEPUP_MAX_AGE_SECONDS", 120, 30, 900),
    }


def requires_approval(operation: str) -> bool:
    mode = settings()["approval"]
    return mode == "all" or (mode == "delete" and operation == "delete")


__all__ = ["APPROVE", "ENTRY", "EXECUTE", "PLAN", "requires_approval", "settings"]
