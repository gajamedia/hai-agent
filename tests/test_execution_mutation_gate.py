from __future__ import annotations

import hashlib
import json

import pytest

from api.plan_run_idempotency import (
    PlanRunExecutionBlocked,
    PlanRunIdempotencyStore,
)


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_unknown_plan_run_blocks_other_mutations_and_new_keys(tmp_path):
    store = PlanRunIdempotencyStore(tmp_path / "gate.sqlite3")
    body = {"goal": "open"}
    store.claim(idempotency_key="unknown-key", request_fingerprint=_fingerprint(body))
    store.mark_unknown(idempotency_key="unknown-key", request_fingerprint=_fingerprint(body))

    with pytest.raises(PlanRunExecutionBlocked):
        store.acquire_mutation("reset:request-2")
    with pytest.raises(PlanRunExecutionBlocked):
        store.claim(idempotency_key="new-key", request_fingerprint=_fingerprint({"goal": "unlock"}))

    record = store.get_record("unknown-key")
    assert record is not None and record.status == "unknown"


def test_explicit_reconciliation_clears_global_gate_but_not_old_key(tmp_path):
    store = PlanRunIdempotencyStore(tmp_path / "gate-reconcile.sqlite3")
    body = {"goal": "open"}
    fingerprint = _fingerprint(body)
    store.claim(idempotency_key="uncertain-key", request_fingerprint=fingerprint)
    store.mark_unknown(idempotency_key="uncertain-key", request_fingerprint=fingerprint)

    result = store.reconcile(notes="Reviewed observed simulation state and confirmed no process is executing.")
    assert result["cleared"] is True
    assert result["reconciled_keys"] == ["uncertain-key"]

    # The old uncertain key is never replayable, even after the global gate clears.
    old_key_claim = store.claim(idempotency_key="uncertain-key", request_fingerprint=fingerprint)
    assert old_key_claim.status == "unknown"

    fresh = store.claim(idempotency_key="fresh-key", request_fingerprint=_fingerprint({"goal": "unlock"}))
    assert fresh.status == "running"


def test_running_non_idempotent_mutation_survives_store_restart(tmp_path):
    database = tmp_path / "gate-restart.sqlite3"
    first = PlanRunIdempotencyStore(database)
    first.acquire_mutation("reset:request-1")

    second = PlanRunIdempotencyStore(database)
    with pytest.raises(PlanRunExecutionBlocked):
        second.acquire_mutation("run:request-2")

    second.reconcile(notes="Operator confirmed process is stopped and inspected the current state.")
    second.acquire_mutation("run:request-3")
    second.release_mutation("run:request-3")
