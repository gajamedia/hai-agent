from __future__ import annotations

import json

import pytest

from agent.llm.execution_trace_store import (
    ExecutionTraceInputError,
    ExecutionTraceIntegrityError,
    ExecutionTraceStore,
)


def event(event_id: str, event_type: str = "step_recorded", *, execution_id: str = "run-1", payload=None):
    return {
        "event_id": event_id,
        "execution_id": execution_id,
        "event_type": event_type,
        "payload": payload or {"step_index": 0, "action": "open_box", "success": True},
    }


def test_persists_events_and_verifies_hash_chain(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    first = store.append(event("evt-1", "started"))
    second = store.append(event("evt-2", "completed", payload={"goal_achieved": True}))

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["previous_hash"] == first["record_hash"]
    assert store.verify()["valid"] is True
    assert store.verify()["record_count"] == 2
    assert [row["event_id"] for row in store.read_all()] == ["evt-1", "evt-2"]


def test_identical_duplicate_append_is_idempotent_but_conflict_is_rejected(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    original = event("evt-1", payload={"action": "open_box"})
    first = store.append(original)
    duplicate = store.append(original)

    assert duplicate == first
    assert len(store.read_all()) == 1
    with pytest.raises(ExecutionTraceIntegrityError, match="different content"):
        store.append(event("evt-1", payload={"action": "close_box"}))


def test_replay_filters_by_execution_and_never_invokes_actions(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    store.append(event("evt-1", "action_started", execution_id="run-1", payload={"action": "dangerous_action"}))
    store.append(event("evt-2", "action_completed", execution_id="run-2", payload={"action": "other_action"}))

    replayed = store.replay("run-1")
    assert len(replayed) == 1
    assert replayed[0]["payload"]["action"] == "dangerous_action"
    # Replay is data-only; the API returns records, not executable callbacks.
    assert not callable(replayed[0])
    assert len(store.replay()) == 2


def test_tampered_payload_is_detected(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)
    store.append(event("evt-1"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["payload"]["success"] = False
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ExecutionTraceIntegrityError, match="record hash"):
        store.read_all()


def test_truncation_and_sequence_discontinuity_fail_closed(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)
    store.append(event("evt-1"))
    store.append(event("evt-2"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["sequence"] = 4
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ExecutionTraceIntegrityError, match="sequence"):
        store.read_all()

    path.write_text('{"partial":', encoding="utf-8")
    with pytest.raises(ExecutionTraceIntegrityError, match="incomplete"):
        store.read_all()


def test_rejects_missing_ids_and_non_json_payloads(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    with pytest.raises(ExecutionTraceInputError, match="event_id"):
        store.append({"execution_id": "run-1", "event_type": "started"})
    with pytest.raises(ExecutionTraceInputError, match="JSON-serializable"):
        store.append(event("evt-1", payload={"bad": object()}))
    assert store.read_all() == []
