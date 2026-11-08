from __future__ import annotations

import json

import pytest

from agent.llm.execution_trace_store import (
    ExecutionTraceIntegrityError,
    ExecutionTraceStore,
)


def make_event(
    event_id: str,
    *,
    execution_id: str = "run-recovery-1",
    event_type: str = "step_recorded",
    payload: dict | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "execution_id": execution_id,
        "event_type": event_type,
        "payload": payload if payload is not None else {"action": "open_box"},
    }


def test_idempotent_append_survives_store_recreation(tmp_path):
    path = tmp_path / "trace.jsonl"
    first_store = ExecutionTraceStore(path, fsync=False)
    first = first_store.append(make_event("event-1", event_type="started"))

    # A new instance simulates application restart.
    restarted_store = ExecutionTraceStore(path, fsync=False)
    duplicate = restarted_store.append(make_event("event-1", event_type="started"))

    assert duplicate == first
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert restarted_store.verify()["record_count"] == 1


def test_conflicting_duplicate_does_not_mutate_existing_evidence(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)
    store.append(make_event("event-1", payload={"action": "open_box"}))
    before = path.read_bytes()

    with pytest.raises(ExecutionTraceIntegrityError, match="different content"):
        store.append(make_event("event-1", payload={"action": "unlock_door"}))

    assert path.read_bytes() == before
    assert store.verify()["record_count"] == 1


def test_append_fails_closed_and_preserves_bytes_after_truncated_tail(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)
    store.append(make_event("event-1", event_type="started"))

    # Simulate a torn write: the last JSONL record has no terminating newline.
    with path.open("ab") as stream:
        stream.write(b'{"partial":')
    before = path.read_bytes()

    with pytest.raises(ExecutionTraceIntegrityError, match="incomplete"):
        store.append(make_event("event-2", event_type="completed"))

    assert path.read_bytes() == before


def test_reordered_records_are_detected_without_automatic_repair(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)
    store.append(make_event("event-1", event_type="started"))
    store.append(make_event("event-2", event_type="completed"))
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(lines[1] + lines[0], encoding="utf-8", newline="")
    corrupted_bytes = path.read_bytes()

    with pytest.raises(ExecutionTraceIntegrityError, match="sequence"):
        store.verify()

    # Verification is read-only: it must not silently reorder or rewrite data.
    assert path.read_bytes() == corrupted_bytes


def test_replay_is_read_only_and_does_not_execute_or_persist_mutations(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)
    store.append(
        make_event(
            "event-1",
            event_type="action_completed",
            payload={"action": "open_box", "success": True},
        )
    )
    before = path.read_bytes()

    replayed = store.replay("run-recovery-1")
    assert len(replayed) == 1
    assert not callable(replayed[0])

    # Mutating the returned replay object must not alter persisted evidence.
    replayed[0]["payload"]["success"] = False
    assert path.read_bytes() == before
    assert store.read_all()[0]["payload"]["success"] is True


def test_invalid_hash_chain_blocks_append_and_preserves_file(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)
    store.append(make_event("event-1", event_type="started"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["record_hash"] = "0" * 64
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ExecutionTraceIntegrityError, match="record hash"):
        store.append(make_event("event-2", event_type="completed"))

    assert path.read_bytes() == before
