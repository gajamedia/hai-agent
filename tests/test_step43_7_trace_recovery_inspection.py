
from __future__ import annotations

import json

from agent.llm.execution_trace_recovery import inspect_trace_file
from agent.llm.execution_trace_store import ExecutionTraceStore


def make_event(
    event_id: str,
    event_type: str = "step_recorded",
) -> dict:
    return {
        "event_id": event_id,
        "execution_id": "recovery-test",
        "event_type": event_type,
        "payload": {
            "action": "open_box",
            "success": True,
        },
    }


def test_valid_trace_reports_fingerprint_and_last_record(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)

    store.append(make_event("evt-1", "started"))
    last = store.append(make_event("evt-2", "completed"))
    original = path.read_bytes()

    report = inspect_trace_file(path)

    assert report["status"] == "valid"
    assert report["valid"] is True
    assert report["record_count"] == 2
    assert report["checked_lines"] == 2
    assert report["first_invalid_line"] is None
    assert report["last_valid_sequence"] == last["sequence"]
    assert report["last_valid_hash"] == last["record_hash"]
    assert len(report["file_sha256"]) == 64
    assert path.read_bytes() == original


def test_truncated_tail_is_reported_without_repair(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)
    store.append(make_event("evt-1", "started"))

    with path.open("ab") as stream:
        stream.write(b'{"partial":')

    original = path.read_bytes()
    report = inspect_trace_file(path)

    assert report["status"] == "corrupt"
    assert report["valid"] is False
    assert report["record_count"] == 1
    assert report["first_invalid_line"] == 2
    assert "incomplete" in report["error"]
    assert path.read_bytes() == original


def test_modified_payload_is_detected_without_repair(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = ExecutionTraceStore(path, fsync=False)

    store.append(make_event("evt-1", "started"))
    store.append(make_event("evt-2", "completed"))

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["payload"]["success"] = False

    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    original = path.read_bytes()
    report = inspect_trace_file(path)

    assert report["status"] == "corrupt"
    assert report["record_count"] == 1
    assert report["first_invalid_line"] == 2
    assert "record hash" in report["error"]
    assert path.read_bytes() == original


def test_empty_file_and_missing_file_are_distinguished(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_bytes(b"")

    empty_report = inspect_trace_file(empty_path)
    missing_report = inspect_trace_file(
        tmp_path / "missing.jsonl"
    )

    assert empty_report["status"] == "valid"
    assert empty_report["valid"] is True
    assert empty_report["record_count"] == 0
    assert empty_report["last_valid_hash"] is None

    assert missing_report["status"] == "missing"
    assert missing_report["valid"] is False
    assert missing_report["file_sha256"] is None


def test_invalid_json_is_reported_without_modifying_file(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(b"{not-json}\n")
    original = path.read_bytes()

    report = inspect_trace_file(path)

    assert report["status"] == "corrupt"
    assert report["first_invalid_line"] == 1
    assert "valid JSON" in report["error"]
    assert path.read_bytes() == original
