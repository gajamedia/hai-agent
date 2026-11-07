from __future__ import annotations

import hashlib
import json

import pytest

from agent.llm.execution_trace_store import (
    ExecutionTraceIntegrityError,
    ExecutionTraceStore,
)
from agent.llm import execution_trace_recovery_transaction as recovery
from agent.llm.execution_trace_recovery_transaction import (
    TraceRecoveryConfirmationError,
    TraceRecoveryError,
    TraceRecoveryIntegrityError,
    recover_trace_to_new_file,
)


def make_event(event_id: str, event_type: str = "step_recorded") -> dict:
    return {
        "event_id": event_id,
        "execution_id": "step43-8-test",
        "event_type": event_type,
        "payload": {"action": "open_box", "success": True},
    }


def corrupt_second_record(path) -> None:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["payload"]["success"] = False
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def make_corrupt_source(path):
    store = ExecutionTraceStore(path, fsync=False)
    store.append(make_event("evt-1", "started"))
    store.append(make_event("evt-2", "completed"))
    corrupt_second_record(path)
    return path.read_bytes()


def test_recovers_valid_prefix_and_preserves_source_and_backup(tmp_path):
    source = tmp_path / "source.jsonl"
    backup_dir = tmp_path / "backups"
    recovered = tmp_path / "recovered" / "trace.jsonl"
    original_bytes = make_corrupt_source(source)
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    result = recover_trace_to_new_file(
        source, backup_dir, recovered, confirm_recovery=True
    )

    assert result["status"] == "recovered"
    assert result["source_unchanged"] is True
    assert result["source_sha256"] == original_hash
    assert result["backup_verified"] is True
    assert result["backup_sha256"] == original_hash
    assert result["recovered_record_count"] == 1
    assert result["automatic_source_repair"] is False
    assert source.read_bytes() == original_bytes
    assert open(result["backup_path"], "rb").read() == original_bytes

    records = ExecutionTraceStore(recovered, fsync=False).read_all()
    assert len(records) == 1
    assert records[0]["event_id"] == "evt-1"


def test_requires_explicit_confirmation(tmp_path):
    source = tmp_path / "source.jsonl"
    make_corrupt_source(source)

    with pytest.raises(TraceRecoveryConfirmationError):
        recover_trace_to_new_file(
            source, tmp_path / "backups", tmp_path / "recovered.jsonl"
        )

    assert not (tmp_path / "backups").exists()
    assert not (tmp_path / "recovered.jsonl").exists()


def test_refuses_valid_source_without_creating_backup(tmp_path):
    source = tmp_path / "valid.jsonl"
    ExecutionTraceStore(source, fsync=False).append(make_event("evt-1", "started"))
    original = source.read_bytes()

    with pytest.raises(TraceRecoveryError, match="corrupt trace"):
        recover_trace_to_new_file(
            source, tmp_path / "backups", tmp_path / "recovered.jsonl",
            confirm_recovery=True,
        )

    assert source.read_bytes() == original
    assert not (tmp_path / "backups").exists()
    assert not (tmp_path / "recovered.jsonl").exists()


def test_refuses_trace_with_no_valid_prefix(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_bytes(b"{invalid-json}\n")

    with pytest.raises(TraceRecoveryError, match="No verified valid prefix"):
        recover_trace_to_new_file(
            source, tmp_path / "backups", tmp_path / "recovered.jsonl",
            confirm_recovery=True,
        )

    assert not (tmp_path / "recovered.jsonl").exists()


def test_existing_output_is_refused_before_backup_creation(tmp_path):
    source = tmp_path / "source.jsonl"
    make_corrupt_source(source)
    recovered = tmp_path / "recovered.jsonl"
    recovered.write_bytes(b"keep-existing-output")
    original_output = recovered.read_bytes()

    with pytest.raises(TraceRecoveryError, match="already exists"):
        recover_trace_to_new_file(
            source, tmp_path / "backups", recovered, confirm_recovery=True
        )

    assert recovered.read_bytes() == original_output
    assert not (tmp_path / "backups").exists()


def test_recovered_file_is_valid_and_source_stays_corrupt(tmp_path):
    source = tmp_path / "source.jsonl"
    original = make_corrupt_source(source)
    recovered = tmp_path / "recovered.jsonl"

    result = recover_trace_to_new_file(
        source, tmp_path / "backups", recovered, confirm_recovery=True
    )

    assert ExecutionTraceStore(recovered, fsync=False).verify()["valid"] is True
    assert source.read_bytes() == original
    with pytest.raises(ExecutionTraceIntegrityError):
        ExecutionTraceStore(source, fsync=False).verify()
    backups = list((tmp_path / "backups").glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert result["backup_verified"] is True


def test_partial_backup_write_is_cleaned_up(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    original = make_corrupt_source(source)
    real_fsync = recovery.os.fsync

    def fail_fsync(_fd):
        raise OSError("injected backup fsync failure")

    monkeypatch.setattr(recovery.os, "fsync", fail_fsync)

    with pytest.raises(TraceRecoveryError, match="Unable to create source backup"):
        recover_trace_to_new_file(
            source, tmp_path / "backups", tmp_path / "recovered.jsonl",
            confirm_recovery=True,
        )

    assert source.read_bytes() == original
    assert list((tmp_path / "backups").glob("*.bak")) == []
    assert not (tmp_path / "recovered.jsonl").exists()
    monkeypatch.setattr(recovery.os, "fsync", real_fsync)


def test_partial_recovered_output_write_is_cleaned_up_and_backup_retained(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.jsonl"
    original = make_corrupt_source(source)
    output = tmp_path / "recovered.jsonl"
    real_fsync = recovery.os.fsync
    calls = {"count": 0}

    def fail_second_fsync(fd):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected recovered output fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(recovery.os, "fsync", fail_second_fsync)

    with pytest.raises(TraceRecoveryError, match="Unable to"):
        recover_trace_to_new_file(
            source, tmp_path / "backups", output, confirm_recovery=True
        )

    assert source.read_bytes() == original
    assert len(list((tmp_path / "backups").glob("*.bak"))) == 1
    assert not output.exists()


def test_source_change_during_recovery_removes_untrusted_output(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    original = make_corrupt_source(source)
    output = tmp_path / "recovered.jsonl"
    real_inspect = recovery.inspect_trace_file
    changed = {"done": False}

    def inspect_then_change(path):
        result = real_inspect(path)
        if path == output and not changed["done"]:
            changed["done"] = True
            source.write_bytes(original + b"external-change\n")
        return result

    monkeypatch.setattr(recovery, "inspect_trace_file", inspect_then_change)

    with pytest.raises(TraceRecoveryIntegrityError, match="Source changed during recovery"):
        recover_trace_to_new_file(
            source, tmp_path / "backups", output, confirm_recovery=True
        )

    assert source.read_bytes() == original + b"external-change\n"
    assert len(list((tmp_path / "backups").glob("*.bak"))) == 1
    assert not output.exists()
