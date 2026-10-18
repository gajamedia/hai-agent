from __future__ import annotations

"""Append-only JSONL persistence and read-only replay for execution traces.

This module stores events; it deliberately never invokes an executor, action,
planner, or agent. Events are hash chained so accidental edits, truncation,
reordering, and duplicate event identifiers can be detected on read.
"""

import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Mapping


class ExecutionTraceError(RuntimeError):
    """Base error for execution trace operations."""


class ExecutionTraceInputError(ExecutionTraceError, ValueError):
    """Raised when an event does not meet the trace contract."""


class ExecutionTraceIntegrityError(ExecutionTraceError):
    """Raised when persisted trace data is malformed or inconsistent."""


class ExecutionTraceStore:
    """Append-only, hash-chained JSONL event store with idempotent appends.

    Each event must provide non-empty ``event_id``, ``execution_id``, and
    ``event_type``. ``payload`` may be any JSON-serializable mapping.

    Re-appending the same event_id and same event content is idempotent. Using
    the same event_id for different content raises an integrity error.
    ``replay`` returns stored events in sequence order and never executes them.
    """

    SCHEMA_VERSION = 1
    _HASH_FIELDS = (
        "schema_version",
        "sequence",
        "event_id",
        "execution_id",
        "event_type",
        "payload",
        "previous_hash",
    )

    def __init__(self, path: str | os.PathLike[str], *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = bool(fsync)
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _canonical_json(value: Any) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionTraceInputError(
                f"Trace event must be JSON-serializable: {exc}"
            ) from exc

    @classmethod
    def _record_hash(cls, record: Mapping[str, Any]) -> str:
        material = {key: record[key] for key in cls._HASH_FIELDS}
        return hashlib.sha256(cls._canonical_json(material).encode("utf-8")).hexdigest()

    @staticmethod
    def _required_text(event: Mapping[str, Any], key: str) -> str:
        value = event.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ExecutionTraceInputError(f"event.{key} must be a non-empty string.")
        return value.strip()

    def _normalise_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise ExecutionTraceInputError("event must be a mapping.")
        event_id = self._required_text(event, "event_id")
        execution_id = self._required_text(event, "execution_id")
        event_type = self._required_text(event, "event_type")
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            raise ExecutionTraceInputError("event.payload must be a mapping.")
        # Round-trip to detach caller-owned nested structures and validate JSON.
        detached_payload = json.loads(self._canonical_json(dict(payload)))
        return {
            "event_id": event_id,
            "execution_id": execution_id,
            "event_type": event_type,
            "payload": detached_payload,
        }

    def read_all(self) -> list[dict[str, Any]]:
        """Read and verify the complete store; fail closed on any corruption."""
        with self._lock:
            if not self.path.exists():
                return []
            records: list[dict[str, Any]] = []
            seen_ids: dict[str, dict[str, Any]] = {}
            previous_hash = "0" * 64
            try:
                with self.path.open("r", encoding="utf-8", newline="") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.endswith("\n"):
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} is incomplete (missing newline)."
                            )
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} is not valid JSON."
                            ) from exc
                        if not isinstance(record, dict):
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} must contain a JSON object."
                            )
                        expected_sequence = len(records) + 1
                        if record.get("schema_version") != self.SCHEMA_VERSION:
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} has an unsupported schema version."
                            )
                        if record.get("sequence") != expected_sequence:
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} has sequence {record.get('sequence')!r}; "
                                f"expected {expected_sequence}."
                            )
                        if record.get("previous_hash") != previous_hash:
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} breaks the hash chain."
                            )
                        required = set(self._HASH_FIELDS) | {"record_hash"}
                        if set(record) != required:
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} has missing or unexpected fields."
                            )
                        event_id = record.get("event_id")
                        if not isinstance(event_id, str) or not event_id.strip():
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} has an invalid event_id."
                            )
                        if event_id in seen_ids:
                            raise ExecutionTraceIntegrityError(
                                f"Duplicate event_id {event_id!r} at line {line_number}."
                            )
                        if not all(
                            isinstance(record.get(key), str) and record[key].strip()
                            for key in ("execution_id", "event_type")
                        ):
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} has invalid event metadata."
                            )
                        if not isinstance(record.get("payload"), dict):
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} payload must be an object."
                            )
                        actual_hash = self._record_hash(record)
                        if record.get("record_hash") != actual_hash:
                            raise ExecutionTraceIntegrityError(
                                f"Line {line_number} record hash does not match."
                            )
                        records.append(record)
                        seen_ids[event_id] = record
                        previous_hash = actual_hash
            except OSError as exc:
                raise ExecutionTraceError(f"Unable to read trace store: {exc}") from exc
            return records

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Persist one event, returning its immutable persisted envelope.

        A duplicate identical event_id is a no-op. A conflicting duplicate is
        rejected instead of silently overwriting the original event.
        """
        normalised = self._normalise_event(event)
        with self._lock:
            existing = self.read_all()
            for record in existing:
                if record["event_id"] == normalised["event_id"]:
                    original = {
                        "event_id": record["event_id"],
                        "execution_id": record["execution_id"],
                        "event_type": record["event_type"],
                        "payload": record["payload"],
                    }
                    if original != normalised:
                        raise ExecutionTraceIntegrityError(
                            f"event_id {normalised['event_id']!r} already exists with different content."
                        )
                    return dict(record)
            previous_hash = existing[-1]["record_hash"] if existing else "0" * 64
            record: dict[str, Any] = {
                "schema_version": self.SCHEMA_VERSION,
                "sequence": len(existing) + 1,
                **normalised,
                "previous_hash": previous_hash,
            }
            record["record_hash"] = self._record_hash(record)
            line = self._canonical_json(record) + "\n"
            try:
                with self.path.open("a", encoding="utf-8", newline="") as stream:
                    stream.write(line)
                    stream.flush()
                    if self.fsync:
                        os.fsync(stream.fileno())
            except OSError as exc:
                raise ExecutionTraceError(f"Unable to append trace event: {exc}") from exc
            return dict(record)

    def replay(self, execution_id: str | None = None) -> list[dict[str, Any]]:
        """Return a read-only logical replay; this never runs stored actions."""
        if execution_id is not None and (
            not isinstance(execution_id, str) or not execution_id.strip()
        ):
            raise ExecutionTraceInputError("execution_id must be non-empty when provided.")
        records = self.read_all()
        if execution_id is not None:
            records = [r for r in records if r["execution_id"] == execution_id]
        return [dict(record) for record in records]

    def verify(self) -> dict[str, Any]:
        """Return a concise integrity summary or raise on corrupted storage."""
        records = self.read_all()
        return {
            "valid": True,
            "record_count": len(records),
            "execution_ids": sorted({r["execution_id"] for r in records}),
            "first_sequence": records[0]["sequence"] if records else None,
            "last_sequence": records[-1]["sequence"] if records else None,
            "last_hash": records[-1]["record_hash"] if records else None,
        }
