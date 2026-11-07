from __future__ import annotations

"""Append-only JSONL persistence and read-only replay for execution traces.

The store uses both an in-process reentrant lock and an operating-system file
lock. The sidecar lock coordinates independent ExecutionTraceStore instances
and worker processes that share the same local trace path.

The JSONL schema and hash-chain format remain compatible with schema version 1.
This module never invokes an executor, action, planner, or agent.
"""

import errno
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping


class ExecutionTraceError(RuntimeError):
    """Base error for trace-store operations."""


class ExecutionTraceInputError(ExecutionTraceError, ValueError):
    """Raised when an event does not meet the trace contract."""


class ExecutionTraceIntegrityError(ExecutionTraceError):
    """Raised when persisted trace data is malformed or inconsistent."""


class ExecutionTraceStore:
    """Append-only, hash-chained JSONL event store with process-safe appends.

    Identical duplicate event IDs are idempotent; conflicting duplicates are
    rejected. All reads and writes acquire the same sidecar OS lock so a reader
    cannot observe a writer halfway through an append.

    The lock is intended for processes sharing a local filesystem path. Network
    filesystems may have different locking guarantees and are not covered.
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
        self._lock_path = self.path.with_name(self.path.name + ".lock")
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
        return hashlib.sha256(
            cls._canonical_json(material).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _required_text(event: Mapping[str, Any], key: str) -> str:
        value = event.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ExecutionTraceInputError(
                f"event.{key} must be a non-empty string."
            )
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
        detached_payload = json.loads(self._canonical_json(dict(payload)))
        return {
            "event_id": event_id,
            "execution_id": execution_id,
            "event_type": event_type,
            "payload": detached_payload,
        }

    @contextmanager
    def _process_file_lock(self) -> Iterator[None]:
        """Acquire an exclusive OS lock on a stable sidecar file."""
        try:
            stream = self._lock_path.open("a+b")
        except OSError as exc:
            raise ExecutionTraceError(
                f"Unable to open trace lock file: {exc}"
            ) from exc

        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                while True:
                    try:
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                        break
                    except OSError as exc:
                        if exc.errno not in {
                            errno.EACCES,
                            errno.EAGAIN,
                            errno.EDEADLK,
                        }:
                            raise
                        time.sleep(0.05)
            elif os.name == "posix":
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                acquired = True
            else:
                raise ExecutionTraceError(
                    f"Process-safe trace locking is unsupported on os.name={os.name!r}."
                )

            yield
        except ExecutionTraceError:
            raise
        except OSError as exc:
            raise ExecutionTraceError(
                f"Unable to acquire or use trace lock: {exc}"
            ) from exc
        finally:
            if acquired:
                try:
                    if os.name == "nt":
                        import msvcrt

                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    elif os.name == "posix":
                        import fcntl

                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                finally:
                    stream.close()
            else:
                stream.close()

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        # The thread lock prevents same-instance threads from racing; the
        # sidecar OS lock coordinates independent instances and processes.
        with self._lock:
            with self._process_file_lock():
                yield

    def _read_all_unlocked(self) -> list[dict[str, Any]]:
        """Read and verify while the caller already holds both locks."""
        if not self.path.exists():
            return []

        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
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
                            f"Line {line_number} has sequence "
                            f"{record.get('sequence')!r}; expected {expected_sequence}."
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
                    seen_ids.add(event_id)
                    previous_hash = actual_hash
        except OSError as exc:
            raise ExecutionTraceError(
                f"Unable to read trace store: {exc}"
            ) from exc

        return records

    def read_all(self) -> list[dict[str, Any]]:
        """Read and verify the complete store under the process lock."""
        with self._exclusive_lock():
            return self._read_all_unlocked()

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Persist one event, returning its immutable persisted envelope."""
        normalised = self._normalise_event(event)

        with self._exclusive_lock():
            existing = self._read_all_unlocked()
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
                            f"event_id {normalised['event_id']!r} "
                            "already exists with different content."
                        )
                    return dict(record)

            previous_hash = (
                existing[-1]["record_hash"] if existing else "0" * 64
            )
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
                raise ExecutionTraceError(
                    f"Unable to append trace event: {exc}"
                ) from exc

            return dict(record)

    def replay(self, execution_id: str | None = None) -> list[dict[str, Any]]:
        """Return a read-only logical replay; this never runs stored actions."""
        if execution_id is not None and (
            not isinstance(execution_id, str) or not execution_id.strip()
        ):
            raise ExecutionTraceInputError(
                "execution_id must be non-empty when provided."
            )
        records = self.read_all()
        if execution_id is not None:
            records = [
                record for record in records
                if record["execution_id"] == execution_id
            ]
        return [dict(record) for record in records]

    def verify(self) -> dict[str, Any]:
        """Return an integrity summary or raise on corrupted storage."""
        records = self.read_all()
        return {
            "valid": True,
            "record_count": len(records),
            "execution_ids": sorted(
                {record["execution_id"] for record in records}
            ),
            "first_sequence": records[0]["sequence"] if records else None,
            "last_sequence": records[-1]["sequence"] if records else None,
            "last_hash": records[-1]["record_hash"] if records else None,
        }


__all__ = [
    "ExecutionTraceError",
    "ExecutionTraceInputError",
    "ExecutionTraceIntegrityError",
    "ExecutionTraceStore",
]
