
from __future__ import annotations

"""Read-only inspection for JSONL execution trace files.

This module never repairs, truncates, appends to, or rewrites the inspected
trace file. It reports the valid prefix and the first detected integrity
failure so recovery can be handled separately and explicitly.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from .execution_trace_store import ExecutionTraceStore


def inspect_trace_file(path: str | Path) -> dict[str, Any]:
    """Inspect a trace file without modifying its contents.

    Returns:
        path: Inspected path.
        status: valid, corrupt, missing, or unreadable.
        valid: Whether the complete file passed inspection.
        record_count: Number of consecutive valid records.
        checked_lines: Number of lines examined, including the bad line.
        first_invalid_line: First line that failed validation, if any.
        error: Description of the first detected failure.
        last_valid_sequence: Sequence of the final valid record.
        last_valid_hash: Hash of the final valid record.
        file_size_bytes: Source file size.
        file_sha256: SHA-256 fingerprint of the exact source bytes.
    """
    trace_path = Path(path)

    if not trace_path.exists():
        return {
            "path": str(trace_path),
            "status": "missing",
            "valid": False,
            "record_count": 0,
            "checked_lines": 0,
            "first_invalid_line": None,
            "error": "Trace file does not exist.",
            "last_valid_sequence": None,
            "last_valid_hash": None,
            "file_size_bytes": None,
            "file_sha256": None,
        }

    try:
        source_bytes = trace_path.read_bytes()
    except OSError as exc:
        return {
            "path": str(trace_path),
            "status": "unreadable",
            "valid": False,
            "record_count": 0,
            "checked_lines": 0,
            "first_invalid_line": None,
            "error": f"Unable to read trace file: {exc}",
            "last_valid_sequence": None,
            "last_valid_hash": None,
            "file_size_bytes": None,
            "file_sha256": None,
        }

    digest = hashlib.sha256(source_bytes).hexdigest()
    lines = source_bytes.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_hash = "0" * 64

    def make_report(
        status: str,
        valid: bool,
        invalid_line: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        last = records[-1] if records else None
        return {
            "path": str(trace_path),
            "status": status,
            "valid": valid,
            "record_count": len(records),
            "checked_lines": (
                len(records) + 1
                if invalid_line is not None
                else len(records)
            ),
            "first_invalid_line": invalid_line,
            "error": error,
            "last_valid_sequence": (
                last.get("sequence") if last else None
            ),
            "last_valid_hash": (
                last.get("record_hash") if last else None
            ),
            "file_size_bytes": len(source_bytes),
            "file_sha256": digest,
        }

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.endswith(b"\n"):
            return make_report(
                "corrupt",
                False,
                line_number,
                f"Line {line_number} is incomplete (missing newline).",
            )

        try:
            decoded_line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            return make_report(
                "corrupt",
                False,
                line_number,
                f"Line {line_number} is not valid UTF-8.",
            )

        try:
            record = json.loads(decoded_line)
        except json.JSONDecodeError:
            return make_report(
                "corrupt",
                False,
                line_number,
                f"Line {line_number} is not valid JSON.",
            )

        if not isinstance(record, dict):
            return make_report(
                "corrupt",
                False,
                line_number,
                f"Line {line_number} must contain a JSON object.",
            )

        expected_sequence = len(records) + 1
        reason = None

        if (
            record.get("schema_version")
            != ExecutionTraceStore.SCHEMA_VERSION
        ):
            reason = (
                f"Line {line_number} has an unsupported schema version."
            )

        elif record.get("sequence") != expected_sequence:
            reason = (
                f"Line {line_number} has sequence "
                f"{record.get('sequence')!r}; "
                f"expected {expected_sequence}."
            )

        elif record.get("previous_hash") != previous_hash:
            reason = f"Line {line_number} breaks the hash chain."

        else:
            required_fields = (
                set(ExecutionTraceStore._HASH_FIELDS)
                | {"record_hash"}
            )

            if set(record) != required_fields:
                reason = (
                    f"Line {line_number} has missing or unexpected fields."
                )

            elif (
                not isinstance(record.get("event_id"), str)
                or not record["event_id"].strip()
            ):
                reason = f"Line {line_number} has an invalid event_id."

            elif record["event_id"] in seen_ids:
                reason = (
                    f"Duplicate event_id {record['event_id']!r} "
                    f"at line {line_number}."
                )

            elif not all(
                isinstance(record.get(key), str)
                and record[key].strip()
                for key in ("execution_id", "event_type")
            ):
                reason = f"Line {line_number} has invalid event metadata."

            elif not isinstance(record.get("payload"), dict):
                reason = f"Line {line_number} payload must be an object."

            else:
                try:
                    expected_hash = ExecutionTraceStore._record_hash(record)
                except (KeyError, TypeError, ValueError) as exc:
                    reason = (
                        f"Line {line_number} cannot be hashed: {exc}"
                    )
                else:
                    if record.get("record_hash") != expected_hash:
                        reason = (
                            f"Line {line_number} record hash does not match."
                        )

        if reason is not None:
            return make_report(
                "corrupt",
                False,
                line_number,
                reason,
            )

        records.append(record)
        seen_ids.add(record["event_id"])
        previous_hash = record["record_hash"]

    return make_report("valid", True)
