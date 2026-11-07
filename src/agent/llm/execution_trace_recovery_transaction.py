from __future__ import annotations

"""Explicit, preservation-first recovery for execution trace JSONL files.

The source trace is never modified. Recovery is allowed only after explicit
confirmation, a verified backup, and validation of the source fingerprint.
Recovered output is written to a new path.
"""

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from .execution_trace_recovery import inspect_trace_file


class TraceRecoveryError(RuntimeError):
    """Base error for trace recovery operations."""


class TraceRecoveryConfirmationError(TraceRecoveryError):
    """Raised when explicit recovery confirmation is missing."""


class TraceRecoveryIntegrityError(TraceRecoveryError):
    """Raised when source, backup, or recovered output fails validation."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new_file(path: Path, data: bytes) -> None:
    """Create a file exclusively and remove it if writing fails.

    Existing files are never overwritten. Cleanup is best-effort: if the
    filesystem refuses unlinking, the original write exception is preserved.
    """
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def recover_trace_to_new_file(
    source_path: str | Path,
    backup_dir: str | Path,
    recovered_path: str | Path,
    *,
    confirm_recovery: bool = False,
) -> dict[str, Any]:
    """Back up a corrupt trace and recover its verified valid prefix.

    Requires explicit confirmation, preserves the source, refuses to overwrite
    existing backup/output files, verifies the backup and source fingerprint,
    and validates the recovered output before reporting success.
    """
    if confirm_recovery is not True:
        raise TraceRecoveryConfirmationError(
            "Recovery requires explicit confirm_recovery=True."
        )

    source = Path(source_path)
    backup_root = Path(backup_dir)
    output = Path(recovered_path)

    if not source.exists() or not source.is_file():
        raise TraceRecoveryError(f"Source trace is missing or not a file: {source}")

    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if source_resolved == output_resolved:
        raise TraceRecoveryError("Recovered output must differ from the source.")

    # Preflight the output before creating a backup. This is advisory only;
    # exclusive creation below remains the final no-overwrite guarantee.
    if output.exists():
        raise TraceRecoveryError(
            f"Recovered output already exists; refusing to overwrite: {output}"
        )
    if output.is_dir():
        raise TraceRecoveryError(
            f"Recovered output points to a directory: {output}"
        )

    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise TraceRecoveryError(f"Unable to read source trace: {exc}") from exc

    source_digest = _sha256(source_bytes)
    report = inspect_trace_file(source)

    if report.get("status") != "corrupt":
        raise TraceRecoveryError(
            "Recovery requires a corrupt trace; "
            f"inspection status was {report.get('status')!r}."
        )

    if report.get("file_sha256") != source_digest:
        raise TraceRecoveryIntegrityError(
            "Source changed while inspection was running; no recovery performed."
        )

    valid_count = report.get("record_count")
    invalid_line = report.get("first_invalid_line")
    if not isinstance(valid_count, int) or valid_count < 1:
        raise TraceRecoveryError(
            "No verified valid prefix exists; automatic reconstruction is refused."
        )
    if not isinstance(invalid_line, int) or invalid_line <= valid_count:
        raise TraceRecoveryIntegrityError(
            "Inspection report has inconsistent valid-prefix metadata."
        )

    lines = source_bytes.splitlines(keepends=True)
    if len(lines) < valid_count:
        raise TraceRecoveryIntegrityError(
            "Source bytes do not contain the reported valid prefix."
        )

    prefix_bytes = b"".join(lines[:valid_count])
    if not prefix_bytes.endswith(b"\n"):
        raise TraceRecoveryIntegrityError(
            "Verified prefix does not end at a complete record boundary."
        )

    backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / (
        f"{source.name}.{uuid.uuid4().hex}.original.bak"
    )

    try:
        _write_new_file(backup_path, source_bytes)
    except OSError as exc:
        raise TraceRecoveryError(f"Unable to create source backup: {exc}") from exc

    try:
        backup_bytes = backup_path.read_bytes()
    except OSError as exc:
        raise TraceRecoveryIntegrityError(
            f"Backup was created but could not be read back: {exc}"
        ) from exc

    backup_digest = _sha256(backup_bytes)
    if backup_digest != source_digest or backup_bytes != source_bytes:
        raise TraceRecoveryIntegrityError(
            f"Backup verification failed; original backup retained at {backup_path}."
        )

    try:
        current_source_digest = _sha256(source.read_bytes())
    except OSError as exc:
        raise TraceRecoveryIntegrityError(
            f"Cannot re-check source after backup; backup retained at {backup_path}."
        ) from exc

    if current_source_digest != source_digest:
        raise TraceRecoveryIntegrityError(
            f"Source changed after backup; backup retained at {backup_path}."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        # "xb" inside _write_new_file is authoritative against races and
        # prevents overwriting a path created after the preflight check.
        try:
            _write_new_file(output, prefix_bytes)
        except OSError as exc:
            raise TraceRecoveryError(
                f"Unable to write recovered output; backup retained at {backup_path}: {exc}"
            ) from exc

        recovered_report = inspect_trace_file(output)
        try:
            recovered_bytes = output.read_bytes()
        except OSError as exc:
            raise TraceRecoveryIntegrityError(
                f"Recovered output could not be read; backup retained at {backup_path}."
            ) from exc

        if (
            recovered_report.get("status") != "valid"
            or recovered_report.get("record_count") != valid_count
            or recovered_bytes != prefix_bytes
        ):
            raise TraceRecoveryIntegrityError(
                "Recovered output failed verification; "
                f"source and backup remain untouched. Output: {output}"
            )

        final_source_digest = _sha256(source.read_bytes())
        if final_source_digest != source_digest:
            raise TraceRecoveryIntegrityError(
                "Source changed during recovery; source and backup were not rewritten."
            )

    except Exception:
        # _write_new_file removes a partial file when its own write fails.
        # If a later verification fails, remove the complete but untrusted output.
        if output.exists():
            try:
                output.unlink()
            except OSError:
                pass
        raise

    return {
        "status": "recovered",
        "source_path": str(source),
        "source_unchanged": True,
        "source_sha256": source_digest,
        "backup_path": str(backup_path),
        "backup_sha256": backup_digest,
        "backup_verified": True,
        "recovered_path": str(output),
        "recovered_sha256": _sha256(prefix_bytes),
        "recovered_record_count": valid_count,
        "source_first_invalid_line": invalid_line,
        "source_error": report.get("error"),
        "source_bytes_preserved_in_backup": len(source_bytes),
        "recovered_bytes": len(prefix_bytes),
        "automatic_source_repair": False,
    }
