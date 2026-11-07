"""Workflow report generation and immutable local artifact persistence.

Consumes a succeeded analytical workflow task. It does not access databases,
capability gateways, network services, or modify source/analytical task results.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Any, Mapping
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

from agent.workflow.contracts import TaskStatus, WorkflowStatus, WorkflowTask
from agent.workflow.analytics import ANALYTICAL_TASK_TYPE


REPORT_TASK_TYPE = "report.generate"
_ALLOWED_INPUT_FIELDS = frozenset({"source_task_key", "title", "format"})
_REPORT_FORMATS = frozenset({"json", "csv", "html", "xlsx"})


class WorkflowReportGenerationError(RuntimeError):
    def __init__(
        self, message: str, *, task: WorkflowTask | None = None,
        error_code: str = "workflow_report_generation_failed",
    ) -> None:
        super().__init__(message)
        self.task = task
        self.error_code = error_code


class ArtifactPersistenceError(RuntimeError):
    pass


class LocalArtifactStore:
    """Persist immutable artifacts beneath an explicitly configured root."""

    def __init__(self, root: str | Path, *, max_artifact_bytes: int = 20 * 1024 * 1024):
        if isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int) or max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be a positive integer")
        raw_root = Path(root).expanduser()
        if not str(raw_root).strip():
            raise ValueError("artifact root must be a non-empty path")
        self.root = raw_root.resolve()
        self.max_artifact_bytes = max_artifact_bytes

    def persist(
        self, *, workflow_id: str, task_id: str, artifact_id: str,
        filename: str, content: bytes, media_type: str,
    ) -> dict[str, Any]:
        for value, name in ((workflow_id, "workflow_id"), (task_id, "task_id"), (artifact_id, "artifact_id")):
            try:
                uuid.UUID(str(value))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ArtifactPersistenceError(f"{name} must be a UUID") from exc
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        if len(content) > self.max_artifact_bytes:
            raise ArtifactPersistenceError("artifact exceeds configured size limit")
        if filename not in {"report.json", "report.csv", "report.html", "report.xlsx"}:
            raise ArtifactPersistenceError("unsupported artifact filename")
        if media_type not in {
            "application/json; charset=utf-8", "text/csv; charset=utf-8",
            "text/html; charset=utf-8",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }:
            raise ArtifactPersistenceError("unsupported artifact media type")

        relative = Path(str(workflow_id)) / str(task_id) / f"{artifact_id}-{filename}"
        destination = (self.root / relative).resolve()
        try:
            destination.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactPersistenceError("artifact path escapes configured root") from exc

        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            # Write a complete temporary file in the destination directory.
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=str(destination.parent), prefix=".artifact-", suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

            # Hard-link creation is atomic and fails if the destination exists;
            # it never replaces an existing artifact.
            os.link(str(temp_path), str(destination))
            temp_path.unlink(missing_ok=True)
        except FileExistsError as exc:
            raise ArtifactPersistenceError("refusing to overwrite an existing artifact") from exc
        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if isinstance(exc, ArtifactPersistenceError):
                raise
            raise ArtifactPersistenceError("artifact persistence failed") from exc

        return {
            "artifact_id": artifact_id,
            "filename": filename,
            "relative_path": relative.as_posix(),
            "media_type": media_type,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(
            dict(payload), ensure_ascii=False, allow_nan=False,
            sort_keys=True, indent=2,
        ) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("report contains values that are not strict JSON") from exc


def _csv_bytes(aggregation: Any) -> bytes:
    if not isinstance(aggregation, Mapping):
        raise ValueError("CSV report requires an aggregation result")
    rows = aggregation.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("CSV report requires at least one aggregated row")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("aggregation rows must be JSON objects")
    fieldnames = list(rows[0].keys())
    if not fieldnames or any(list(row.keys()) != fieldnames for row in rows):
        raise ValueError("aggregation rows must have consistent columns")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        safe_row = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                safe_row[key] = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
            else:
                safe_row[key] = value
        writer.writerow(safe_row)
    return output.getvalue().encode("utf-8-sig")


def _aggregation_rows(aggregation: Any) -> tuple[list[str], list[Mapping[str, Any]]]:
    """Validate the tabular report shape shared by CSV, HTML, and XLSX."""
    if not isinstance(aggregation, Mapping):
        raise ValueError("tabular report requires an aggregation result")
    rows = aggregation.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("tabular report requires at least one aggregated row")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("aggregation rows must be JSON objects")
    fieldnames = list(rows[0].keys())
    if not fieldnames or any(list(row.keys()) != fieldnames for row in rows):
        raise ValueError("aggregation rows must have consistent columns")
    return fieldnames, rows


def _cell_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return "" if value is None else str(value)


def _html_bytes(*, title: str, aggregation: Any) -> bytes:
    fieldnames, rows = _aggregation_rows(aggregation)
    header = "".join(f"<th>{html_escape(name)}</th>" for name in fieldnames)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{html_escape(_cell_text(row[name]))}</td>" for name in fieldnames
        ) + "</tr>"
        for row in rows
    )
    document = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html_escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.45rem;text-align:left}"
        "th{background:#f1f5f9}</style></head><body>"
        f"<h1>{html_escape(title)}</h1><table><thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>"
    )
    return document.encode("utf-8")


def _xlsx_bytes(*, title: str, aggregation: Any) -> bytes:
    """Produce a minimal, dependency-free XLSX workbook from validated rows.

    Formula cells are deliberately never emitted: all user/data values become
    inline strings, preventing spreadsheet formula injection on open.
    """
    fieldnames, rows = _aggregation_rows(aggregation)
    output = io.BytesIO()
    def cell(value: Any, column: int, row: int) -> str:
        # Excel column labels beyond Z are not needed for the small report
        # schemas accepted here, but this keeps the encoder generally correct.
        letters = ""
        position = column
        while position:
            position, remainder = divmod(position - 1, 26)
            letters = chr(65 + remainder) + letters
        reference = f"{letters}{row}"
        text = xml_escape(_cell_text(value), {'"': '&quot;'})
        return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'

    worksheet_rows = []
    for index, values in enumerate([fieldnames, *rows], start=1):
        worksheet_rows.append(
            f'<row r="{index}">' + "".join(
                cell(value, column, index)
                for column, value in enumerate(
                    values if isinstance(values, list) else [values[name] for name in fieldnames],
                    start=1,
                )
            ) + "</row>"
        )
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>' + "".join(worksheet_rows) + '</sheetData></worksheet>'
    )
    safe_title = title.replace("'", "")[:31] or "Report"
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>""")
        archive.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>""")
        archive.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name=""" + xml_escape(safe_title, {'"': '&quot;'}) + """" sheetId="1" r:id="rId1"/></sheets></workbook>""")
        archive.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>""")
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
    return output.getvalue()


class WorkflowReportTaskExecutor:
    """Build a JSON, CSV, HTML, or XLSX report from verified analytical data."""

    def __init__(
        self, *, store: Any, artifact_store: LocalArtifactStore,
        max_report_bytes: int = 10 * 1024 * 1024,
    ):
        if store is None or artifact_store is None:
            raise TypeError("store and artifact_store are required")
        for name in ("get_workflow", "get_task", "get_task_dependencies", "transition_task"):
            if not callable(getattr(store, name, None)):
                raise TypeError(f"store must expose callable {name}(...)")
        if isinstance(max_report_bytes, bool) or not isinstance(max_report_bytes, int) or max_report_bytes < 1:
            raise ValueError("max_report_bytes must be a positive integer")
        self.store = store
        self.artifact_store = artifact_store
        self.max_report_bytes = max_report_bytes

    def _fail(self, claimed: WorkflowTask, code: str, message: str) -> WorkflowTask:
        try:
            return self.store.transition_task(
                claimed.task_id, TaskStatus.FAILED, expected_version=claimed.version,
                error_code=code, metadata={"reason": code},
            )
        except Exception as exc:
            raise WorkflowReportGenerationError(
                "Report failed and failure state could not be persisted.",
                task=claimed, error_code="report_failure_persistence_failed",
            ) from exc

    def execute_task(self, workflow_id: str, task_id: str) -> WorkflowTask:
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise WorkflowReportGenerationError("Workflow does not exist.", error_code="workflow_not_found")
        task = self.store.get_task(task_id)
        if task is None or task.workflow_id != workflow_id:
            raise WorkflowReportGenerationError("Task does not belong to workflow.", error_code="task_not_found")
        if workflow.status is not WorkflowStatus.RUNNING:
            raise WorkflowReportGenerationError("Workflow must be running.", task=task, error_code="workflow_not_running")
        if task.task_type != REPORT_TASK_TYPE:
            raise WorkflowReportGenerationError("Unsupported task type.", task=task, error_code="unsupported_task_type")
        if task.status is not TaskStatus.READY:
            raise WorkflowReportGenerationError("Report task must be ready.", task=task, error_code="task_not_ready")

        payload = dict(task.input_data)
        unknown = set(payload) - _ALLOWED_INPUT_FIELDS
        if unknown:
            raise WorkflowReportGenerationError("Report task input contains unsupported fields.", task=task, error_code="invalid_report_input")
        source_key = payload.get("source_task_key")
        title = payload.get("title", "Workflow Report")
        output_format = payload.get("format", "json")
        if not isinstance(source_key, str) or not source_key.strip():
            raise WorkflowReportGenerationError("source_task_key is required.", task=task, error_code="invalid_report_input")
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise WorkflowReportGenerationError("title must contain 1-200 characters.", task=task, error_code="invalid_report_input")
        if output_format not in _REPORT_FORMATS:
            raise WorkflowReportGenerationError(
                "format must be json, csv, html, or xlsx.",
                task=task,
                error_code="unsupported_report_format",
            )

        dependencies = self.store.get_task_dependencies(workflow_id, task.task_key)
        if source_key not in dependencies:
            raise WorkflowReportGenerationError("Analytical source must be a direct dependency.", task=task, error_code="invalid_report_source")
        source = next((item for item in self.store.list_tasks(workflow_id) if item.task_key == source_key), None) if callable(getattr(self.store, "list_tasks", None)) else None
        if source is None or source.task_type != ANALYTICAL_TASK_TYPE or source.status is not TaskStatus.SUCCEEDED:
            raise WorkflowReportGenerationError("Source analytical task must have succeeded.", task=task, error_code="report_source_not_succeeded")
        analytical = dict(source.result_data or {})
        required = {"source_task_key", "source_execution_id", "source_row_count", "dataset_validation", "data_quality", "aggregation", "descriptive_statistics"}
        if not required.issubset(analytical):
            raise WorkflowReportGenerationError("Analytical result is incomplete.", task=task, error_code="invalid_analytical_result")
        if analytical["dataset_validation"].get("valid") is not True or analytical["data_quality"].get("valid") is not True:
            raise WorkflowReportGenerationError("Analytical validation did not pass.", task=task, error_code="analytical_result_not_verified")

        if output_format == "json":
            report = {
                "report_version": 1,
                "title": title.strip(),
                "workflow_id": workflow_id,
                "task_id": task_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "task_key": source.task_key,
                    "task_id": source.task_id,
                    "execution_id": analytical["source_execution_id"],
                    "row_count": analytical["source_row_count"],
                },
                "validation": analytical["dataset_validation"],
                "data_quality": analytical["data_quality"],
                "aggregation": analytical["aggregation"],
                "descriptive_statistics": analytical["descriptive_statistics"],
            }
            content = _json_bytes(report)
            filename, media_type = "report.json", "application/json; charset=utf-8"
        elif output_format == "csv":
            content = _csv_bytes(analytical["aggregation"])
            filename, media_type = "report.csv", "text/csv; charset=utf-8"
        elif output_format == "html":
            content = _html_bytes(title=title.strip(), aggregation=analytical["aggregation"])
            filename, media_type = "report.html", "text/html; charset=utf-8"
        else:
            content = _xlsx_bytes(title=title.strip(), aggregation=analytical["aggregation"])
            filename = "report.xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        if len(content) > self.max_report_bytes:
            raise WorkflowReportGenerationError("Rendered report exceeds size limit.", task=task, error_code="report_too_large")

        claimed = self.store.transition_task(
            task.task_id, TaskStatus.RUNNING, expected_version=task.version,
            metadata={"reason": "report_generation_started", "format": output_format},
        )
        artifact_id = str(uuid.uuid4())
        try:
            artifact = self.artifact_store.persist(
                workflow_id=workflow_id, task_id=task_id, artifact_id=artifact_id,
                filename=filename, content=content, media_type=media_type,
            )
            result_data = {
                "report_format": output_format,
                "title": title.strip(),
                "source_task_key": source.task_key,
                "source_task_id": source.task_id,
                "source_execution_id": analytical["source_execution_id"],
                "artifact": artifact,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            encoded = _json_bytes(result_data)
            if len(encoded) > self.max_report_bytes:
                raise ValueError("artifact metadata exceeds report size limit")
            completed = self.store.transition_task(
                claimed.task_id, TaskStatus.SUCCEEDED, expected_version=claimed.version,
                result_data=result_data,
                metadata={"reason": "report_generation_completed", "artifact_id": artifact_id},
            )
            return completed
        except Exception as exc:
            failed = self._fail(
                claimed,
                "report_artifact_persistence_failed",
                "Report artifact could not be persisted.",
            )
            raise WorkflowReportGenerationError(
                "Report artifact could not be persisted.", task=failed,
                error_code="report_artifact_persistence_failed",
            ) from exc


__all__ = [
    "REPORT_TASK_TYPE", "ArtifactPersistenceError", "LocalArtifactStore",
    "WorkflowReportGenerationError", "WorkflowReportTaskExecutor",
]
