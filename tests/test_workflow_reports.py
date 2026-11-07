from __future__ import annotations

import csv
import hashlib
import json
import uuid
from zipfile import ZipFile

import pytest

from agent.workflow import SQLiteWorkflowStore, TaskDependencyPlanner, TaskSpec, TaskStatus, WorkflowStatus
from agent.workflow.analytics import ANALYTICAL_TASK_TYPE
from agent.workflow.reports import (
    ArtifactPersistenceError, LocalArtifactStore, REPORT_TASK_TYPE,
    WorkflowReportGenerationError, WorkflowReportTaskExecutor,
)

ANALYTICS = {
    "source_task_key": "analyze",
    "source_execution_id": "execution-123",
    "source_row_count": 3,
    "dataset_validation": {"valid": True, "row_count": 3, "issues": []},
    "data_quality": {"valid": True, "issue_count": 0, "issues": []},
    "aggregation": {
        "source_row_count": 3, "group_count": 2, "group_by": ["region"],
        "rows": [{"region": "west", "revenue": 175}, {"region": "east", "revenue": 50}],
    },
    "descriptive_statistics": {"numeric": [{"column": "amount", "mean": 75.0}], "categorical": []},
}


def setup_flow(tmp_path, *, report_input=None, source_result=None, source_status=TaskStatus.SUCCEEDED):
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    wf = store.create_workflow(goal="database to report", idempotency_key="report-test")
    specs = [
        TaskSpec("analyze", ANALYTICAL_TASK_TYPE, "Analyze rows"),
        TaskSpec("report", REPORT_TASK_TYPE, "Build report", ("analyze",),
                 input_data=report_input or {"source_task_key": "analyze", "title": "Sales", "format": "json"}),
    ]
    tasks = store.create_tasks_from_plan(wf.workflow_id, TaskDependencyPlanner().plan(specs))
    source = next(x for x in tasks if x.task_key == "analyze")
    report = next(x for x in tasks if x.task_key == "report")
    source = store.transition_task(source.task_id, TaskStatus.READY, expected_version=source.version)
    source = store.transition_task(source.task_id, TaskStatus.RUNNING, expected_version=source.version)
    source = store.transition_task(
        source.task_id, source_status, expected_version=source.version,
        result_data=ANALYTICS if source_result is None else source_result,
    )
    store.refresh_task_readiness(wf.workflow_id)
    wf = store.get_workflow(wf.workflow_id)
    wf = store.transition_workflow(wf.workflow_id, WorkflowStatus.PLANNED, expected_version=wf.version)
    wf = store.transition_workflow(wf.workflow_id, WorkflowStatus.READY, expected_version=wf.version)
    wf = store.transition_workflow(wf.workflow_id, WorkflowStatus.RUNNING, expected_version=wf.version)
    report = store.get_task(report.task_id)
    return store, wf, source, report


def make_executor(tmp_path, store, *, max_report_bytes=1024 * 1024):
    return WorkflowReportTaskExecutor(
        store=store, artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        max_report_bytes=max_report_bytes,
    )


def test_json_report_is_persisted_and_task_succeeds(tmp_path):
    store, wf, source, task = setup_flow(tmp_path)
    result = make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    assert result.status is TaskStatus.SUCCEEDED
    artifact = result.result_data["artifact"]
    path = tmp_path / "artifacts" / artifact["relative_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["title"] == "Sales"
    assert payload["source"]["execution_id"] == "execution-123"
    assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_csv_report_contains_aggregated_rows(tmp_path):
    store, wf, source, task = setup_flow(
        tmp_path, report_input={"source_task_key": "analyze", "format": "csv"}
    )
    result = make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    artifact = result.result_data["artifact"]
    path = tmp_path / "artifacts" / artifact["relative_path"]
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    assert rows == [{"region": "west", "revenue": "175"}, {"region": "east", "revenue": "50"}]


def test_html_report_escapes_title_and_data_for_browser_rendering(tmp_path):
    analytics = dict(ANALYTICS)
    analytics["aggregation"] = {
        "rows": [{"region": "<script>alert(1)</script>", "revenue": 175}],
    }
    store, wf, _, task = setup_flow(
        tmp_path,
        source_result=analytics,
        report_input={"source_task_key": "analyze", "title": "<Sales>", "format": "html"},
    )
    result = make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    artifact = result.result_data["artifact"]
    html = (tmp_path / "artifacts" / artifact["relative_path"]).read_text(encoding="utf-8")
    assert artifact["filename"] == "report.html"
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>" not in html


def test_xlsx_report_is_a_valid_workbook_and_uses_inline_strings(tmp_path):
    store, wf, _, task = setup_flow(
        tmp_path, report_input={"source_task_key": "analyze", "format": "xlsx"}
    )
    result = make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    artifact = result.result_data["artifact"]
    path = tmp_path / "artifacts" / artifact["relative_path"]
    assert artifact["filename"] == "report.xlsx"
    assert artifact["media_type"].endswith("spreadsheetml.sheet")
    with ZipFile(path) as workbook:
        assert "xl/workbook.xml" in workbook.namelist()
        worksheet = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert 't="inlineStr"' in worksheet
    assert "west" in worksheet


def test_source_analysis_is_not_modified(tmp_path):
    store, wf, source, task = setup_flow(tmp_path)
    original = dict(source.result_data)
    make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    assert store.get_task(source.task_id).result_data == original


@pytest.mark.parametrize("bad_input", [
    {"source_task_key": "analyze", "format": "pdf"},
    {"source_task_key": "analyze", "format": "json", "file_path": "../../evil"},
    {"source_task_key": "", "format": "json"},
])
def test_invalid_report_input_rejected_before_claim(tmp_path, bad_input):
    store, wf, source, task = setup_flow(tmp_path, report_input=bad_input)
    with pytest.raises(WorkflowReportGenerationError):
        make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    assert store.get_task(task.task_id).status is TaskStatus.READY
    assert store.get_task(task.task_id).attempt_count == 0


def test_report_source_must_be_direct_dependency(tmp_path):
    store, wf, source, task = setup_flow(
        tmp_path,
        report_input={"source_task_key": "not_a_dependency", "format": "json"},
    )
    with pytest.raises(WorkflowReportGenerationError) as error:
        make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    assert error.value.error_code == "invalid_report_source"
    assert store.get_task(task.task_id).status is TaskStatus.READY


def test_source_must_be_successful_analytical_task(tmp_path):
    store, wf, source, task = setup_flow(tmp_path, source_result={"wrong": "shape"})
    with pytest.raises(WorkflowReportGenerationError) as error:
        make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    assert error.value.error_code == "invalid_analytical_result"


def test_invalid_analytical_quality_is_rejected(tmp_path):
    bad = dict(ANALYTICS)
    bad["data_quality"] = {"valid": False, "issues": ["bad"]}
    store, wf, source, task = setup_flow(tmp_path, source_result=bad)
    with pytest.raises(WorkflowReportGenerationError) as error:
        make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    assert error.value.error_code == "analytical_result_not_verified"


def test_csv_requires_aggregation_rows(tmp_path):
    bad = dict(ANALYTICS)
    bad["aggregation"] = {"rows": []}
    store, wf, source, task = setup_flow(
        tmp_path, source_result=bad,
        report_input={"source_task_key": "analyze", "format": "csv"},
    )
    with pytest.raises(ValueError):
        make_executor(tmp_path, store).execute_task(wf.workflow_id, task.task_id)
    assert store.get_task(task.task_id).status is TaskStatus.READY


def test_artifact_store_refuses_overwrite(tmp_path):
    store = LocalArtifactStore(tmp_path / "store")
    wf, task, artifact = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    kwargs = dict(
        workflow_id=wf, task_id=task, artifact_id=artifact,
        filename="report.json", content=b'{"ok":true}\n',
        media_type="application/json; charset=utf-8",
    )
    store.persist(**kwargs)
    with pytest.raises(ArtifactPersistenceError, match="overwrite"):
        store.persist(**kwargs)


def test_artifact_store_rejects_non_uuid_and_oversize(tmp_path):
    store = LocalArtifactStore(tmp_path / "store", max_artifact_bytes=4)
    with pytest.raises(ArtifactPersistenceError, match="UUID"):
        store.persist(
            workflow_id="../bad", task_id=str(uuid.uuid4()), artifact_id=str(uuid.uuid4()),
            filename="report.json", content=b"{}", media_type="application/json; charset=utf-8",
        )
    with pytest.raises(ArtifactPersistenceError, match="size"):
        store.persist(
            workflow_id=str(uuid.uuid4()), task_id=str(uuid.uuid4()), artifact_id=str(uuid.uuid4()),
            filename="report.json", content=b"12345", media_type="application/json; charset=utf-8",
        )


def test_task_must_be_ready_and_workflow_running(tmp_path):
    store, wf, source, task = setup_flow(tmp_path)
    executor = make_executor(tmp_path, store)
    current = store.get_workflow(wf.workflow_id)
    store.transition_workflow(wf.workflow_id, WorkflowStatus.BLOCKED, expected_version=current.version)
    with pytest.raises(WorkflowReportGenerationError) as error:
        executor.execute_task(wf.workflow_id, task.task_id)
    assert error.value.error_code == "workflow_not_running"


def test_task_type_must_be_report_generate(tmp_path):
    store, wf, source, task = setup_flow(tmp_path)
    # The task's immutable type is set during plan creation; create a separate
    # task with another type in a separate workflow to test the executor guard.
    other_wf = store.create_workflow(goal="other", idempotency_key="other")
    other = store.create_task(
        other_wf.workflow_id, task_key="wrong", task_type="other",
        description="wrong task",
    )
    other_wf = store.transition_workflow(other_wf.workflow_id, WorkflowStatus.PLANNED, expected_version=other_wf.version)
    other_wf = store.transition_workflow(other_wf.workflow_id, WorkflowStatus.READY, expected_version=other_wf.version)
    store.transition_workflow(other_wf.workflow_id, WorkflowStatus.RUNNING, expected_version=other_wf.version)
    other = store.transition_task(other.task_id, TaskStatus.READY, expected_version=other.version)
    with pytest.raises(WorkflowReportGenerationError) as error:
        make_executor(tmp_path, store).execute_task(other_wf.workflow_id, other.task_id)
    assert error.value.error_code == "unsupported_task_type"
