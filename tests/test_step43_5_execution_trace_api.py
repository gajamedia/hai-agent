from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.main import create_app
from tests.test_api_execution import FakeCognitiveRuntime, make_plan


def test_plan_run_returns_and_persists_execution_trace(monkeypatch, tmp_path):
    """The plan-run response must agree with the persisted JSONL trace."""
    trace_path = tmp_path / "execution-traces.jsonl"
    monkeypatch.setenv("HAI_EXECUTION_TRACE_JSONL", str(trace_path))
    monkeypatch.setenv(
        "HAI_PLAN_RUN_IDEMPOTENCY_DB",
        str(tmp_path / "idempotency.sqlite3"),
    )
    monkeypatch.setenv(
        "HAI_EXECUTION_SECURITY_AUDIT_DB",
        str(tmp_path / "security-audit.sqlite3"),
    )
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan("Open the box", ["open_box"]),
    )

    app = create_app(runtime_factory=FakeCognitiveRuntime)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": {"description": "A closed box."},
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    trace_summary = body["execution_trace"]

    assert trace_summary["enabled"] is True
    execution_id = trace_summary["execution_id"]
    assert isinstance(execution_id, str) and execution_id
    assert trace_summary["persisted_event_count"] > 0
    assert trace_summary["last_persisted_sequence"] is not None
    assert trace_summary["error_count"] == 0
    assert trace_summary["last_error"] is None

    assert trace_path.is_file()
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    execution_records = [
        record for record in records if record.get("execution_id") == execution_id
    ]

    assert len(execution_records) == trace_summary["persisted_event_count"]
    assert {record["event_type"] for record in execution_records} >= {
        "started",
        "step_recorded",
        "completed",
    }
    assert trace_summary["last_persisted_sequence"] == execution_records[-1]["sequence"]

    # Verify the persisted chain links for this execution's consecutive records.
    for previous, current in zip(execution_records, execution_records[1:]):
        assert current["previous_hash"] == previous["record_hash"]
