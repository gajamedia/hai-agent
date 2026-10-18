from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


HERE = Path(__file__).resolve()


def _find_telemetry_candidate() -> Path:
    """Find the staged candidate, extracted-package candidate, or live route."""
    candidates = [
        # When this test is copied into the repository's tests/ directory.
        HERE.parents[1] / "_step42_2_4_1_review" / "api" / "routes" / "telemetry.py",
        HERE.parents[1] / "api" / "routes" / "telemetry.py",
        # When running directly from the extracted review package.
        HERE.parents[1] / "src" / "api" / "routes" / "telemetry.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find telemetry.py. Checked these locations:\\n" + searched
    )


CANDIDATE = _find_telemetry_candidate()
SPEC = importlib.util.spec_from_file_location(
    "step42_2_4_1_telemetry_candidate",
    CANDIDATE,
)
assert SPEC and SPEC.loader
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)


class FakeAgent:
    def __init__(self, name: str, event_count: int):
        self.name = name
        self.event_count = event_count

    def get_capability_telemetry_snapshot(self):
        return {
            "event_count": self.event_count,
            "capabilities": ({self.name: {"attempts": self.event_count}} if self.event_count else {}),
        }

    def get_capability_telemetry_events(self):
        return [{"capability": self.name, "success": True, "index": i} for i in range(self.event_count)]


def request_with(cognitive=None, execution=None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        cognitive_runtime=SimpleNamespace(agent=cognitive) if cognitive is not None else None,
        execution_runtime=SimpleNamespace(agent=execution) if execution is not None else None,
    )))


def test_explicit_cognitive_source_does_not_switch_to_execution_evidence():
    cognitive = FakeAgent("reasoning", 0)
    execution = FakeAgent("execution", 3)
    agent, source = telemetry._get_agent(request_with(cognitive, execution), "cognitive_runtime")
    assert agent is cognitive
    assert source == "cognitive_runtime"


def test_explicit_execution_source_selects_execution_agent():
    cognitive = FakeAgent("reasoning", 2)
    execution = FakeAgent("execution", 3)
    agent, source = telemetry._get_agent(request_with(cognitive, execution), "execution_runtime")
    assert agent is execution
    assert source == "execution_runtime"


def test_legacy_source_selection_remains_evidence_aware():
    cognitive = FakeAgent("reasoning", 0)
    execution = FakeAgent("execution", 3)
    agent, source = telemetry._get_agent(request_with(cognitive, execution))
    assert agent is execution
    assert source == "execution_runtime"


def test_explicit_missing_runtime_returns_503():
    with pytest.raises(HTTPException) as exc:
        telemetry._get_agent(request_with(cognitive=FakeAgent("reasoning", 0)), "execution_runtime")
    assert exc.value.status_code == 503
    assert exc.value.detail["source"] == "execution_runtime"


def test_source_endpoint_keeps_results_separate_and_labeled():
    cognitive = FakeAgent("reasoning", 1)
    execution = FakeAgent("execution", 2)
    request = request_with(cognitive, execution)
    result = telemetry.get_capability_telemetry_sources(request, limit=20)
    sources = result["sources"]
    assert sources["cognitive_runtime"]["source"] == "cognitive_runtime"
    assert sources["execution_runtime"]["source"] == "execution_runtime"
    assert sources["cognitive_runtime"]["snapshot"]["event_count"] == 1
    assert sources["execution_runtime"]["snapshot"]["event_count"] == 2
    assert sources["cognitive_runtime"]["events"][0]["capability"] == "reasoning"
    assert sources["execution_runtime"]["events"][0]["capability"] == "execution"


def test_source_endpoint_reports_missing_runtime_without_hiding_available_source():
    result = telemetry.get_capability_telemetry_sources(
        request_with(cognitive=FakeAgent("reasoning", 1)), limit=20
    )
    assert result["sources"]["cognitive_runtime"]["status"] == "ok"
    assert result["sources"]["execution_runtime"]["status"] == "unavailable"
