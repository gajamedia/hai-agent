from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "step41_2_runtime_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("step41_2_runtime_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_script_exposes_callable_runner():
    module = _load_script()
    assert callable(module.run_smoke)


def test_evidence_serializer_rejects_unknown_contract():
    module = _load_script()

    class UnknownEvidence:
        pass

    try:
        module._evidence_dict(UnknownEvidence())
    except TypeError as exc:
        assert "serialization contract" in str(exc)
    else:
        raise AssertionError("Unknown evidence contract must fail closed")


def test_report_requires_every_safety_and_trace_field():
    module = _load_script()
    required = {
        "correct_completion",
        "deterministic_verification_passed",
        "unauthorized_high_risk_actions",
        "duplicate_harmful_side_effects",
        "audit_evidence_complete",
        "required_approval_evidence_complete",
        "trace_integrity_passed",
    }
    assert len(required) == 7
