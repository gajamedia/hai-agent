"""Contract tests for Step 41.14 closure report validation helpers."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "step41_14_final_regression_closure.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("step41_14_closure", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_passed_step_report_is_accepted():
    module = _load_module()
    result = module.validate_step_report({"overall_passed": True, "total_cases": 12, "passed_cases": 12, "failed_cases": 0})
    assert result["valid"] is True


def test_failed_step_report_is_rejected():
    module = _load_module()
    result = module.validate_step_report({"overall_passed": False, "total_cases": 12, "passed_cases": 11, "failed_cases": 1})
    assert result["valid"] is False


def test_missing_report_is_not_fabricated_as_pass():
    module = _load_module()
    result = module.validate_step_report(None)
    assert result["valid"] is None


def test_90_case_matrix_requires_all_cases_passed():
    module = _load_module()
    good = module.validate_matrix_report({"overall_passed": True, "total_cases": 90, "passed_cases": 90, "failed_cases": 0})
    bad = module.validate_matrix_report({"overall_passed": True, "total_cases": 90, "passed_cases": 89, "failed_cases": 1})
    assert good["valid"] is True
    assert bad["valid"] is False
