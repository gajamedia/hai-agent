"""Fill rules from typed text: sensitive columns stay empty instead of refusing the whole request."""
from __future__ import annotations

import json
from types import SimpleNamespace

from agent.workflow.column_fill import apply_fill, propose_fill

COLUMNS = {"nama": "string", "nik_ktp": "string", "gaji": "integer", "agama": "string"}


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer

    def complete(self, messages, **kw):
        return SimpleNamespace(content=json.dumps(self.answer))


def test_sensitive_columns_are_dropped_and_the_rest_still_filled():
    llm = FakeLLM({"supported": True, "fill": {
        "nik_ktp": {"random_int": [1, 9]}, "gaji": {"random_int": [4, 9]},
        "agama": {"choice": ["Islam", "Kristen"]}}})
    rules, extra = propose_fill(llm, "isi semua kolom lain acak", COLUMNS, list(COLUMNS), ["nama"], "tb")
    assert set(rules) == {"gaji", "agama"} and extra == []
    rows = apply_fill([{"nama": "Ani"}, {"nama": "Budi"}], rules, COLUMNS, seed=1)
    assert all(4 <= r["gaji"] <= 9 and r["agama"] in {"Islam", "Kristen"} and "nik_ktp" not in r for r in rows)


def test_an_exact_value_for_a_sensitive_column_is_kept():
    llm = FakeLLM({"supported": True, "fill": {"nik_ktp": {"value": "123"}}})
    rules, _ = propose_fill(llm, "nik 123", COLUMNS, list(COLUMNS), ["nama"], "tb")
    assert rules == {"nik_ktp": {"value": "123"}}


def test_a_sequence_gives_unique_running_numbers():
    rules = {"nama": {"sequence": ["EMP-2026-", 1, 3]}}
    rows = apply_fill([{}, {}, {}], rules, COLUMNS)
    assert [r["nama"] for r in rows] == ["EMP-2026-001", "EMP-2026-002", "EMP-2026-003"]


def test_a_database_complaint_about_a_missing_value_names_the_column():
    from agent.workflow.data_change import _missing_required_column
    assert _missing_required_column(Exception("(1364, \"Field 'nip' doesn't have a default value\")")) == "nip"
    assert _missing_required_column(Exception("NOT NULL constraint failed: tb_pegawai.nip")) == "nip"
    assert _missing_required_column(Exception("Column 'nip' cannot be null")) == "nip"
    assert _missing_required_column(Exception("something else")) is None
