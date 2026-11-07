from datetime import date, datetime, time
from decimal import Decimal

import pytest

from agent.llm.business_database_adapter import _json_safe


@pytest.mark.parametrize("value,expected", [
    (None, None), (True, True), (3, 3), ("x", "x"), (1.5, 1.5),
    (Decimal("12.25"), 12.25), (date(2026, 1, 2), "2026-01-02"),
    (datetime(2026, 1, 2, 3, 4), "2026-01-02T03:04:00"), (time(5, 6), "05:06:00"),
    (b"\x00\x01", "<2 bytes>"), (memoryview(b"abc"), "<3 bytes>"),
    (float("nan"), None), (float("inf"), None), (Decimal("NaN"), None),
])
def test_values_become_json_safe(value, expected):
    assert _json_safe(value) == expected


def test_unknown_objects_are_stringified():
    import uuid
    u = uuid.uuid4()
    assert _json_safe(u) == str(u)
