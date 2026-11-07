from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

from agent.llm.cognitive_pass_handoff import CognitivePassHandoff
from agent.llm.sqlite_cognitive_pass_handoff_store import (
    SQLiteCognitivePassHandoffStore,
)


def make_handoff(*, now, ttl_seconds=30, handoff_id="12345678-1234-5678-1234-567812345678"):
    expires = now + timedelta(seconds=ttl_seconds)
    plan = {
        "goal": "open the box",
        "rationale": "The box must be opened first.",
        "steps": [{
            "index": 1, "action": "open_box", "purpose": "Open the box",
            "preconditions": [], "expected_observation": "box is open",
            "success_criteria": "box is open", "risk": "low",
        }],
        "assumptions": [], "constraints": [], "stop_conditions": [],
        "confidence": 0.8, "provider": None, "model": None,
    }
    plan_json = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return CognitivePassHandoff.from_dict({
        "handoff_id": handoff_id, "contract_version": 1,
        "created_at": now.isoformat(), "expires_at": expires.isoformat(),
        "cognitive_cycle": 1, "goal": "open the box",
        "observation_fingerprint": hashlib.sha256(b'{"box":"closed"}').hexdigest(),
        "plan_fingerprint": hashlib.sha256(plan_json.encode("utf-8")).hexdigest(),
        "plan": plan,
    })


def issue_codes(report):
    return {issue.code for issue in report.issues}


def test_integrity_verifier_accepts_prepared_created_chain(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "prepared.sqlite3")
    created = store.create(make_handoff(now=base), now=base)

    report = store.verify_audit_integrity()

    assert report.is_valid is True
    assert report.checked_handoffs == 1
    assert report.verified_handoffs == 1
    assert report.untracked_handoffs == 0
    assert report.issues == ()


def test_integrity_verifier_accepts_consumed_chain(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "consumed.sqlite3")
    created = store.create(make_handoff(now=base), now=base)
    store.consume(created.handoff_id, now=base + timedelta(seconds=5))

    report = store.verify_audit_integrity()

    assert report.is_valid is True
    assert report.verified_handoffs == 1
    assert report.issues == ()


def test_integrity_verifier_accepts_expired_chain(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "expired.sqlite3")
    store.create(make_handoff(now=base), now=base)
    store.expire_due(now=base + timedelta(seconds=30))

    report = store.verify_audit_integrity()

    assert report.is_valid is True
    assert report.verified_handoffs == 1
    assert report.issues == ()


def test_legacy_handoff_without_events_is_reported_as_untracked(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER IF EXISTS trg_cognitive_pass_handoff_events_no_update")
        connection.execute("DROP TRIGGER IF EXISTS trg_cognitive_pass_handoff_events_no_delete")
        connection.execute("DELETE FROM cognitive_pass_handoff_events")

    report = store.verify_audit_integrity()

    assert report.is_valid is True
    assert report.checked_handoffs == 1
    assert report.verified_handoffs == 0
    assert report.untracked_handoffs == 1
    assert report.issues == ()


def test_missing_terminal_transition_event_is_detected(tmp_path):
    path = tmp_path / "missing_event.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)
    store.consume(created.handoff_id, now=base + timedelta(seconds=5))

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER IF EXISTS trg_cognitive_pass_handoff_events_no_delete")
        connection.execute(
            "DELETE FROM cognitive_pass_handoff_events WHERE event_type='consumed'"
        )

    report = store.verify_audit_integrity()

    assert report.is_valid is False
    assert "missing_terminal_transition" in issue_codes(report)


def test_duplicate_created_event_is_detected(tmp_path):
    path = tmp_path / "duplicate_created.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO cognitive_pass_handoff_events (
                event_id, handoff_id, event_type, from_status, to_status,
                occurred_at, metadata_json
            )
            VALUES (?, ?, 'created', NULL, 'prepared', ?, '{}')
            """,
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", created.handoff_id, base.isoformat()),
        )

    report = store.verify_audit_integrity()

    assert report.is_valid is False
    assert "duplicate_created_event" in issue_codes(report)


def test_illegal_event_transition_is_detected(tmp_path):
    path = tmp_path / "illegal_transition.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO cognitive_pass_handoff_events (
                event_id, handoff_id, event_type, from_status, to_status,
                occurred_at, metadata_json
            )
            VALUES (?, ?, 'consumed', 'expired', 'consumed', ?, '{}')
            """,
            ("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", created.handoff_id,
             (base + timedelta(seconds=1)).isoformat()),
        )

    report = store.verify_audit_integrity()

    assert report.is_valid is False
    assert "invalid_transition" in issue_codes(report)


def test_event_timestamp_before_creation_is_detected(tmp_path):
    path = tmp_path / "timestamp_mismatch.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO cognitive_pass_handoff_events (
                event_id, handoff_id, event_type, from_status, to_status,
                occurred_at, metadata_json
            )
            VALUES (?, ?, 'expired', 'prepared', 'expired', ?, '{}')
            """,
            ("cccccccc-cccc-4ccc-8ccc-cccccccccccc", created.handoff_id,
             (base - timedelta(seconds=1)).isoformat()),
        )

    report = store.verify_audit_integrity()

    assert report.is_valid is False
    assert "event_timestamp_regression" in issue_codes(report)


def test_integrity_verifier_is_read_only(tmp_path):
    path = tmp_path / "read_only.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)
    store.consume(created.handoff_id, now=base + timedelta(seconds=2))

    with sqlite3.connect(path) as connection:
        before_handoffs = connection.execute(
            "SELECT * FROM cognitive_pass_handoffs ORDER BY handoff_id"
        ).fetchall()
        before_events = connection.execute(
            "SELECT * FROM cognitive_pass_handoff_events ORDER BY rowid"
        ).fetchall()

    store.verify_audit_integrity()

    with sqlite3.connect(path) as connection:
        after_handoffs = connection.execute(
            "SELECT * FROM cognitive_pass_handoffs ORDER BY handoff_id"
        ).fetchall()
        after_events = connection.execute(
            "SELECT * FROM cognitive_pass_handoff_events ORDER BY rowid"
        ).fetchall()

    assert after_handoffs == before_handoffs
    assert after_events == before_events


def test_integrity_report_to_dict_is_machine_readable(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "report.sqlite3")
    store.create(make_handoff(now=base), now=base)

    report = store.verify_audit_integrity()
    payload = report.to_dict()

    assert payload["is_valid"] is True
    assert payload["checked_handoffs"] == 1
    assert payload["verified_handoffs"] == 1
    assert payload["untracked_handoffs"] == 0
    assert payload["issues"] == []
