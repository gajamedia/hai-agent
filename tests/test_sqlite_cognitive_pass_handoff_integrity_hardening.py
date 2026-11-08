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


def test_multiple_handoffs_are_verified_independently(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "multiple.sqlite3")
    first_id = "12345678-1234-5678-1234-567812345678"
    second_id = "22345678-1234-5678-1234-567812345678"
    third_id = "32345678-1234-5678-1234-567812345678"

    first = store.create(make_handoff(now=base, handoff_id=first_id), now=base)
    second = store.create(
        make_handoff(now=base + timedelta(seconds=1), handoff_id=second_id),
        now=base + timedelta(seconds=1),
    )
    store.consume(first.handoff_id, now=base + timedelta(seconds=2))
    store.create(
        make_handoff(now=base + timedelta(seconds=3), handoff_id=third_id),
        now=base + timedelta(seconds=3),
    )

    report = store.verify_audit_integrity()

    assert report.is_valid is True
    assert report.checked_handoffs == 3
    assert report.verified_handoffs == 3
    assert report.untracked_handoffs == 0
    assert report.issues == ()


def test_orphan_audit_event_is_detected(tmp_path):
    path = tmp_path / "orphan.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """INSERT INTO cognitive_pass_handoff_events (
                event_id, handoff_id, event_type, from_status, to_status,
                occurred_at, metadata_json
            ) VALUES (?, ?, 'created', NULL, 'prepared', ?, '{}')""",
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
             "99999999-9999-4999-8999-999999999999", base.isoformat()),
        )

    report = store.verify_audit_integrity()
    assert report.is_valid is False
    assert "orphan_event" in issue_codes(report)


def test_malformed_event_metadata_is_detected_without_crashing(tmp_path):
    path = tmp_path / "malformed_event.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER IF EXISTS trg_cognitive_pass_handoff_events_no_update")
        connection.execute(
            "UPDATE cognitive_pass_handoff_events SET metadata_json = '{broken' WHERE handoff_id = ?",
            (created.handoff_id,),
        )

    report = store.verify_audit_integrity()
    assert report.is_valid is False
    assert "malformed_event" in issue_codes(report)


def test_duplicate_terminal_events_are_detected(tmp_path):
    path = tmp_path / "duplicate_terminal.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)
    consumed_at = base + timedelta(seconds=5)
    store.consume(created.handoff_id, now=consumed_at)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO cognitive_pass_handoff_events (
                event_id, handoff_id, event_type, from_status, to_status,
                occurred_at, metadata_json
            ) VALUES (?, ?, 'consumed', 'prepared', 'consumed', ?, '{}')""",
            ("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
             created.handoff_id, consumed_at.isoformat()),
        )

    report = store.verify_audit_integrity()
    assert report.is_valid is False
    assert "duplicate_terminal_event" in issue_codes(report)


def test_terminal_event_on_prepared_handoff_is_detected(tmp_path):
    path = tmp_path / "terminal_on_prepared.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO cognitive_pass_handoff_events (
                event_id, handoff_id, event_type, from_status, to_status,
                occurred_at, metadata_json
            ) VALUES (?, ?, 'consumed', 'prepared', 'consumed', ?, '{}')""",
            ("cccccccc-cccc-4ccc-8ccc-cccccccccccc",
             created.handoff_id, (base + timedelta(seconds=1)).isoformat()),
        )

    report = store.verify_audit_integrity()
    assert report.is_valid is False
    assert "terminal_event_status_mismatch" in issue_codes(report)


def test_verification_report_is_stable_after_store_reopen(tmp_path):
    path = tmp_path / "reopen_stable.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first_store = SQLiteCognitivePassHandoffStore(path)
    created = first_store.create(make_handoff(now=base), now=base)
    first_store.consume(created.handoff_id, now=base + timedelta(seconds=4))
    first_report = first_store.verify_audit_integrity().to_dict()

    reopened_store = SQLiteCognitivePassHandoffStore(path)
    second_report = reopened_store.verify_audit_integrity().to_dict()

    assert second_report == first_report
    assert second_report["is_valid"] is True
    assert second_report["verified_handoffs"] == 1


def test_verifier_does_not_change_database_schema_or_journal_mode(tmp_path):
    path = tmp_path / "schema_read_only.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        schema_before = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        journal_before = connection.execute("PRAGMA journal_mode").fetchone()[0]

    store.verify_audit_integrity()

    with sqlite3.connect(path) as connection:
        schema_after = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        journal_after = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert schema_after == schema_before
    assert journal_after == journal_before


def test_invalid_event_type_is_reported_as_malformed_event(tmp_path):
    path = tmp_path / "invalid_event_type.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)

    with sqlite3.connect(path) as connection:
        # Deliberately bypass the CHECK constraint to simulate on-disk corruption.
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("DROP TRIGGER IF EXISTS trg_cognitive_pass_handoff_events_no_update")
        connection.execute(
            "UPDATE cognitive_pass_handoff_events SET event_type = 'unknown' WHERE handoff_id = ?",
            (created.handoff_id,),
        )

    report = store.verify_audit_integrity()
    assert report.is_valid is False
    assert "malformed_event" in issue_codes(report)
