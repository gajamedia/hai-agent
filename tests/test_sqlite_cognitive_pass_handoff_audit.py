from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agent.llm.cognitive_pass_handoff import (
    CognitivePassHandoff,
    CognitivePassHandoffValidationError,
)
from agent.llm.sqlite_cognitive_pass_handoff_store import (
    CognitivePassHandoffExpiredError,
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


def test_create_and_consume_write_atomic_ordered_audit_events(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "audit.sqlite3")
    created = store.create(make_handoff(now=base), now=base)
    consumed = store.consume(created.handoff_id, expected_goal="open the box",
                             now=base + timedelta(seconds=5))
    events = store.list_events(handoff_id=created.handoff_id)
    assert consumed.status == "consumed"
    assert [e.event_type for e in events] == ["created", "consumed"]
    assert [(e.from_status, e.to_status, e.metadata) for e in events] == [
        (None, "prepared", {}), ("prepared", "consumed", {})
    ]
    assert events[0].occurred_at == base.isoformat()
    assert events[1].occurred_at == (base + timedelta(seconds=5)).isoformat()
    assert events[0].event_id != events[1].event_id


def test_expire_due_writes_one_expired_event_and_is_idempotent(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "expire.sqlite3")
    created = store.create(make_handoff(now=base), now=base)
    at_expiry = base + timedelta(seconds=30)
    assert store.expire_due(now=at_expiry) == 1
    assert store.expire_due(now=at_expiry) == 0
    events = store.list_events(handoff_id=created.handoff_id)
    assert [e.event_type for e in events] == ["created", "expired"]
    assert (events[1].from_status, events[1].to_status) == ("prepared", "expired")


def test_expired_consume_writes_expiration_event_before_raising(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "consume_expired.sqlite3")
    created = store.create(make_handoff(now=base), now=base)
    with pytest.raises(CognitivePassHandoffExpiredError):
        store.consume(created.handoff_id, now=base + timedelta(seconds=30))
    assert [e.event_type for e in store.list_events(handoff_id=created.handoff_id)] == [
        "created", "expired"
    ]


def test_failed_context_validation_does_not_write_consumed_event(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "failed_validation.sqlite3")
    created = store.create(make_handoff(now=base), now=base)
    with pytest.raises(CognitivePassHandoffValidationError):
        store.consume(created.handoff_id, expected_goal="close the box",
                      now=base + timedelta(seconds=1))
    assert [e.event_type for e in store.list_events(handoff_id=created.handoff_id)] == ["created"]
    assert store.get(created.handoff_id).status == "prepared"


def test_audit_events_are_append_only_against_update_and_delete(tmp_path):
    path = tmp_path / "append_only.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    created = store.create(make_handoff(now=base), now=base)
    event = store.list_events(handoff_id=created.handoff_id)[0]
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE cognitive_pass_handoff_events SET metadata_json='{}' WHERE event_id=?",
                (event.event_id,))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM cognitive_pass_handoff_events WHERE event_id=?",
                (event.event_id,))
    assert len(store.list_events(handoff_id=created.handoff_id)) == 1


@pytest.mark.parametrize("operation", ["create", "consume", "expire"])
def test_event_insert_failure_rolls_back_lifecycle_change(tmp_path, operation):
    path = tmp_path / f"atomic_{operation}.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    handoff = make_handoff(now=base)
    if operation != "create":
        created = store.create(handoff, now=base)
        target_id = created.handoff_id
    else:
        target_id = handoff.handoff_id
    # Ensure the event table exists, then inject a failure on event insertion.
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TRIGGER reject_audit_insert
            BEFORE INSERT ON cognitive_pass_handoff_events
            BEGIN SELECT RAISE(ABORT, 'injected audit insert failure'); END
        """)
    if operation == "create":
        with pytest.raises(sqlite3.IntegrityError, match="injected audit"):
            store.create(handoff, now=base)
        assert store.get(target_id) is None
        assert store.list_events(handoff_id=target_id) == ()
    elif operation == "consume":
        with pytest.raises(sqlite3.IntegrityError, match="injected audit"):
            store.consume(target_id, now=base + timedelta(seconds=1))
        assert store.get(target_id).status == "prepared"
        assert [e.event_type for e in store.list_events(handoff_id=target_id)] == ["created"]
    else:
        with pytest.raises(sqlite3.IntegrityError, match="injected audit"):
            store.expire_due(now=base + timedelta(seconds=30))
        assert store.get(target_id).status == "prepared"
        assert [e.event_type for e in store.list_events(handoff_id=target_id)] == ["created"]


def test_list_events_supports_handoff_and_event_type_filters(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "filters.sqlite3")
    first = store.create(make_handoff(now=base), now=base)
    second = store.create(make_handoff(
        now=base, handoff_id="22345678-1234-5678-1234-567812345678"), now=base)
    store.consume(first.handoff_id, now=base + timedelta(seconds=1))
    assert len(store.list_events()) == 3
    assert len(store.list_events(handoff_id=first.handoff_id)) == 2
    assert len(store.list_events(event_type="created")) == 2
    assert len(store.list_events(handoff_id=second.handoff_id, event_type="created")) == 1


def test_list_events_rejects_invalid_filters(tmp_path):
    store = SQLiteCognitivePassHandoffStore(tmp_path / "invalid_filters.sqlite3")
    with pytest.raises(ValueError, match="handoff_id"):
        store.list_events(handoff_id=" ")
    with pytest.raises(ValueError, match="event_type"):
        store.list_events(event_type="deleted")


def test_legacy_rows_are_not_given_fabricated_creation_events(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff(now=base), now=base)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER IF EXISTS trg_cognitive_pass_handoff_events_no_update")
        connection.execute("DROP TRIGGER IF EXISTS trg_cognitive_pass_handoff_events_no_delete")
        connection.execute("DROP TABLE IF EXISTS cognitive_pass_handoff_events")
    migrated = SQLiteCognitivePassHandoffStore(path)
    assert migrated.get(created.handoff_id).status == "prepared"
    assert migrated.list_events(handoff_id=created.handoff_id) == ()


def test_retention_policy_is_non_destructive_and_has_no_delete_api(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = SQLiteCognitivePassHandoffStore(tmp_path / "retention.sqlite3")
    created = store.create(make_handoff(now=base), now=base)
    store.expire_due(now=base + timedelta(seconds=30))
    assert not hasattr(store, "purge_events")
    assert not hasattr(store, "delete_events")
    assert len(store.list_events(handoff_id=created.handoff_id)) == 2
