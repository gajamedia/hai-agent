from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from agent.llm.cognitive_pass_handoff import (
    CognitivePassHandoff,
    CognitivePassHandoffValidationError,
)
from agent.llm.sqlite_cognitive_pass_handoff_store import (
    CognitivePassHandoffAlreadyConsumedError,
    CognitivePassHandoffAlreadyExistsError,
    CognitivePassHandoffExpiredError,
    CognitivePassHandoffNotFoundError,
    CognitivePassHandoffStateError,
    SQLiteCognitivePassHandoffStore,
)


def make_handoff(*, now=None, ttl_seconds=300, handoff_id="12345678-1234-5678-1234-567812345678"):
    created = now or datetime.now(timezone.utc)
    expires = created + timedelta(seconds=ttl_seconds)
    plan = {
        "goal": "open the box",
        "rationale": "The box must be opened first.",
        "steps": [
            {
                "index": 1,
                "action": "open_box",
                "purpose": "Open the box",
                "preconditions": [],
                "expected_observation": "box is open",
                "success_criteria": "box is open",
                "risk": "low",
            }
        ],
        "assumptions": [],
        "constraints": [],
        "stop_conditions": [],
        "confidence": 0.8,
        "provider": None,
        "model": None,
    }
    plan_json = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return CognitivePassHandoff.from_dict(
        {
            "handoff_id": handoff_id,
            "contract_version": 1,
            "created_at": created.isoformat(),
            "expires_at": expires.isoformat(),
            "cognitive_cycle": 1,
            "goal": "open the box",
            "observation_fingerprint": hashlib.sha256(b'{"box":"closed"}').hexdigest(),
            "plan_fingerprint": hashlib.sha256(plan_json.encode("utf-8")).hexdigest(),
            "plan": plan,
        }
    )


def test_create_and_get_persist_across_store_instances(tmp_path):
    path = tmp_path / "handoffs.sqlite3"
    first = SQLiteCognitivePassHandoffStore(path)
    created = first.create(make_handoff())

    assert created.status == "prepared"
    assert created.consumed_at is None
    assert first.count() == 1

    reopened = SQLiteCognitivePassHandoffStore(path)
    loaded = reopened.get(created.handoff_id)
    assert loaded is not None
    assert loaded.handoff == created.handoff
    assert loaded.status == "prepared"
    assert loaded.handoff.plan["steps"][0]["action"] == "open_box"


def test_duplicate_handoff_id_is_rejected(tmp_path):
    store = SQLiteCognitivePassHandoffStore(tmp_path / "handoffs.sqlite3")
    handoff = make_handoff()
    store.create(handoff)
    with pytest.raises(CognitivePassHandoffAlreadyExistsError):
        store.create(handoff)
    assert store.count("prepared") == 1


def test_consume_is_single_use_and_auditable_after_restart(tmp_path):
    path = tmp_path / "handoffs.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff())

    consumed = store.consume(created.handoff_id, expected_goal="open the box")
    assert consumed.status == "consumed"
    assert consumed.consumed_at is not None
    assert consumed.updated_at == consumed.consumed_at

    reopened = SQLiteCognitivePassHandoffStore(path)
    persisted = reopened.get(created.handoff_id)
    assert persisted is not None
    assert persisted.status == "consumed"
    assert persisted.consumed_at == consumed.consumed_at
    with pytest.raises(CognitivePassHandoffAlreadyConsumedError):
        reopened.consume(created.handoff_id)


def test_expired_consume_persists_expired_status_before_raising(tmp_path):
    store = SQLiteCognitivePassHandoffStore(tmp_path / "handoffs.sqlite3")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    created = store.create(make_handoff(now=created_at, ttl_seconds=30), now=created_at)

    with pytest.raises(CognitivePassHandoffExpiredError):
        store.consume(
            created.handoff_id,
            now=created_at + timedelta(seconds=30),
        )

    persisted = store.get(created.handoff_id)
    assert persisted is not None
    assert persisted.status == "expired"
    assert persisted.consumed_at is None


def test_expire_due_updates_only_prepared_due_records(tmp_path):
    store = SQLiteCognitivePassHandoffStore(tmp_path / "handoffs.sqlite3")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    due = store.create(make_handoff(now=base, ttl_seconds=10), now=base)
    future = store.create(
        make_handoff(
            now=base,
            ttl_seconds=100,
            handoff_id="22345678-1234-5678-1234-567812345678",
        ),
        now=base,
    )

    changed = store.expire_due(now=base + timedelta(seconds=10))
    assert changed == 1
    assert store.get(due.handoff_id).status == "expired"
    assert store.get(future.handoff_id).status == "prepared"


def test_context_mismatch_does_not_consume_handoff(tmp_path):
    store = SQLiteCognitivePassHandoffStore(tmp_path / "handoffs.sqlite3")
    created = store.create(make_handoff())

    with pytest.raises(CognitivePassHandoffValidationError, match="goal does not match"):
        store.consume(created.handoff_id, expected_goal="close the box")

    persisted = store.get(created.handoff_id)
    assert persisted is not None
    assert persisted.status == "prepared"


def test_missing_handoff_raises_not_found(tmp_path):
    store = SQLiteCognitivePassHandoffStore(tmp_path / "handoffs.sqlite3")
    with pytest.raises(CognitivePassHandoffNotFoundError):
        store.consume("12345678-1234-5678-1234-567812345678")


def test_concurrent_consumption_only_one_request_succeeds(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    created = SQLiteCognitivePassHandoffStore(path).create(make_handoff())

    def consume_once(_):
        store = SQLiteCognitivePassHandoffStore(path, timeout=15.0)
        try:
            return store.consume(created.handoff_id).status
        except CognitivePassHandoffAlreadyConsumedError:
            return "already_consumed"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(consume_once, range(8)))

    assert outcomes.count("consumed") == 1
    assert outcomes.count("already_consumed") == 7
    assert SQLiteCognitivePassHandoffStore(path).count("consumed") == 1


def test_concurrent_store_initialization_and_distinct_creates(tmp_path):
    path = tmp_path / "parallel" / "handoffs.sqlite3"

    def create_one(index):
        store = SQLiteCognitivePassHandoffStore(path, timeout=15.0)
        handoff_id = f"{index:08d}-1234-5678-1234-567812345678"
        return store.create(make_handoff(handoff_id=handoff_id)).handoff_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(create_one, range(1, 9)))

    assert len(set(outcomes)) == 8
    assert SQLiteCognitivePassHandoffStore(path).count("prepared") == 8


def test_persisted_index_payload_mismatch_fails_closed(tmp_path):
    path = tmp_path / "tampered.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff())

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cognitive_pass_handoffs SET expires_at=? WHERE handoff_id=?",
            ("2030-01-01T00:00:00+00:00", created.handoff_id),
        )

    with pytest.raises(CognitivePassHandoffStateError, match="index fields"):
        store.get(created.handoff_id)


def test_list_and_count_can_filter_by_status(tmp_path):
    store = SQLiteCognitivePassHandoffStore(tmp_path / "handoffs.sqlite3")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prepared = store.create(make_handoff(now=base), now=base)
    consumed = store.create(
        make_handoff(now=base, handoff_id="32345678-1234-5678-1234-567812345678"),
        now=base,
    )
    store.consume(consumed.handoff_id, now=base + timedelta(seconds=1))

    assert store.count() == 2
    assert store.count("prepared") == 1
    assert store.count("consumed") == 1
    assert [item.handoff_id for item in store.list_all("prepared")] == [prepared.handoff_id]


def test_invalid_database_arguments_are_rejected():
    with pytest.raises(ValueError):
        SQLiteCognitivePassHandoffStore(None)
    with pytest.raises(ValueError):
        SQLiteCognitivePassHandoffStore(":memory:")
    with pytest.raises(ValueError):
        SQLiteCognitivePassHandoffStore("handoffs.sqlite3", timeout=0)



def test_prepared_record_rejects_consumed_timestamp_tampering(tmp_path):
    path = tmp_path / "prepared-tampered.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff())

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cognitive_pass_handoffs SET consumed_at=? WHERE handoff_id=?",
            (datetime.now(timezone.utc).isoformat(), created.handoff_id),
        )

    with pytest.raises(CognitivePassHandoffStateError, match="Prepared handoff"):
        store.get(created.handoff_id)


def test_consumed_record_requires_consistent_audit_timestamps(tmp_path):
    path = tmp_path / "consumed-tampered.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff())
    consumed = store.consume(created.handoff_id)

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cognitive_pass_handoffs SET consumed_at=NULL WHERE handoff_id=?",
            (created.handoff_id,),
        )

    with pytest.raises(CognitivePassHandoffStateError, match="must have consumed_at"):
        store.get(consumed.handoff_id)


def test_expired_record_requires_updated_at_at_or_after_expiration(tmp_path):
    path = tmp_path / "expired-tampered.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    created = store.create(make_handoff(now=base, ttl_seconds=30), now=base)
    store.expire_due(now=base + timedelta(seconds=30))

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cognitive_pass_handoffs SET updated_at=? WHERE handoff_id=?",
            (base.isoformat(), created.handoff_id),
        )

    with pytest.raises(CognitivePassHandoffStateError, match="precedes expires_at"):
        store.get(created.handoff_id)


def test_invalid_persisted_updated_timestamp_fails_closed(tmp_path):
    path = tmp_path / "invalid-updated.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff())

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cognitive_pass_handoffs SET updated_at=? WHERE handoff_id=?",
            ("not-a-timestamp", created.handoff_id),
        )

    with pytest.raises(CognitivePassHandoffStateError, match="malformed"):
        store.get(created.handoff_id)


def test_consumed_timestamp_must_match_updated_timestamp(tmp_path):
    path = tmp_path / "consumed-mismatch.sqlite3"
    store = SQLiteCognitivePassHandoffStore(path)
    created = store.create(make_handoff())
    store.consume(created.handoff_id)

    with sqlite3.connect(path) as connection:
        consumed_at = connection.execute(
            "SELECT consumed_at FROM cognitive_pass_handoffs WHERE handoff_id=?",
            (created.handoff_id,),
        ).fetchone()[0]
        mismatched_updated_at = (
            datetime.fromisoformat(consumed_at) - timedelta(seconds=1)
        ).isoformat()
        connection.execute(
            "UPDATE cognitive_pass_handoffs SET updated_at=? WHERE handoff_id=?",
            (mismatched_updated_at, created.handoff_id),
        )

    with pytest.raises(CognitivePassHandoffStateError, match="updated_at must equal"):
        store.get(created.handoff_id)
