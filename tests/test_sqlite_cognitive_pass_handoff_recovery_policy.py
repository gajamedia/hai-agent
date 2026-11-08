
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from agent.llm.cognitive_pass_handoff import CognitivePassHandoff
from agent.llm.sqlite_cognitive_pass_handoff_store import (
    SQLiteCognitivePassHandoffStore,
)


def make_handoff(
    *,
    now: datetime,
    ttl_seconds: int = 30,
    handoff_id: str = "12345678-1234-5678-1234-567812345678",
) -> CognitivePassHandoff:
    expires = now + timedelta(seconds=ttl_seconds)

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

    plan_json = json.dumps(
        plan,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return CognitivePassHandoff.from_dict(
        {
            "handoff_id": handoff_id,
            "contract_version": 1,
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "cognitive_cycle": 1,
            "goal": "open the box",
            "observation_fingerprint": hashlib.sha256(
                b'{"box":"closed"}'
            ).hexdigest(),
            "plan_fingerprint": hashlib.sha256(
                plan_json.encode("utf-8")
            ).hexdigest(),
            "plan": plan,
        }
    )


def test_recovery_is_idempotent_after_store_restart(tmp_path):
    database_path = tmp_path / "recovery_restart.sqlite3"

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recovery_time = base + timedelta(seconds=30)

    first_store = SQLiteCognitivePassHandoffStore(database_path)

    created = first_store.create(
        make_handoff(now=base),
        now=base,
    )

    first_count = first_store.expire_due(now=recovery_time)

    first_record = first_store.get(created.handoff_id)

    assert first_count == 1
    assert first_record is not None
    assert first_record.status == "expired"

    # Simulate application restart by creating a new store instance.
    restarted_store = SQLiteCognitivePassHandoffStore(database_path)

    second_count = restarted_store.expire_due(now=recovery_time)

    second_record = restarted_store.get(created.handoff_id)

    assert second_count == 0
    assert second_record is not None
    assert second_record.status == "expired"

    assert second_record.created_at == first_record.created_at
    assert second_record.updated_at == first_record.updated_at
    assert second_record.consumed_at is None


def test_concurrent_recovery_expires_each_record_only_once(tmp_path):
    database_path = tmp_path / "recovery_concurrent.sqlite3"

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recovery_time = base + timedelta(seconds=30)

    initial_store = SQLiteCognitivePassHandoffStore(database_path)

    created = initial_store.create(
        make_handoff(now=base),
        now=base,
    )

    def run_recovery(_):
        store = SQLiteCognitivePassHandoffStore(
            database_path,
            timeout=15.0,
        )
        return store.expire_due(now=recovery_time)

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(run_recovery, range(8)))

    assert sum(outcomes) == 1
    assert outcomes.count(1) == 1
    assert outcomes.count(0) == 7

    final_store = SQLiteCognitivePassHandoffStore(database_path)
    persisted = final_store.get(created.handoff_id)

    assert persisted is not None
    assert persisted.status == "expired"
    assert persisted.updated_at == recovery_time.isoformat()


def test_recovery_preserves_consumed_and_unexpired_records(tmp_path):
    database_path = tmp_path / "recovery_preservation.sqlite3"

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recovery_time = base + timedelta(seconds=30)

    store = SQLiteCognitivePassHandoffStore(database_path)

    consumed = store.create(
        make_handoff(
            now=base,
            handoff_id="22345678-1234-5678-1234-567812345678",
        ),
        now=base,
    )

    due = store.create(
        make_handoff(
            now=base,
            handoff_id="32345678-1234-5678-1234-567812345678",
        ),
        now=base,
    )

    future = store.create(
        make_handoff(
            now=base,
            ttl_seconds=120,
            handoff_id="42345678-1234-5678-1234-567812345678",
        ),
        now=base,
    )

    consumed_result = store.consume(
        consumed.handoff_id,
        now=base + timedelta(seconds=10),
    )

    changed = store.expire_due(now=recovery_time)

    assert changed == 1

    consumed_after = store.get(consumed.handoff_id)
    due_after = store.get(due.handoff_id)
    future_after = store.get(future.handoff_id)

    assert consumed_after is not None
    assert consumed_after.status == "consumed"
    assert consumed_after.consumed_at == consumed_result.consumed_at

    assert due_after is not None
    assert due_after.status == "expired"

    assert future_after is not None
    assert future_after.status == "prepared"


def test_recovery_does_not_delete_terminal_audit_records(tmp_path):
    database_path = tmp_path / "recovery_audit.sqlite3"

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recovery_time = base + timedelta(seconds=30)

    store = SQLiteCognitivePassHandoffStore(database_path)

    created = store.create(
        make_handoff(now=base),
        now=base,
    )

    store.expire_due(now=recovery_time)

    assert store.count() == 1
    assert store.count("prepared") == 0
    assert store.count("expired") == 1

    reopened = SQLiteCognitivePassHandoffStore(database_path)

    records = reopened.list_all()

    assert len(records) == 1
    assert records[0].handoff_id == created.handoff_id
    assert records[0].status == "expired"
