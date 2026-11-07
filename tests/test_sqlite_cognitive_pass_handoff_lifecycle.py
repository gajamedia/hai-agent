
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agent.llm.cognitive_pass_handoff import CognitivePassHandoff
from agent.llm.sqlite_cognitive_pass_handoff_store import (
    CognitivePassHandoffAlreadyConsumedError,
    CognitivePassHandoffExpiredError,
    SQLiteCognitivePassHandoffStore,
)


def make_handoff(
    *,
    now=None,
    ttl_seconds=300,
    handoff_id="12345678-1234-5678-1234-567812345678",
):
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
            "created_at": created.isoformat(),
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


def force_status(database_path, handoff_id, status, timestamp):
    """Test-only helper to establish a lifecycle state."""
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            UPDATE cognitive_pass_handoffs
            SET status = ?,
                consumed_at = ?,
                updated_at = ?
            WHERE handoff_id = ?
            """,
            (status, timestamp, timestamp, handoff_id),
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("initial_status", "target_status"),
    [
        ("consumed", "prepared"),
        ("expired", "prepared"),
        ("consumed", "expired"),
        ("expired", "consumed"),
    ],
)
def test_database_rejects_illegal_terminal_transitions(
    tmp_path,
    initial_status,
    target_status,
):
    database_path = tmp_path / "lifecycle_guard.sqlite3"
    store = SQLiteCognitivePassHandoffStore(database_path)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    created = store.create(
        make_handoff(now=base),
        now=base,
    )

    if initial_status == "consumed":
        store.consume(
            created.handoff_id,
            expected_goal="open the box",
            now=base + timedelta(seconds=1),
        )
    else:
        with pytest.raises(CognitivePassHandoffExpiredError):
            store.consume(
                created.handoff_id,
                now=base + timedelta(seconds=300),
            )

    before = store.get(created.handoff_id)
    assert before is not None
    assert before.status == initial_status

    connection = sqlite3.connect(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE cognitive_pass_handoffs
                SET status = ?
                WHERE handoff_id = ?
                """,
                (target_status, created.handoff_id),
            )
            connection.commit()
    finally:
        connection.close()

    after = store.get(created.handoff_id)
    assert after is not None
    assert after.status == initial_status


def test_expire_due_is_idempotent(tmp_path):
    database_path = tmp_path / "idempotent_expiration.sqlite3"
    store = SQLiteCognitivePassHandoffStore(database_path)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    created = store.create(
        make_handoff(now=base, ttl_seconds=30),
        now=base,
    )

    expiration_time = base + timedelta(seconds=30)

    first_count = store.expire_due(now=expiration_time)
    first_record = store.get(created.handoff_id)

    second_count = store.expire_due(now=expiration_time)
    second_record = store.get(created.handoff_id)

    assert first_count == 1
    assert second_count == 0

    assert first_record is not None
    assert second_record is not None

    assert first_record.status == "expired"
    assert second_record.status == "expired"

    assert first_record.updated_at == second_record.updated_at
    assert second_record.consumed_at is None


def test_expiration_recovery_never_changes_consumed_handoff(tmp_path):
    database_path = tmp_path / "consumed_terminal.sqlite3"
    store = SQLiteCognitivePassHandoffStore(database_path)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    created = store.create(
        make_handoff(now=base, ttl_seconds=30),
        now=base,
    )

    consumed = store.consume(
        created.handoff_id,
        expected_goal="open the box",
        now=base + timedelta(seconds=10),
    )

    expiration_count = store.expire_due(
        now=base + timedelta(seconds=60),
    )

    persisted = store.get(created.handoff_id)

    assert expiration_count == 0
    assert persisted is not None
    assert persisted.status == "consumed"
    assert persisted.consumed_at == consumed.consumed_at
    assert persisted.updated_at == consumed.updated_at


def test_terminal_handoff_cannot_be_consumed_again(tmp_path):
    database_path = tmp_path / "terminal_consume.sqlite3"
    store = SQLiteCognitivePassHandoffStore(database_path)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    created = store.create(
        make_handoff(now=base),
        now=base,
    )

    store.consume(
        created.handoff_id,
        expected_goal="open the box",
        now=base + timedelta(seconds=1),
    )

    with pytest.raises(CognitivePassHandoffAlreadyConsumedError):
        store.consume(
            created.handoff_id,
            expected_goal="open the box",
            now=base + timedelta(seconds=2),
        )

    persisted = store.get(created.handoff_id)

    assert persisted is not None
    assert persisted.status == "consumed"
    assert persisted.consumed_at is not None
