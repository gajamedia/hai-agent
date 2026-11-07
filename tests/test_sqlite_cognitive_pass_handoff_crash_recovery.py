
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
from datetime import datetime, timedelta, timezone


from agent.llm.cognitive_pass_handoff import (
    CognitivePassHandoff,
    CognitivePassHandoffValidationError,
)
from agent.llm.sqlite_cognitive_pass_handoff_store import (
    CognitivePassHandoffAlreadyConsumedError,
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


def _crash_before_commit_worker(database_path: str, handoff_id: str):
    """
    Simulate abrupt process termination after UPDATE but before COMMIT.

    os._exit() intentionally bypasses Python cleanup and connection close.
    SQLite must recover the uncommitted transaction on the next open.
    """
    connection = sqlite3.connect(
        database_path,
        timeout=20.0,
        isolation_level=None,
    )

    connection.execute("PRAGMA busy_timeout = 20000")
    connection.execute("PRAGMA synchronous = FULL")

    now_text = datetime.now(timezone.utc).isoformat()

    connection.execute("BEGIN IMMEDIATE")

    connection.execute(
        """
        UPDATE cognitive_pass_handoffs
        SET status = 'consumed',
            consumed_at = ?,
            updated_at = ?
        WHERE handoff_id = ?
          AND status = 'prepared'
        """,
        (now_text, now_text, handoff_id),
    )

    # Intentionally crash with an open, uncommitted transaction.
    os._exit(23)


def _consume_then_crash_worker(database_path: str, handoff_id: str):
    """
    Complete the production consume operation, then terminate immediately.

    The parent process verifies that the committed state survives restart.
    """
    store = SQLiteCognitivePassHandoffStore(
        database_path,
        timeout=20.0,
    )

    result = store.consume(
        handoff_id,
        expected_goal="open the box",
    )

    if result.status != "consumed":
        os._exit(24)

    # The store's consume() has already committed and closed its connection.
    os._exit(0)


def _run_crashing_process(target, *args, expected_exitcode):
    context = multiprocessing.get_context("spawn")

    process = context.Process(
        target=target,
        args=args,
    )

    process.start()
    process.join(timeout=30.0)

    if process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
        raise AssertionError("Crash simulation process timed out")

    assert process.exitcode == expected_exitcode


def test_uncommitted_consume_is_rolled_back_after_process_crash(tmp_path):
    database_path = tmp_path / "crash_before_commit.sqlite3"

    store = SQLiteCognitivePassHandoffStore(database_path)
    created = store.create(make_handoff())

    _run_crashing_process(
        _crash_before_commit_worker,
        str(database_path),
        created.handoff_id,
        expected_exitcode=23,
    )

    # Reopen the database after abrupt process termination.
    reopened = SQLiteCognitivePassHandoffStore(database_path)

    recovered = reopened.get(created.handoff_id)

    assert recovered is not None
    assert recovered.status == "prepared"
    assert recovered.consumed_at is None

    # The recovered handoff must still be usable exactly once.
    consumed = reopened.consume(
        created.handoff_id,
        expected_goal="open the box",
    )

    assert consumed.status == "consumed"
    assert consumed.consumed_at is not None
    assert consumed.updated_at == consumed.consumed_at


def test_committed_consume_survives_immediate_process_crash(tmp_path):
    database_path = tmp_path / "crash_after_commit.sqlite3"

    store = SQLiteCognitivePassHandoffStore(database_path)
    created = store.create(make_handoff())

    _run_crashing_process(
        _consume_then_crash_worker,
        str(database_path),
        created.handoff_id,
        expected_exitcode=0,
    )

    # Simulate application/database reopening after the worker has exited.
    reopened = SQLiteCognitivePassHandoffStore(database_path)

    recovered = reopened.get(created.handoff_id)

    assert recovered is not None
    assert recovered.status == "consumed"
    assert recovered.consumed_at is not None
    assert recovered.updated_at == recovered.consumed_at

    # A committed consume must never become available again.
    try:
        reopened.consume(
            created.handoff_id,
            expected_goal="open the box",
        )
    except CognitivePassHandoffAlreadyConsumedError:
        pass
    else:
        raise AssertionError(
            "A committed handoff must not be consumed a second time"
        )


def test_validation_failure_rolls_back_without_partial_state_change(tmp_path):
    database_path = tmp_path / "validation_rollback.sqlite3"

    store = SQLiteCognitivePassHandoffStore(database_path)
    created = store.create(make_handoff())

    try:
        store.consume(
            created.handoff_id,
            expected_goal="close the box",
        )
    except CognitivePassHandoffValidationError:
        pass
    else:
        raise AssertionError(
            "Mismatched goal should fail handoff validation"
        )

    reopened = SQLiteCognitivePassHandoffStore(database_path)
    recovered = reopened.get(created.handoff_id)

    assert recovered is not None
    assert recovered.status == "prepared"
    assert recovered.consumed_at is None

    assert reopened.count("prepared") == 1
    assert reopened.count("consumed") == 0
    assert reopened.count("expired") == 0
