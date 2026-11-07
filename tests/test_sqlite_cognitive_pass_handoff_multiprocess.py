
from __future__ import annotations

import hashlib
import json
import multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.llm.cognitive_pass_handoff import CognitivePassHandoff
from agent.llm.sqlite_cognitive_pass_handoff_store import (
    CognitivePassHandoffAlreadyConsumedError,
    SQLiteCognitivePassHandoffStore,
)


def make_handoff():
    created = datetime.now(timezone.utc)
    expires = created + timedelta(seconds=300)

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
            "handoff_id": "12345678-1234-5678-1234-567812345678",
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


def consume_worker(
    database_path,
    handoff_id,
    ready_event,
    start_event,
    result_queue,
):
    """
    Worker berjalan di proses OS terpisah.
    Setiap proses membuat instance store dan koneksi SQLite sendiri.
    """
    try:
        store = SQLiteCognitivePassHandoffStore(
            database_path,
            timeout=20.0,
        )

        ready_event.set()

        if not start_event.wait(timeout=30):
            result_queue.put("worker_start_timeout")
            return

        try:
            result = store.consume(handoff_id)
            result_queue.put(result.status)

        except CognitivePassHandoffAlreadyConsumedError:
            result_queue.put("already_consumed")

    except BaseException as exc:
        ready_event.set()
        result_queue.put(
            f"worker_error:{type(exc).__name__}:{exc}"
        )


def test_concurrent_consumption_across_processes_is_single_use(tmp_path):
    database_path = tmp_path / "multiprocess.sqlite3"

    # Persiapkan satu handoff.
    initial_store = SQLiteCognitivePassHandoffStore(database_path)
    created = initial_store.create(make_handoff())

    process_count = 6

    # Gunakan spawn agar setiap worker benar-benar proses terpisah.
    context = multiprocessing.get_context("spawn")

    start_event = context.Event()
    result_queue = context.Queue()

    ready_events = [
        context.Event()
        for _ in range(process_count)
    ]

    processes = []

    for index in range(process_count):
        process = context.Process(
            target=consume_worker,
            args=(
                str(database_path),
                created.handoff_id,
                ready_events[index],
                start_event,
                result_queue,
            ),
        )

        processes.append(process)

    try:
        # Jalankan semua proses.
        for process in processes:
            process.start()

        # Pastikan seluruh proses sudah membuka store.
        for index, ready_event in enumerate(ready_events):
            assert ready_event.wait(timeout=30), (
                f"Worker {index} gagal melakukan inisialisasi"
            )

        # Lepaskan semua proses secara bersamaan.
        start_event.set()

        outcomes = [
            result_queue.get(timeout=35)
            for _ in range(process_count)
        ]

        # Tunggu seluruh proses selesai.
        for process in processes:
            process.join(timeout=10)

        # Semua proses harus selesai secara normal.
        assert all(
            process.exitcode == 0
            for process in processes
        )

        # Tepat satu proses berhasil.
        assert outcomes.count("consumed") == 1

        # Lima proses lainnya ditolak.
        assert outcomes.count("already_consumed") == (
            process_count - 1
        )

        # Tidak boleh ada error worker.
        assert not any(
            outcome.startswith("worker_error:")
            or outcome == "worker_start_timeout"
            for outcome in outcomes
        )

        # Periksa status final setelah semua proses selesai.
        final_store = SQLiteCognitivePassHandoffStore(
            database_path
        )

        persisted = final_store.get(created.handoff_id)

        assert persisted is not None
        assert persisted.status == "consumed"
        assert persisted.consumed_at is not None
        assert persisted.updated_at == persisted.consumed_at

        assert final_store.count("consumed") == 1
        assert final_store.count("prepared") == 0
        assert final_store.count("expired") == 0

    finally:
        # Bersihkan proses jika terjadi kegagalan test.
        start_event.set()

        for process in processes:
            if process.is_alive():
                process.terminate()

            process.join(timeout=5)

        result_queue.close()
        result_queue.join_thread()
