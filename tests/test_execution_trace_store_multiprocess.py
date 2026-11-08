from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from agent.llm.execution_trace_store import ExecutionTraceStore


def _append_unique_events(args):
    path, worker_id, count = args
    store = ExecutionTraceStore(path, fsync=False)
    for index in range(count):
        store.append({
            "event_id": f"worker-{worker_id}-event-{index}",
            "execution_id": f"worker-{worker_id}",
            "event_type": "completed",
            "payload": {"worker": worker_id, "index": index},
        })
    return count


def _append_same_event(path):
    store = ExecutionTraceStore(path, fsync=False)
    return store.append({
        "event_id": "shared-idempotent-event",
        "execution_id": "shared-execution",
        "event_type": "started",
        "payload": {"source": "concurrent-test"},
    })


def test_multiple_processes_preserve_sequence_and_hash_chain(tmp_path):
    path = str(tmp_path / "multiprocess-trace.jsonl")
    workers = [(path, worker_id, 8) for worker_id in range(4)]

    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=get_context("spawn"),
    ) as pool:
        assert sum(pool.map(_append_unique_events, workers)) == 32

    store = ExecutionTraceStore(path, fsync=False)
    records = store.read_all()

    assert len(records) == 32
    assert [record["sequence"] for record in records] == list(range(1, 33))
    assert store.verify()["valid"] is True
    assert len({record["event_id"] for record in records}) == 32


def test_concurrent_identical_event_id_is_idempotent_across_processes(tmp_path):
    path = str(tmp_path / "multiprocess-idempotency.jsonl")

    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=get_context("spawn"),
    ) as pool:
        results = list(pool.map(_append_same_event, [path] * 4))

    assert len({result["sequence"] for result in results}) == 1

    store = ExecutionTraceStore(path, fsync=False)
    records = store.read_all()
    assert len(records) == 1
    assert records[0]["event_id"] == "shared-idempotent-event"
    assert store.verify()["valid"] is True


def test_reader_observes_only_complete_records(tmp_path):
    path = tmp_path / "reader-writer-trace.jsonl"
    first = ExecutionTraceStore(path, fsync=False)
    second = ExecutionTraceStore(path, fsync=False)

    first.append({
        "event_id": "before",
        "execution_id": "read-test",
        "event_type": "started",
        "payload": {},
    })
    second.append({
        "event_id": "after",
        "execution_id": "read-test",
        "event_type": "completed",
        "payload": {},
    })

    records = first.read_all()
    assert [record["event_id"] for record in records] == ["before", "after"]
    assert first.verify()["valid"] is True
