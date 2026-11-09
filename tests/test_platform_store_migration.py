"""A task file written by an older version keeps working after the schema grew."""
import sqlite3

import pytest

from agent.platform import SQLiteTaskStore, TaskStatus

# the table exactly as the first release created it (step 1b/1c): no exec_key, no approval_json
OLD_TASKS = """
CREATE TABLE platform_tasks (
    task_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, owner TEXT NOT NULL, status TEXT NOT NULL,
    version INTEGER NOT NULL, plan_json TEXT NOT NULL, preview_json TEXT, confirm_phrase TEXT,
    result_json TEXT, verification_json TEXT, error TEXT, approved_by TEXT, run_key TEXT,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE platform_audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, event TEXT NOT NULL, task_id TEXT,
    operation_id TEXT, actor TEXT, detail_json TEXT NOT NULL);
"""


@pytest.fixture
def old_file(tmp_path):
    path = tmp_path / "platform.sqlite3"
    with sqlite3.connect(path) as c:
        c.executescript(OLD_TASKS)
        c.execute("INSERT INTO platform_tasks VALUES ('t1','database.report','alice','approved',1,"
                  "'{\"q\":1}',NULL,NULL,NULL,NULL,NULL,NULL,NULL,1.0,1.0,9999999999.0)")
        c.execute("INSERT INTO platform_audit(at,event,task_id,operation_id,actor,detail_json) "
                  "VALUES (1.0,'task_planned','t1','database.report','alice','{}')")
    return path


def test_an_old_file_is_upgraded_and_its_tasks_and_audit_are_kept(old_file):
    store = SQLiteTaskStore(old_file, clock=lambda: 5.0)
    task = store.get("t1")
    assert task.status is TaskStatus.APPROVED and task.plan == {"q": 1} and task.exec_key is None
    assert task.approval == {"confirm_phrase": False, "fresh_login": False, "second_approver": False}
    assert [r["event"] for r in store.audit_records("t1")] == ["task_planned"]


def test_the_upgraded_file_accepts_new_tasks(old_file):
    store = SQLiteTaskStore(old_file, clock=lambda: 5.0)
    new = store.create(operation_id="database.change", owner="bob", status=TaskStatus.AWAITING_APPROVAL,
                       plan={"x": 1}, preview=None, confirm_phrase="P",
                       approval={"confirm_phrase": True, "fresh_login": True, "second_approver": True},
                       run_key=None, ttl_seconds=60, audit_event="task_planned", audit_detail={})
    assert store.get(new.task_id).approval["second_approver"] is True


def test_opening_an_upgraded_or_new_file_again_changes_nothing(old_file):
    SQLiteTaskStore(old_file, clock=lambda: 5.0)
    again = SQLiteTaskStore(old_file, clock=lambda: 5.0)
    assert again.get("t1").approval["second_approver"] is False
    fresh = SQLiteTaskStore(old_file.parent / "new.sqlite3", clock=lambda: 5.0)
    SQLiteTaskStore(old_file.parent / "new.sqlite3", clock=lambda: 5.0)
    assert fresh.list_by_status(list(TaskStatus)) == []
