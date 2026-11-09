import json
import threading
import time
from dataclasses import dataclass

import pytest

from agent.platform import (
    ApprovalPolicy, ApprovalRequired, ConfirmationMismatch, Effect, ExecutionFailed, Feature,
    Operation, OperationDisabled, OperationRegistry, PermissionDenied, PlanError, PlatformPipeline,
    PolicyViolation, SQLiteTaskStore, StepUpRequired, TaskConflict, TaskExpired, TaskNotFound,
    TaskStatus, UnknownOperation, Verdict, Verification,
)


@dataclass(frozen=True)
class P:
    actor_id: str
    permissions: tuple = ()
    auth_time: float | None = None


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def authorize(principal, permission):
    if permission not in principal.permissions:
        raise PermissionError(permission)


ALL = ("demo:read", "demo:write", "demo:approve")
alice = P("alice", ALL, auth_time=1_000.0)
bob = P("bob", ALL, auth_time=1_000.0)
carol = P("carol", ("demo:read",))


class Calls:
    def __init__(self):
        self.executed = []
        self.verified = 0


def build(tmp_path=None, **pipeline_kw):
    calls = Calls()
    clock = Clock()

    def plan(request, ctx):
        return {"target": request["target"], "secret": request.get("secret")}

    def preview(plan, ctx):
        return {"summary": f"change {plan['target']}", "confirm_phrase": f"CHANGE {plan['target']}"}

    def execute(plan, ctx):
        calls.executed.append((plan["target"], ctx.task_id))
        return {"changed": plan["target"]}

    def verify(plan, result, ctx):
        calls.verified += 1
        return Verification(Verdict.VERIFIED, {"seen": result["changed"]})

    ops = [
        Operation("demo", "look", Effect.READ, "demo:read", 0.1, ApprovalPolicy.none(),
                  plan=plan, execute=execute),
        Operation("demo", "change", Effect.WRITE, "demo:write", 0.6,
                  ApprovalPolicy(confirm_phrase=True, fresh_login=True), plan=plan, preview=preview,
                  execute=execute, verify=verify),
        Operation("demo", "dual", Effect.WRITE, "demo:write", 0.9,
                  ApprovalPolicy(confirm_phrase=True, second_approver=True), plan=plan,
                  preview=preview, execute=execute, verify=verify, approver_permission="demo:approve"),
        Operation("demo", "naive", Effect.WRITE, "demo:write", 0.9, ApprovalPolicy.none(), plan=plan,
                  preview=preview, execute=execute, verify_waived_reason="cannot observe"),
        Operation("demo", "waived", Effect.WRITE, "demo:write", 0.1, ApprovalPolicy.none(), plan=plan,
                  preview=preview, execute=execute, verify_waived_reason="cannot observe"),
    ]
    registry = OperationRegistry()
    registry.register(Feature("demo", "demo", tuple(ops)))
    db = str(tmp_path / "platform.sqlite3") if tmp_path else ":memory:"
    store = SQLiteTaskStore(db, clock=clock)
    pipeline = PlatformPipeline(registry=registry, store=store, authorize=authorize, clock=clock,
                                **pipeline_kw)
    return pipeline, store, registry, clock, calls


def events(store, task_id=None):
    return [r["event"] for r in store.audit_records(task_id)]


# ---- read path --------------------------------------------------------------

def test_read_operation_runs_and_is_audited():
    pipe, store, _, _, calls = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    assert task.status is TaskStatus.APPROVED
    out = pipe.run(task.task_id, alice)
    assert out.task.status is TaskStatus.COMPLETED and not out.replayed
    assert out.task.verification == {"verdict": "not_applicable"}
    assert calls.executed == [("a", task.task_id)]
    assert events(store, task.task_id) == [
        "task_planned", "task_executing", "task_executed", "task_completed"]


def test_unknown_and_disabled_operations_fail_closed_and_are_audited():
    pipe, store, registry, _, calls = build()
    with pytest.raises(UnknownOperation):
        pipe.submit("demo.nothing", {}, alice)
    registry.disable_feature("demo")
    with pytest.raises(OperationDisabled):
        pipe.submit("demo.look", {"target": "a"}, alice)
    assert "operation_refused" in events(store)
    assert calls.executed == []


def test_disabling_after_planning_blocks_the_run():
    pipe, _, registry, _, calls = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    registry.disable_operation("demo.look")
    with pytest.raises(OperationDisabled):
        pipe.run(task.task_id, alice)
    assert calls.executed == []


# ---- authorization ---------------------------------------------------------

def test_missing_permission_is_denied_and_creates_no_task():
    pipe, store, _, _, _ = build()
    with pytest.raises(PermissionDenied):
        pipe.submit("demo.change", {"target": "a"}, carol)
    assert store.list_by_status(list(TaskStatus)) == []
    assert "permission_denied" in events(store)


def test_a_failing_authorizer_denies():
    pipe, _, _, _, calls = build()
    pipe._authorize = lambda *_: (_ for _ in ()).throw(RuntimeError("down"))
    with pytest.raises(PermissionDenied):
        pipe.submit("demo.look", {"target": "a"}, alice)
    assert calls.executed == []


def test_permission_is_checked_again_at_run_time():
    pipe, _, _, _, calls = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    pipe._authorize = lambda *_: (_ for _ in ()).throw(PermissionError("revoked"))
    with pytest.raises(PermissionDenied):
        pipe.run(task.task_id, alice)
    assert calls.executed == []


def test_caller_without_identity_is_refused():
    pipe, _, _, _, _ = build()
    with pytest.raises(PermissionDenied):
        pipe.submit("demo.look", {"target": "a"}, object())


def test_risky_operation_without_approval_is_refused():
    pipe, store, _, _, calls = build()
    with pytest.raises(PolicyViolation):
        pipe.submit("demo.naive", {"target": "a"}, alice)
    assert "policy_violation" in events(store)
    assert store.list_by_status(list(TaskStatus)) == []


# ---- planning --------------------------------------------------------------

def test_invalid_plans_create_no_task():
    pipe, store, registry, _, _ = build()
    op = registry.get("demo.look")
    for bad in (lambda r, c: "text", lambda r, c: {"x": object()}, lambda r, c: 1 / 0):
        object.__setattr__(op, "plan", bad)
        with pytest.raises(PlanError):
            pipe.submit("demo.look", {"target": "a"}, alice)
    assert store.list_by_status(list(TaskStatus)) == []
    assert events(store).count("plan_failed") == 3


def test_confirm_phrase_must_come_from_the_preview():
    pipe, _, registry, _, _ = build()
    object.__setattr__(registry.get("demo.change"), "preview", lambda p, c: {"summary": "x"})
    with pytest.raises(PlanError):
        pipe.submit("demo.change", {"target": "a"}, alice)


# ---- confirmation and fresh login ------------------------------------------

def test_wrong_phrase_does_not_run_and_the_task_stays_runnable():
    pipe, _, _, _, calls = build()
    task = pipe.submit("demo.change", {"target": "a"}, alice)
    for wrong in (None, "", "change a", "CHANGE b"):
        with pytest.raises(ConfirmationMismatch):
            pipe.run(task.task_id, alice, confirm=wrong)
    assert calls.executed == []
    out = pipe.run(task.task_id, alice, confirm="  CHANGE a ")
    assert out.task.status is TaskStatus.COMPLETED
    assert out.task.verification["verdict"] == "verified"
    assert calls.verified == 1


def test_fresh_login_is_required():
    pipe, _, _, clock, calls = build()
    task = pipe.submit("demo.change", {"target": "a"}, alice)
    with pytest.raises(StepUpRequired):
        pipe.run(task.task_id, P("alice", ALL, auth_time=None), confirm="CHANGE a")
    clock.now += 301
    with pytest.raises(StepUpRequired):
        pipe.run(task.task_id, alice, confirm="CHANGE a")
    with pytest.raises(StepUpRequired):  # a login "from the future" is not trusted either
        pipe.run(task.task_id, P("alice", ALL, auth_time=clock.now + 50), confirm="CHANGE a")
    assert calls.executed == []
    assert pipe.run(task.task_id, P("alice", ALL, auth_time=clock.now), confirm="CHANGE a") \
        .task.status is TaskStatus.COMPLETED


# ---- second approver -------------------------------------------------------

def test_second_approver_flow():
    pipe, store, _, _, calls = build()
    task = pipe.submit("demo.dual", {"target": "a"}, alice)
    assert task.status is TaskStatus.AWAITING_APPROVAL
    with pytest.raises(ApprovalRequired):
        pipe.run(task.task_id, alice, confirm="CHANGE a")
    with pytest.raises(PolicyViolation):
        pipe.approve(task.task_id, alice)
    with pytest.raises(PermissionDenied):
        pipe.approve(task.task_id, carol)
    assert [t.task_id for t in pipe.waiting_for_approval(bob)] == [task.task_id]
    assert pipe.waiting_for_approval(alice) == []
    assert pipe.waiting_for_approval(carol) == []
    approved = pipe.approve(task.task_id, bob)
    assert approved.status is TaskStatus.APPROVED and approved.approved_by == "bob"
    out = pipe.run(task.task_id, alice, confirm="CHANGE a")
    assert out.task.status is TaskStatus.COMPLETED
    assert calls.executed == [("a", task.task_id)]
    assert "self_approval_refused" in events(store)


def test_rejected_task_cannot_run():
    pipe, _, _, _, calls = build()
    task = pipe.submit("demo.dual", {"target": "a"}, alice)
    assert pipe.approve(task.task_id, bob, reject=True).status is TaskStatus.REJECTED
    with pytest.raises(TaskConflict):
        pipe.run(task.task_id, alice, confirm="CHANGE a")
    with pytest.raises(TaskConflict):
        pipe.approve(task.task_id, bob)
    assert calls.executed == []


def test_self_approval_is_opt_in_and_audited():
    pipe, store, _, _, _ = build(allow_self_approval=True)
    task = pipe.submit("demo.dual", {"target": "a"}, alice)
    assert [t.task_id for t in pipe.waiting_for_approval(alice)] == [task.task_id]
    pipe.approve(task.task_id, alice)
    approved = [r for r in store.audit_records(task.task_id) if r["event"] == "task_approved"][0]
    assert approved["detail"]["self_approved"] is True


def test_approving_an_operation_without_a_second_approver_is_a_conflict():
    pipe, _, _, _, _ = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    with pytest.raises(TaskConflict):
        pipe.approve(task.task_id, bob)


# ---- once only -------------------------------------------------------------

def test_running_twice_returns_the_stored_outcome_without_running_again():
    pipe, _, _, _, calls = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    first = pipe.run(task.task_id, alice)
    again = pipe.run(task.task_id, alice)
    assert again.replayed and again.task.result == first.task.result
    assert len(calls.executed) == 1


def test_concurrent_runs_execute_once():
    pipe, _, registry, _, calls = build()
    op = registry.get("demo.look")
    original = op.execute

    def slow(plan, ctx):
        time.sleep(0.2)
        return original(plan, ctx)

    object.__setattr__(op, "execute", slow)
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    results, barrier = [], threading.Barrier(6)

    def worker():
        barrier.wait()
        try:
            results.append(pipe.run(task.task_id, alice))
        except TaskConflict as exc:
            results.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls.executed) == 1
    assert sum(1 for r in results if not isinstance(r, Exception) and not r.replayed) == 1


def test_idempotency_key_returns_the_same_task_and_plans_once():
    pipe, _, registry, _, _ = build()
    count = []
    op = registry.get("demo.look")
    inner = op.plan
    object.__setattr__(op, "plan", lambda r, c: count.append(1) or inner(r, c))
    one = pipe.submit("demo.look", {"target": "a"}, alice, idempotency_key="k1")
    two = pipe.submit("demo.look", {"target": "a"}, alice, idempotency_key=" k1 ")
    other = pipe.submit("demo.look", {"target": "a"}, bob, idempotency_key="k1")
    assert one.task_id == two.task_id != other.task_id
    assert len(count) == 2
    with pytest.raises(PolicyViolation):
        pipe.submit("demo.look", {"target": "a"}, alice, idempotency_key="  ")


# ---- ownership --------------------------------------------------------------

def test_other_people_cannot_see_run_or_cancel_a_task():
    pipe, _, _, _, calls = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    for call in (lambda: pipe.run(task.task_id, bob), lambda: pipe.cancel(task.task_id, bob),
                 lambda: pipe.get_task(task.task_id, carol)):
        with pytest.raises(TaskNotFound):
            call()
    assert calls.executed == []
    assert pipe.get_task(task.task_id, alice).task_id == task.task_id


def test_an_approver_may_look_at_a_task_waiting_for_them():
    pipe, _, _, _, _ = build()
    task = pipe.submit("demo.dual", {"target": "a"}, alice)
    assert pipe.get_task(task.task_id, bob).task_id == task.task_id
    with pytest.raises(TaskNotFound):
        pipe.get_task(task.task_id, carol)


def test_owner_can_cancel_before_running_but_not_after():
    pipe, _, _, _, calls = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    assert pipe.cancel(task.task_id, alice).status is TaskStatus.CANCELLED
    with pytest.raises(TaskConflict):
        pipe.run(task.task_id, alice)
    done = pipe.submit("demo.look", {"target": "b"}, alice)
    pipe.run(done.task_id, alice)
    with pytest.raises(TaskConflict):
        pipe.cancel(done.task_id, alice)


# ---- failures and verification ---------------------------------------------

def test_a_failing_operation_is_recorded_and_never_retried():
    pipe, store, registry, _, _ = build()
    object.__setattr__(registry.get("demo.look"), "execute", lambda p, c: 1 / 0)
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    with pytest.raises(ExecutionFailed):
        pipe.run(task.task_id, alice)
    stored = store.get(task.task_id)
    assert stored.status is TaskStatus.FAILED and "check the target" in stored.error
    with pytest.raises(TaskConflict):
        pipe.run(task.task_id, alice)


@pytest.mark.parametrize("verdict,expected", [
    (Verdict.FAILED, TaskStatus.FAILED), (Verdict.UNVERIFIED, TaskStatus.UNVERIFIED)])
def test_only_a_confirmed_result_is_completed(verdict, expected):
    pipe, _, registry, _, _ = build()
    object.__setattr__(registry.get("demo.change"), "verify",
                       lambda p, r, c: Verification(verdict, {"why": "no"}))
    task = pipe.submit("demo.change", {"target": "a"}, alice)
    out = pipe.run(task.task_id, alice, confirm="CHANGE a")
    assert out.task.status is expected and out.task.status is not TaskStatus.COMPLETED
    assert out.task.result == {"changed": "a"}  # the result is kept even when it is not confirmed


@pytest.mark.parametrize("bad", [lambda p, r, c: 1 / 0, lambda p, r, c: "verified"])
def test_a_broken_verifier_never_yields_completed(bad):
    pipe, _, registry, _, _ = build()
    object.__setattr__(registry.get("demo.change"), "verify", bad)
    task = pipe.submit("demo.change", {"target": "a"}, alice)
    out = pipe.run(task.task_id, alice, confirm="CHANGE a")
    assert out.task.status is TaskStatus.UNVERIFIED


def test_a_declared_waiver_is_recorded_not_hidden():
    pipe, _, _, _, _ = build()
    task = pipe.submit("demo.waived", {"target": "a"}, alice)
    out = pipe.run(task.task_id, alice)
    assert out.task.status is TaskStatus.COMPLETED
    assert out.task.verification == {"verdict": "waived", "reason": "cannot observe"}


# ---- expiry ----------------------------------------------------------------

def test_an_expired_task_is_cancelled_and_cannot_run():
    pipe, store, _, clock, calls = build(ttl_seconds=60)
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    clock.now += 61
    with pytest.raises(TaskExpired):
        pipe.run(task.task_id, alice)
    assert store.get(task.task_id).status is TaskStatus.CANCELLED
    assert calls.executed == []
    assert "task_expired" in events(store, task.task_id)


def test_an_expired_task_cannot_be_approved():
    pipe, _, _, clock, _ = build(ttl_seconds=60)
    task = pipe.submit("demo.dual", {"target": "a"}, alice)
    clock.now += 61
    assert pipe.waiting_for_approval(bob) == []
    with pytest.raises(TaskExpired):
        pipe.approve(task.task_id, bob)


# ---- durability and audit --------------------------------------------------

def test_tasks_survive_a_restart(tmp_path):
    pipe, store, registry, clock, calls = build(tmp_path)
    task = pipe.submit("demo.dual", {"target": "a"}, alice)
    pipe.approve(task.task_id, bob)
    # a new process: new store, new pipeline over the same file
    store2 = SQLiteTaskStore(tmp_path / "platform.sqlite3", clock=clock)
    pipe2 = PlatformPipeline(registry=registry, store=store2, authorize=authorize, clock=clock)
    assert pipe2.get_task(task.task_id, alice).status is TaskStatus.APPROVED
    assert pipe2.run(task.task_id, alice, confirm="CHANGE a").task.status is TaskStatus.COMPLETED
    assert len(calls.executed) == 1


def test_restart_recovery_never_reruns_anything(tmp_path):
    pipe, store, _, _, calls = build(tmp_path)
    running = pipe.submit("demo.look", {"target": "a"}, alice)
    store.transition(running.task_id, expected_version=running.version,
                     new_status=TaskStatus.EXECUTING, actor="alice", audit_event="x", audit_detail={})
    verifying = pipe.submit("demo.look", {"target": "b"}, alice)
    v = store.transition(verifying.task_id, expected_version=verifying.version,
                         new_status=TaskStatus.EXECUTING, actor="alice", audit_event="x", audit_detail={})
    store.transition(verifying.task_id, expected_version=v.version, new_status=TaskStatus.VERIFYING,
                     actor="alice", audit_event="x", audit_detail={}, result={"changed": "b"})
    recovered = {t.task_id: t for t in pipe.recover_interrupted()}
    assert recovered[running.task_id].status is TaskStatus.FAILED
    assert recovered[verifying.task_id].status is TaskStatus.UNVERIFIED
    assert recovered[verifying.task_id].result == {"changed": "b"}
    assert calls.executed == []
    assert pipe.recover_interrupted() == []


def test_audit_failure_stops_the_change(monkeypatch):
    pipe, store, _, _, calls = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_audit", broken)
    with pytest.raises(OSError):
        pipe.run(task.task_id, alice)
    monkeypatch.undo()
    assert store.get(task.task_id).status is TaskStatus.APPROVED  # nothing half-recorded
    assert calls.executed == []


def test_audit_never_holds_plan_contents_only_a_hash():
    pipe, store, _, _, _ = build()
    task = pipe.submit("demo.look", {"target": "a", "secret": "s3cret-value"}, alice)
    pipe.run(task.task_id, alice)
    blob = json.dumps(store.audit_records())
    assert "s3cret-value" not in blob and "plan_sha256" in blob


def test_status_changes_are_compare_and_set():
    pipe, store, _, _, _ = build()
    task = pipe.submit("demo.look", {"target": "a"}, alice)
    store.transition(task.task_id, expected_version=task.version, new_status=TaskStatus.CANCELLED,
                     actor="alice", audit_event="x", audit_detail={})
    with pytest.raises(TaskConflict):
        store.transition(task.task_id, expected_version=task.version, new_status=TaskStatus.CANCELLED,
                         actor="alice", audit_event="x", audit_detail={})
