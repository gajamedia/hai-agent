import json
from dataclasses import dataclass

import pytest

from agent.platform import (
    ApprovalPolicy, ContractError, Effect, ExecutionFailed, Feature, Operation, OperationRegistry,
    PermissionDenied, PlatformPipeline, PolicyViolation, SQLiteTaskStore, StepUpRequired, TaskNotFound,
    TaskStatus, TooManyTasks, Verdict, Verification,
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


PLAN, RUN, APPROVE = "d:plan", "d:run", "d:approve"
alice = P("alice", (PLAN, RUN), auth_time=1_000.0)
bob = P("bob", (APPROVE,), auth_time=1_000.0)
planner_only = P("pat", (PLAN,), auth_time=1_000.0)

SECRET = "ssn-123-45-6789"


def build(*, mode=None, sensitive=True, execute=None, verify=None, **kw):
    """One operation like 'change data': a delete needs a second person, an update does not."""
    clock = Clock()
    seen = {"ctx": None, "calls": 0}
    state = {"mode": mode or "delete"}

    def plan(request, ctx):
        return {"operation": request["operation"], "value": request.get("value", SECRET)}

    def preview(plan, ctx):
        return {"confirm_phrase": f"{plan['operation'].upper()} 1", "sample": plan["value"]}

    def run(plan, ctx):
        seen["calls"] += 1
        seen["ctx"] = ctx
        return {"done": plan["operation"]}

    def approval_for(plan, ctx):
        needs = state["mode"] == "all" or (state["mode"] == "delete" and plan["operation"] == "delete")
        return ApprovalPolicy(confirm_phrase=True, fresh_login=True, second_approver=needs)

    def default_verify(plan, result, ctx):
        return Verification(Verdict.VERIFIED, {"ok": True})

    op = Operation(
        "d", "change", Effect.WRITE, RUN, 0.8, ApprovalPolicy(confirm_phrase=True, fresh_login=True),
        plan=plan, preview=preview, execute=execute or run, verify=verify or default_verify,
        approver_permission=APPROVE, plan_permission=PLAN, approval_for=approval_for, sensitive=sensitive,
        max_active=kw.pop("max_active", None))
    registry = OperationRegistry()
    registry.register(Feature("d", "d", (op,)))
    store = SQLiteTaskStore(":memory:", clock=clock)
    pipe = PlatformPipeline(registry=registry, store=store, authorize=authorize, clock=clock, **kw)
    return pipe, store, clock, seen, state


def stored_text(store):
    return json.dumps([store.get(t.task_id).__dict__ for t in store.list_by_status(list(TaskStatus))],
                      default=str) + json.dumps(store.audit_records())


# ---- contract rules --------------------------------------------------------

def test_contract_rules_for_the_new_fields():
    base = dict(feature="d", name="x", effect=Effect.READ, permission="p", risk=0.1,
                approval=ApprovalPolicy.none(), execute=lambda *a: {})
    with pytest.raises(ContractError):
        Operation(**base, plan_permission=" ")
    with pytest.raises(ContractError):
        Operation(**base, max_active=0)
    with pytest.raises(ContractError):
        Operation(**base, approval_for="not callable")
    # an approver permission is fine when a per-plan policy may ask for a second approver
    assert Operation(**base, approver_permission="a", approval_for=lambda p, c: ApprovalPolicy()) \
        .approver_permission == "a"


def test_policy_covers_and_round_trips():
    base = ApprovalPolicy(confirm_phrase=True, fresh_login=True)
    assert ApprovalPolicy(True, True, True).covers(base) and base.covers(base)
    assert not ApprovalPolicy(True, False, True).covers(base)
    assert not ApprovalPolicy.none().covers(base)
    assert ApprovalPolicy.from_dict(base.as_dict()) == base


# ---- approval policy decided per plan --------------------------------------

def test_a_delete_waits_for_a_second_person_but_an_update_does_not():
    pipe, _, _, _, _ = build()
    update = pipe.submit("d.change", {"operation": "update"}, alice)
    delete = pipe.submit("d.change", {"operation": "delete"}, alice)
    assert update.status is TaskStatus.APPROVED and delete.status is TaskStatus.AWAITING_APPROVAL
    assert update.approval == {"confirm_phrase": True, "fresh_login": True, "second_approver": False}
    assert delete.approval["second_approver"] is True


def test_the_policy_is_fixed_when_the_plan_is_made_not_when_it_runs():
    pipe, _, _, _, state = build(mode="none")
    task = pipe.submit("d.change", {"operation": "delete"}, alice)
    assert task.status is TaskStatus.APPROVED
    state["mode"] = "all"                                        # the server is made stricter later
    assert pipe.run(task.task_id, alice, confirm="DELETE 1").task.status is TaskStatus.COMPLETED


def test_a_plan_policy_weaker_than_the_baseline_is_refused():
    pipe, store, _, _, _ = build()
    op = pipe.registry.get("d.change")
    object.__setattr__(op, "approval_for", lambda p, c: ApprovalPolicy(confirm_phrase=True))
    with pytest.raises(PolicyViolation, match="only add"):
        pipe.submit("d.change", {"operation": "update"}, alice)
    object.__setattr__(op, "approval_for", lambda p, c: "nothing")
    with pytest.raises(PolicyViolation):
        pipe.submit("d.change", {"operation": "update"}, alice)
    assert store.list_by_status(list(TaskStatus)) == []


# ---- planning and running need different permissions -----------------------

def test_planning_needs_the_plan_permission_and_running_the_run_permission():
    pipe, _, _, _, _ = build()
    with pytest.raises(PermissionDenied):
        pipe.submit("d.change", {"operation": "update"}, bob)            # neither
    task = pipe.submit("d.change", {"operation": "update"}, planner_only)  # may plan...
    with pytest.raises(PermissionDenied):
        pipe.run(task.task_id, planner_only, confirm="UPDATE 1")           # ...but not run


# ---- approving ---------------------------------------------------------------

def test_approving_needs_a_fresh_login_but_rejecting_does_not():
    pipe, _, clock, _, _ = build()
    one = pipe.submit("d.change", {"operation": "delete"}, alice)
    two = pipe.submit("d.change", {"operation": "delete"}, alice)
    stale = P("bob", (APPROVE,), auth_time=clock.now - 3600)
    with pytest.raises(StepUpRequired):
        pipe.approve(one.task_id, stale)
    assert pipe.approve(one.task_id, bob).status is TaskStatus.APPROVED
    assert pipe.approve(two.task_id, stale, reject=True).status is TaskStatus.REJECTED


def test_the_run_knows_who_approved_it():
    pipe, _, _, seen, _ = build()
    task = pipe.submit("d.change", {"operation": "delete"}, alice)
    pipe.approve(task.task_id, bob)
    pipe.run(task.task_id, alice, confirm="DELETE 1")
    assert seen["ctx"].approved_by == "bob" and seen["ctx"].task_id == task.task_id


def test_settings_given_as_functions_are_read_when_needed():
    flags = {"self": False, "window": 300}
    pipe, _, clock, _, _ = build(allow_self_approval=lambda: flags["self"],
                                 step_up_max_age_seconds=lambda: flags["window"])
    solo = P("solo", (PLAN, RUN, APPROVE), auth_time=1_000.0)
    task = pipe.submit("d.change", {"operation": "delete"}, solo)
    assert pipe.waiting_for_approval(solo) == []
    with pytest.raises(PolicyViolation):
        pipe.approve(task.task_id, solo)
    flags["self"] = True
    assert [t.task_id for t in pipe.waiting_for_approval(solo)] == [task.task_id]
    clock.now += 200
    flags["window"] = 100                                    # a shorter window now applies at once
    with pytest.raises(StepUpRequired):
        pipe.approve(task.task_id, solo)
    flags["window"] = 300
    assert pipe.approve(task.task_id, solo).status is TaskStatus.APPROVED


# ---- sensitive plans are erased when the task ends --------------------------

def _gone(store, task_id):
    task = store.get(task_id)
    assert task.plan.get("scrubbed") is True and "plan_sha256" in task.plan
    assert task.preview is None and task.confirm_phrase is None
    assert SECRET not in stored_text(store)


def test_the_plan_is_kept_while_waiting_and_erased_after_a_successful_run():
    pipe, store, _, _, _ = build()
    task = pipe.submit("d.change", {"operation": "update"}, alice)
    assert SECRET in json.dumps(store.get(task.task_id).plan)        # needed to run it
    out = pipe.run(task.task_id, alice, confirm="UPDATE 1")
    assert out.task.status is TaskStatus.COMPLETED
    _gone(store, task.task_id)
    assert store.get(task.task_id).result == {"done": "update"}      # the outcome is kept


@pytest.mark.parametrize("how", ["cancel", "reject", "fail", "unverified", "expire", "recover"])
def test_every_way_a_task_can_end_erases_the_plan(how):
    pipe, store, clock, _, _ = build(
        execute=(lambda p, c: 1 / 0) if how == "fail" else None,
        verify=(lambda p, r, c: Verification(Verdict.UNVERIFIED, {})) if how == "unverified" else None)
    op = "delete" if how == "reject" else "update"
    task = pipe.submit("d.change", {"operation": op}, alice)
    if how == "cancel":
        pipe.cancel(task.task_id, alice)
    elif how == "reject":
        pipe.approve(task.task_id, bob, reject=True)
    elif how in ("fail", "unverified"):
        try:
            pipe.run(task.task_id, alice, confirm="UPDATE 1")
        except ExecutionFailed:
            pass
    elif how == "expire":
        clock.now += 10_000
        pipe.submit("d.change", {"operation": "update", "value": "other"}, alice)   # sweeps the old one
    else:
        store.transition(task.task_id, expected_version=task.version, new_status=TaskStatus.EXECUTING,
                         actor="alice", audit_event="x", audit_detail={})
        pipe.recover_interrupted()
    assert store.get(task.task_id).status.terminal
    _gone(store, task.task_id)


def test_a_not_sensitive_operation_keeps_its_plan():
    pipe, store, _, _, _ = build(sensitive=False)
    task = pipe.submit("d.change", {"operation": "update"}, alice)
    pipe.run(task.task_id, alice, confirm="UPDATE 1")
    assert store.get(task.task_id).plan["value"] == SECRET


def test_the_audit_never_holds_the_data_even_while_the_task_waits():
    pipe, store, _, _, _ = build()
    pipe.submit("d.change", {"operation": "update"}, alice)
    assert SECRET not in json.dumps(store.audit_records())


# ---- caps and ownership ------------------------------------------------------

def test_an_operation_can_set_its_own_cap():
    pipe, _, _, _, _ = build(max_active=1, max_active_per_operation=50)
    pipe.submit("d.change", {"operation": "update"}, alice)
    with pytest.raises(TooManyTasks):
        pipe.submit("d.change", {"operation": "update"}, alice)


def test_an_approver_may_look_at_a_waiting_task_but_a_bystander_may_not():
    pipe, _, _, _, _ = build()
    task = pipe.submit("d.change", {"operation": "delete"}, alice)
    assert pipe.get_task(task.task_id, bob).task_id == task.task_id
    with pytest.raises(TaskNotFound):
        pipe.get_task(task.task_id, planner_only)
    update = pipe.submit("d.change", {"operation": "update"}, alice)     # no second approver: not theirs
    with pytest.raises(TaskNotFound):
        pipe.get_task(update.task_id, bob)
