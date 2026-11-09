import pytest

from agent.platform import (
    ApprovalPolicy, ContractError, Effect, Feature, Operation, OperationDisabled,
    OperationRegistry, TaskStatus, UnknownOperation, Verdict, Verification, can_transition,
)


def _noop(*_a, **_k):
    return None


def _verified(*_a, **_k):
    return Verification(Verdict.VERIFIED, {"rows": 1})


def read_op(**over):
    base = dict(feature="demo", name="look", effect=Effect.READ, permission="demo:read",
                risk=0.1, approval=ApprovalPolicy.none(), plan=_noop, execute=_noop)
    base.update(over)
    return Operation(**base)


def write_op(**over):
    base = dict(feature="demo", name="change", effect=Effect.WRITE, permission="demo:write",
                risk=0.6, approval=ApprovalPolicy(confirm_phrase=True), plan=_noop,
                preview=_noop, execute=_noop, verify=_verified)
    base.update(over)
    return Operation(**base)


# ---- operation rules -------------------------------------------------------

def test_read_operation_needs_no_preview_or_verifier():
    op = read_op()
    assert op.id == "demo.look"
    assert not op.approval.required


@pytest.mark.parametrize("effect", [Effect.WRITE, Effect.EXTERNAL])
def test_world_changing_operation_requires_preview(effect):
    with pytest.raises(ContractError, match="preview"):
        write_op(effect=effect, preview=None)


@pytest.mark.parametrize("effect", [Effect.WRITE, Effect.EXTERNAL])
def test_world_changing_operation_requires_verifier_or_explicit_waiver(effect):
    with pytest.raises(ContractError, match="verify"):
        write_op(effect=effect, verify=None)
    op = write_op(effect=effect, verify=None, verify_waived_reason="no way to observe the result")
    assert op.verify is None


def test_verifier_and_waiver_are_mutually_exclusive():
    with pytest.raises(ContractError):
        write_op(verify_waived_reason="because")


def test_blank_waiver_is_rejected():
    with pytest.raises(ContractError):
        write_op(verify=None, verify_waived_reason="   ")


def test_execute_is_required():
    with pytest.raises(ContractError, match="execute"):
        read_op(execute=None)


@pytest.mark.parametrize("risk", [-0.1, 1.1, True, "0.5", None])
def test_risk_must_be_a_number_between_zero_and_one(risk):
    with pytest.raises(ContractError):
        read_op(risk=risk)


@pytest.mark.parametrize("name", ["", "Bad", "has space", "1abc", "a" * 49, "dot.ted"])
def test_names_are_validated(name):
    with pytest.raises(ContractError):
        read_op(name=name)
    with pytest.raises(ContractError):
        read_op(feature=name)


def test_permission_must_be_text():
    with pytest.raises(ContractError):
        read_op(permission=" ")


def test_callables_must_be_callable():
    with pytest.raises(ContractError):
        read_op(plan="not callable")


def test_verification_that_claims_success_must_name_evidence():
    with pytest.raises(ContractError):
        Verification(Verdict.VERIFIED, {})
    assert Verification(Verdict.FAILED, {}).verdict is Verdict.FAILED


def test_operations_are_immutable():
    op = read_op()
    with pytest.raises(Exception):
        op.risk = 0.9  # type: ignore[misc]


# ---- features --------------------------------------------------------------

def test_feature_needs_operations_of_its_own_name_without_duplicates():
    with pytest.raises(ContractError):
        Feature("demo", "d", ())
    with pytest.raises(ContractError):
        Feature("other", "d", (read_op(),))
    with pytest.raises(ContractError):
        Feature("demo", "d", (read_op(), read_op()))
    assert Feature("demo", "d", [read_op(), write_op()]).operations[1].id == "demo.change"


# ---- registry --------------------------------------------------------------

def make_registry():
    registry = OperationRegistry()
    registry.register(Feature("demo", "d", (read_op(), write_op())))
    return registry


def test_unknown_operation_fails_closed():
    with pytest.raises(UnknownOperation):
        make_registry().get("demo.nothing")


def test_duplicate_feature_is_refused():
    registry = make_registry()
    with pytest.raises(Exception):
        registry.register(Feature("demo", "d", (read_op(),)))


def test_disabling_an_operation_or_a_feature_blocks_it():
    registry = make_registry()
    registry.disable_operation("demo.change")
    with pytest.raises(OperationDisabled):
        registry.get("demo.change")
    assert registry.get("demo.look").id == "demo.look"
    registry.enable_operation("demo.change")
    assert registry.is_enabled("demo.change")

    registry.disable_feature("demo")
    assert not registry.is_enabled("demo.look")
    assert registry.enabled_operations() == ()
    registry.enable_feature("demo")
    assert len(registry.enabled_operations()) == 2


def test_switching_something_unknown_is_an_error_not_a_noop():
    registry = make_registry()
    for call in (registry.disable_feature, registry.enable_feature,
                 registry.disable_operation, registry.enable_operation):
        with pytest.raises(UnknownOperation):
            call("missing")


def test_register_rejects_non_features():
    with pytest.raises(ContractError):
        OperationRegistry().register("demo")  # type: ignore[arg-type]


# ---- lifecycle -------------------------------------------------------------

def test_lifecycle_allows_the_normal_path_and_nothing_after_the_end():
    path = [TaskStatus.PLANNED, TaskStatus.AWAITING_APPROVAL, TaskStatus.APPROVED,
            TaskStatus.EXECUTING, TaskStatus.VERIFYING, TaskStatus.COMPLETED]
    for a, b in zip(path, path[1:]):
        assert can_transition(a, b)
    for status in TaskStatus:
        if status.terminal:
            assert not any(can_transition(status, other) for other in TaskStatus)


def test_lifecycle_cannot_skip_approval_or_verification():
    assert not can_transition(TaskStatus.AWAITING_APPROVAL, TaskStatus.EXECUTING)
    assert not can_transition(TaskStatus.EXECUTING, TaskStatus.COMPLETED)
    assert not can_transition(TaskStatus.PLANNED, TaskStatus.EXECUTING)
    assert not can_transition(TaskStatus.REJECTED, TaskStatus.APPROVED)


def test_second_approver_needs_an_approver_permission_and_not_the_other_way_round():
    with pytest.raises(ContractError, match="approver_permission"):
        write_op(approval=ApprovalPolicy(second_approver=True))
    with pytest.raises(ContractError, match="approver_permission"):
        write_op(approver_permission="demo:approve")
    op = write_op(approval=ApprovalPolicy(second_approver=True), approver_permission="demo:approve")
    assert op.approver_permission == "demo:approve"
