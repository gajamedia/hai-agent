from types import SimpleNamespace

import pytest

from agent.core.action import Action
from agent.llm.capability_approval_manager import (
    CapabilityApprovalAuthorizationError,
    CapabilityApprovalManager,
    CapabilityApprovalRegistrationError,
    CapabilityApprovalValidationError,
)
from agent.llm.capability_proposal import ProposalCategory, ProposalStatus
from agent.llm.capability_proposal_store import CapabilityProposalStore
from agent.llm.capability_proposal_validator import CapabilityProposalValidator
from agent.llm.capability_registry import (
    CapabilityTrustLevel,
    TrustedCapabilityRegistry,
)
from agent.llm.dynamic_workspace_actions import DynamicWorkspaceActionBuilder
from agent.llm.dynamic_workspace_runtime_adapter import DynamicWorkspaceRuntimeAdapter


class FakeAgent:
    def __init__(self):
        self.actions = {}
        self._planner_sync_hook = None
        self.fail_next_configure = False
        self.configure_planner()

    def set_planner_sync_hook(self, hook):
        if self._planner_sync_hook is not None:
            raise RuntimeError("hook already set")
        self._planner_sync_hook = hook

    def clear_planner_sync_hook(self, hook=None):
        if self._planner_sync_hook is None or (
            hook is not None and hook is not self._planner_sync_hook
        ):
            return False
        self._planner_sync_hook = None
        return True

    def register_action(self, action):
        self.actions[action.name] = action
        self.configure_planner()

    def configure_planner(self):
        if self.fail_next_configure:
            self.fail_next_configure = False
            raise RuntimeError("injected planner rebuild failure")
        self.planner = SimpleNamespace(actions=list(self.actions.values()))
        self.adaptive_planner = SimpleNamespace(planner=self.planner)
        self.hierarchical_planner = SimpleNamespace(planner=self.planner)
        self.risk_aware_hierarchical_planner = SimpleNamespace(
            hierarchical_planner=self.hierarchical_planner
        )
        self.unified_executor = object()
        if self._planner_sync_hook is not None:
            self._planner_sync_hook()


def read_only_observer(*, environment, parameters, **kwargs):
    return {"success": True, "observation": environment.observe()}


class FakeEnvironment:
    def observe(self):
        return {"door": "closed"}


def make_manager(authorizer=None):
    agent = FakeAgent()
    registry = TrustedCapabilityRegistry()
    adapter = DynamicWorkspaceRuntimeAdapter(registry=registry, agent=agent)
    # DynamicWorkspaceActionBuilder already ships an application-owned
    # environment.observe handler; use that trusted catalog entry.
    builder = DynamicWorkspaceActionBuilder(agent)
    store = CapabilityProposalStore()
    validator = CapabilityProposalValidator()
    manager = CapabilityApprovalManager(
        store=store,
        validator=validator,
        registry=registry,
        action_builder=builder,
        adapter=adapter,
        approval_authorizer=authorizer or (lambda actor, operation: actor == "admin"),
        read_only_capability_ids={"environment.observe"},
    )
    return agent, registry, adapter, store, manager


def make_proposal(manager, *, category=ProposalCategory.READ_ONLY, effects=None):
    return manager.store.create(
        capability_id="environment.observe",
        action_name="observe_environment",
        description="Observe the current environment.",
        category=category,
        risk_level="low",
        permissions=("observe:environment",) if category == ProposalCategory.READ_ONLY else ("write:door",),
        effects=effects or {},
        parameters={"include": ["door"]},
    )


def test_read_only_proposal_requires_approval_before_registration():
    agent, registry, adapter, store, manager = make_manager()
    proposal = make_proposal(manager)

    with pytest.raises(CapabilityApprovalValidationError):
        manager.register_approved_read_only(proposal.proposal_id, registered_by="admin")

    approved = manager.approve(proposal.proposal_id, approved_by="admin")
    assert approved.status == ProposalStatus.APPROVED

    record = manager.register_approved_read_only(
        proposal.proposal_id,
        registered_by="admin",
    )

    assert record.trust_level == CapabilityTrustLevel.APPROVED_PROPOSAL
    assert registry.get("environment.observe") is record
    assert agent.actions["observe_environment"] is record.action
    assert record.action in agent.planner.actions
    assert store.get(proposal.proposal_id).status == ProposalStatus.REGISTERED


def test_state_changing_proposal_may_be_approved_but_not_registered():
    _, registry, _, _, manager = make_manager()
    proposal = make_proposal(
        manager,
        category=ProposalCategory.STATE_CHANGING,
        effects={"door": "open"},
    )
    manager.approve(proposal.proposal_id, approved_by="admin")

    with pytest.raises(CapabilityApprovalValidationError):
        manager.register_approved_read_only(
            proposal.proposal_id,
            registered_by="admin",
        )
    assert registry.list_all() == ()


def test_approval_fails_closed_for_unauthorized_actor():
    _, _, _, _, manager = make_manager(
        authorizer=lambda actor, operation: False
    )
    proposal = make_proposal(manager)
    with pytest.raises(CapabilityApprovalAuthorizationError):
        manager.approve(proposal.proposal_id, approved_by="llm")
    assert manager.store.get(proposal.proposal_id).status == ProposalStatus.PENDING


def test_rejection_requires_reason_and_is_terminal_for_approval():
    _, _, _, _, manager = make_manager()
    proposal = make_proposal(manager)
    rejected = manager.reject(
        proposal.proposal_id,
        rejected_by="admin",
        reason="Not needed.",
    )
    assert rejected.status == ProposalStatus.REJECTED
    with pytest.raises(Exception):
        manager.approve(proposal.proposal_id, approved_by="admin")


def test_failed_planner_sync_rolls_back_registry_and_agent():
    agent, registry, _, store, manager = make_manager()
    proposal = make_proposal(manager)
    manager.approve(proposal.proposal_id, approved_by="admin")
    agent.fail_next_configure = True

    with pytest.raises(CapabilityApprovalRegistrationError):
        manager.register_approved_read_only(
            proposal.proposal_id,
            registered_by="admin",
        )

    assert registry.list_all() == ()
    assert "observe_environment" not in agent.actions
    assert all(action.name != "observe_environment" for action in agent.planner.actions)
    assert store.get(proposal.proposal_id).status == ProposalStatus.APPROVED


def test_proposal_cannot_supply_or_replace_handler():
    _, _, _, _, manager = make_manager()
    proposal = make_proposal(manager)
    assert not hasattr(proposal, "handler")
    assert "environment.observe" in manager.action_builder.capabilities


def test_read_only_claim_without_application_allowlist_is_rejected():
    agent = FakeAgent()
    registry = TrustedCapabilityRegistry()
    adapter = DynamicWorkspaceRuntimeAdapter(registry=registry, agent=agent)
    builder = DynamicWorkspaceActionBuilder(agent)
    manager = CapabilityApprovalManager(
        store=CapabilityProposalStore(),
        validator=CapabilityProposalValidator(),
        registry=registry,
        action_builder=builder,
        adapter=adapter,
        approval_authorizer=lambda actor, operation: True,
        read_only_capability_ids=set(),
    )
    proposal = manager.store.create(
        capability_id="environment.observe",
        action_name="observe_environment",
        description="Observe the current environment.",
        category=ProposalCategory.READ_ONLY,
        risk_level="low",
        permissions=("observe:environment",),
    )
    with pytest.raises(CapabilityApprovalValidationError):
        manager.approve(proposal.proposal_id, approved_by="admin")
    assert manager.store.get(proposal.proposal_id).status == ProposalStatus.PENDING
