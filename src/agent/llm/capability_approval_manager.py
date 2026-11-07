from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from typing import Callable

from agent.llm.capability_proposal import (
    CapabilityProposal,
    ProposalCategory,
    ProposalStatus,
)
from agent.llm.capability_proposal_store import CapabilityProposalStore
from agent.llm.capability_proposal_validator import CapabilityProposalValidator
from agent.llm.capability_registry import (
    CapabilityRecord,
    CapabilityTrustLevel,
    TrustedCapabilityRegistry,
)
from agent.llm.dynamic_workspace_actions import (
    DynamicActionSpec,
    DynamicWorkspaceActionBuilder,
)
from agent.llm.dynamic_workspace_runtime_adapter import (
    DynamicWorkspaceRuntimeAdapter,
)


class CapabilityApprovalError(RuntimeError):
    """Base error for application approval and registration."""


class CapabilityApprovalAuthorizationError(CapabilityApprovalError):
    """Raised when an application actor is not authorized."""


class CapabilityApprovalValidationError(CapabilityApprovalError):
    """Raised when a proposal cannot be approved or registered."""


class CapabilityApprovalRegistrationError(CapabilityApprovalError):
    """Raised when trusted registration or planner synchronization fails."""


class CapabilityApprovalManager:
    """
    Application-owned approval workflow for declarative capability proposals.

    The LLM can submit proposal metadata only. It cannot provide a handler,
    callable, import path, or executable source. Read-only registration
    resolves the handler exclusively from DynamicWorkspaceActionBuilder's
    application-owned capability catalog.

    State-changing proposals may be reviewed and approved as proposals, but
    this stage deliberately refuses to register them.
    """

    def __init__(
        self,
        *,
        store: CapabilityProposalStore,
        validator: CapabilityProposalValidator,
        registry: TrustedCapabilityRegistry,
        action_builder: DynamicWorkspaceActionBuilder,
        adapter: DynamicWorkspaceRuntimeAdapter,
        approval_authorizer: Callable[[str, str], bool],
        read_only_capability_ids: Iterable[str],
    ) -> None:
        if not isinstance(store, CapabilityProposalStore):
            raise CapabilityApprovalValidationError("store must be a CapabilityProposalStore.")
        if not isinstance(validator, CapabilityProposalValidator):
            raise CapabilityApprovalValidationError(
                "validator must be a CapabilityProposalValidator."
            )
        if not isinstance(registry, TrustedCapabilityRegistry):
            raise CapabilityApprovalValidationError(
                "registry must be a TrustedCapabilityRegistry."
            )
        if not isinstance(action_builder, DynamicWorkspaceActionBuilder):
            raise CapabilityApprovalValidationError(
                "action_builder must be a DynamicWorkspaceActionBuilder."
            )
        if not isinstance(adapter, DynamicWorkspaceRuntimeAdapter):
            raise CapabilityApprovalValidationError(
                "adapter must be a DynamicWorkspaceRuntimeAdapter."
            )
        if adapter.registry is not registry:
            raise CapabilityApprovalValidationError(
                "adapter and manager must use the same registry."
            )
        if action_builder.agent is not adapter.agent:
            raise CapabilityApprovalValidationError(
                "action_builder and adapter must use the same Agent."
            )
        if not callable(approval_authorizer):
            raise CapabilityApprovalValidationError(
                "approval_authorizer must be callable; approval fails closed without it."
            )
        if isinstance(read_only_capability_ids, (str, bytes)):
            raise CapabilityApprovalValidationError(
                "read_only_capability_ids must be an iterable of capability IDs."
            )
        try:
            read_only_ids = frozenset(read_only_capability_ids)
        except TypeError as exc:
            raise CapabilityApprovalValidationError(
                "read_only_capability_ids must be iterable."
            ) from exc
        if any(
            not isinstance(item, str) or not item.strip()
            for item in read_only_ids
        ):
            raise CapabilityApprovalValidationError(
                "Read-only capability IDs must be non-empty strings."
            )
        unknown_read_only_ids = read_only_ids.difference(action_builder.capabilities)
        if unknown_read_only_ids:
            raise CapabilityApprovalValidationError(
                "Read-only allowlist references unknown application handlers: "
                + ", ".join(sorted(unknown_read_only_ids))
            )

        self.store = store
        self.validator = validator
        self.registry = registry
        self.action_builder = action_builder
        self.adapter = adapter
        self.agent = adapter.agent
        self.approval_authorizer = approval_authorizer
        # This allowlist is application configuration, never proposal input.
        self.read_only_capability_ids = read_only_ids
        self._lock = RLock()

    def _authorize(self, actor: str, operation: str) -> None:
        if not isinstance(actor, str) or not actor.strip():
            raise CapabilityApprovalAuthorizationError("Application actor is required.")
        try:
            allowed = self.approval_authorizer(actor.strip(), operation)
        except Exception as exc:
            raise CapabilityApprovalAuthorizationError(
                "Application approval authorization failed."
            ) from exc
        if allowed is not True:
            raise CapabilityApprovalAuthorizationError(
                f"Actor is not authorized to {operation} capability proposals."
            )

    def submit(self, proposal: CapabilityProposal) -> CapabilityProposal:
        self.validator.validate(proposal)
        return self.store.submit(proposal)

    def approve(self, proposal_id: str, *, approved_by: str) -> CapabilityProposal:
        self._authorize(approved_by, "approve")
        with self._lock:
            proposal = self.store.get(proposal_id)
            self.validator.validate(proposal)
            if proposal.category == ProposalCategory.READ_ONLY:
                if proposal.capability_id not in self.read_only_capability_ids:
                    raise CapabilityApprovalValidationError(
                        "Read-only proposal is not mapped to an application-approved "
                        "read-only handler."
                    )
            return self.store.transition(
                proposal_id,
                expected_status=ProposalStatus.PENDING,
                new_status=ProposalStatus.APPROVED,
                actor=approved_by,
            )

    def reject(
        self,
        proposal_id: str,
        *,
        rejected_by: str,
        reason: str,
    ) -> CapabilityProposal:
        self._authorize(rejected_by, "reject")
        if not isinstance(reason, str) or not reason.strip():
            raise CapabilityApprovalValidationError("A rejection reason is required.")
        with self._lock:
            return self.store.transition(
                proposal_id,
                expected_status=ProposalStatus.PENDING,
                new_status=ProposalStatus.REJECTED,
                actor=rejected_by,
                reason=reason,
            )

    def register_approved_read_only(
        self,
        proposal_id: str,
        *,
        registered_by: str,
    ) -> CapabilityRecord:
        self._authorize(registered_by, "register")
        with self._lock:
            proposal = self.store.get(proposal_id)
            if proposal.status != ProposalStatus.APPROVED:
                raise CapabilityApprovalValidationError(
                    "Only approved proposals can be registered."
                )
            if proposal.category != ProposalCategory.READ_ONLY:
                raise CapabilityApprovalValidationError(
                    "State-changing proposals cannot be registered in Stage 2.6."
                )
            self.validator.validate(
                CapabilityProposal(
                    proposal_id=proposal.proposal_id,
                    capability_id=proposal.capability_id,
                    action_name=proposal.action_name,
                    description=proposal.description,
                    category=proposal.category,
                    risk_level=proposal.risk_level,
                    permissions=proposal.permissions,
                    preconditions=proposal.preconditions,
                    effects=proposal.effects,
                    parameters=proposal.parameters,
                )
            )

            if proposal.capability_id not in self.read_only_capability_ids:
                raise CapabilityApprovalValidationError(
                    "Approved proposal is not mapped to an application-approved "
                    "read-only handler."
                )
            if proposal.capability_id not in self.action_builder.capabilities:
                raise CapabilityApprovalValidationError(
                    "Approved proposal no longer maps to an application-owned handler."
                )

            action = self.action_builder.build_action(
                DynamicActionSpec(
                    name=proposal.action_name,
                    description=proposal.description,
                    capability_id=proposal.capability_id,
                    preconditions=proposal.preconditions,
                    effects=proposal.effects,
                    parameters=proposal.parameters,
                )
            )

            try:
                record = self.registry.register_approved_proposal(
                    capability_id=proposal.capability_id,
                    action=action,
                    risk_level=proposal.risk_level,
                    permissions=proposal.permissions,
                )
            except Exception as exc:
                raise CapabilityApprovalRegistrationError(
                    "Approved proposal could not be registered in the trusted registry."
                ) from exc

            try:
                self.adapter.synchronize()
                updated = self.store.transition(
                    proposal_id,
                    expected_status=ProposalStatus.APPROVED,
                    new_status=ProposalStatus.REGISTERED,
                    actor=registered_by,
                )
            except BaseException as exc:
                try:
                    self.registry.rollback_approved_proposal_registration(
                        proposal.capability_id,
                        expected_action=action,
                    )
                    self.agent.configure_planner()
                except BaseException as rollback_exc:
                    raise CapabilityApprovalRegistrationError(
                        "Registration failed and rollback could not fully restore "
                        "registry/planner consistency."
                    ) from rollback_exc
                if isinstance(exc, CapabilityApprovalError):
                    raise
                raise CapabilityApprovalRegistrationError(
                    "Registration failed; registry and planner state were rolled back."
                ) from exc

            # The store transition is the final workflow commit. The registry
            # record is immutable and remains application-owned.
            return record


__all__ = [
    "CapabilityApprovalError",
    "CapabilityApprovalAuthorizationError",
    "CapabilityApprovalValidationError",
    "CapabilityApprovalRegistrationError",
    "CapabilityApprovalManager",
]
