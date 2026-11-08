from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from agent.llm.capability_proposal import (
    CapabilityProposal,
    CapabilityProposalStateError,
    ProposalStatus,
)
from agent.llm.capability_proposal_validator import CapabilityProposalValidator


class CapabilityProposalStore:
    """Thread-safe in-memory proposal store for the initial workflow stage."""

    def __init__(self, *, validator: CapabilityProposalValidator | None = None) -> None:
        self._lock = RLock()
        self._proposals: dict[str, CapabilityProposal] = {}
        self.validator = validator or CapabilityProposalValidator()

    def submit(self, proposal: CapabilityProposal) -> CapabilityProposal:
        self.validator.validate(proposal)
        with self._lock:
            if proposal.proposal_id in self._proposals:
                raise CapabilityProposalStateError(
                    f"Proposal already exists: {proposal.proposal_id}"
                )
            for existing in self._proposals.values():
                if existing.status in (
                    ProposalStatus.PENDING,
                    ProposalStatus.APPROVED,
                    ProposalStatus.REGISTERED,
                ):
                    if existing.capability_id == proposal.capability_id:
                        raise CapabilityProposalStateError(
                            f"An unresolved proposal already exists for capability: {proposal.capability_id}"
                        )
                    if existing.action_name == proposal.action_name:
                        raise CapabilityProposalStateError(
                            f"An unresolved proposal already uses action name: {proposal.action_name}"
                        )
            stored = replace(
                proposal,
                created_at=datetime.now(timezone.utc).isoformat(),
                approved_by=None,
                rejected_by=None,
                rejection_reason=None,
                registered_by=None,
                version=1,
            )
            self._proposals[stored.proposal_id] = stored
            return stored

    def create(
        self,
        *,
        capability_id: str,
        action_name: str,
        description: str,
        category,
        risk_level: str,
        permissions=(),
        preconditions=None,
        effects=None,
        parameters=None,
    ) -> CapabilityProposal:
        proposal = CapabilityProposal(
            proposal_id=uuid4().hex,
            capability_id=capability_id,
            action_name=action_name,
            description=description,
            category=category,
            risk_level=risk_level,
            permissions=tuple(permissions),
            preconditions=preconditions or {},
            effects=effects or {},
            parameters=parameters or {},
        )
        return self.submit(proposal)

    def get(self, proposal_id: str) -> CapabilityProposal:
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise CapabilityProposalStateError(f"Proposal not found: {proposal_id}")
            return proposal

    def list_all(self) -> tuple[CapabilityProposal, ...]:
        with self._lock:
            return tuple(self._proposals.values())

    def transition(
        self,
        proposal_id: str,
        *,
        expected_status: ProposalStatus,
        new_status: ProposalStatus,
        actor: str,
        reason: str | None = None,
    ) -> CapabilityProposal:
        if not isinstance(actor, str) or not actor.strip():
            raise CapabilityProposalStateError("actor must be non-empty.")
        with self._lock:
            current = self._proposals.get(proposal_id)
            if current is None:
                raise CapabilityProposalStateError(f"Proposal not found: {proposal_id}")
            if current.status != expected_status:
                raise CapabilityProposalStateError(
                    f"Expected {expected_status.value}, found {current.status.value}."
                )
            if (expected_status, new_status) not in {
                (ProposalStatus.PENDING, ProposalStatus.APPROVED),
                (ProposalStatus.PENDING, ProposalStatus.REJECTED),
                (ProposalStatus.APPROVED, ProposalStatus.REGISTERED),
            }:
                raise CapabilityProposalStateError(
                    f"Invalid proposal transition: {expected_status.value} -> {new_status.value}."
                )
            updates = {"status": new_status, "version": current.version + 1}
            if new_status == ProposalStatus.APPROVED:
                updates["approved_by"] = actor.strip()
            elif new_status == ProposalStatus.REJECTED:
                updates["rejected_by"] = actor.strip()
                updates["rejection_reason"] = (reason or "").strip() or None
            elif new_status == ProposalStatus.REGISTERED:
                updates["registered_by"] = actor.strip()
            updated = replace(current, **updates)
            self._proposals[proposal_id] = updated
            return updated


__all__ = ["CapabilityProposalStore"]
