from agent.llm.capability_proposal import (
    CapabilityProposal,
    CapabilityProposalStateError,
    ProposalCategory,
    ProposalStatus,
)
from agent.llm.capability_proposal_store import CapabilityProposalStore


def test_store_creates_pending_proposal():
    store = CapabilityProposalStore()
    proposal = store.create(
        capability_id="environment.observe",
        action_name="observe_environment",
        description="Observe the current environment.",
        category=ProposalCategory.READ_ONLY,
        risk_level="low",
        permissions=("observe:environment",),
    )
    assert proposal.status == ProposalStatus.PENDING
    assert store.get(proposal.proposal_id) is proposal


def test_proposal_metadata_is_copied_and_nested_values_are_immutable():
    source = {"filters": {"room": "main"}, "fields": ["door", "box"]}
    proposal = CapabilityProposal(
        proposal_id="proposal-1",
        capability_id="environment.observe",
        action_name="observe_environment",
        description="Observe environment.",
        category=ProposalCategory.READ_ONLY,
        risk_level="low",
        parameters=source,
    )
    source["filters"]["room"] = "changed"
    assert proposal.parameters["filters"]["room"] == "main"
    assert proposal.parameters["fields"] == ("door", "box")


def test_registered_proposal_reserves_its_capability_and_action_name():
    store = CapabilityProposalStore()
    proposal = store.create(
        capability_id="environment.observe",
        action_name="observe_environment",
        description="Observe the current environment.",
        category=ProposalCategory.READ_ONLY,
        risk_level="low",
        permissions=("observe:environment",),
    )
    store.transition(
        proposal.proposal_id,
        expected_status=ProposalStatus.PENDING,
        new_status=ProposalStatus.APPROVED,
        actor="admin",
    )
    store.transition(
        proposal.proposal_id,
        expected_status=ProposalStatus.APPROVED,
        new_status=ProposalStatus.REGISTERED,
        actor="admin",
    )
    import pytest

    with pytest.raises(CapabilityProposalStateError):
        store.create(
            capability_id="environment.observe",
            action_name="observe_environment",
            description="Duplicate proposal.",
            category=ProposalCategory.READ_ONLY,
            risk_level="low",
            permissions=("observe:environment",),
        )
