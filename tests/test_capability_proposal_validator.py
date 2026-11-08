import pytest

from agent.llm.capability_proposal import (
    CapabilityProposal,
    CapabilityProposalValidationError,
    ProposalCategory,
)
from agent.llm.capability_proposal_validator import CapabilityProposalValidator


def make_proposal(**overrides):
    values = dict(
        proposal_id="proposal-1",
        capability_id="environment.observe",
        action_name="observe_environment",
        description="Observe the environment.",
        category=ProposalCategory.READ_ONLY,
        risk_level="low",
        permissions=("observe:environment",),
        effects={},
    )
    values.update(overrides)
    return CapabilityProposal(**values)


def test_valid_read_only_proposal_passes():
    proposal = make_proposal()
    assert CapabilityProposalValidator().validate(proposal) is proposal


def test_read_only_proposal_cannot_declare_effects():
    proposal = make_proposal(effects={"door": "open"})
    with pytest.raises(CapabilityProposalValidationError):
        CapabilityProposalValidator().validate(proposal)


def test_read_only_proposal_cannot_request_write_permission():
    proposal = make_proposal(permissions=("write:door",))
    with pytest.raises(CapabilityProposalValidationError):
        CapabilityProposalValidator().validate(proposal)


def test_callable_metadata_is_rejected():
    proposal = make_proposal(parameters={"handler": lambda: None})
    with pytest.raises(CapabilityProposalValidationError):
        CapabilityProposalValidator().validate(proposal)


def test_state_changing_proposal_can_describe_effects():
    proposal = make_proposal(
        category=ProposalCategory.STATE_CHANGING,
        permissions=("write:door",),
        effects={"door": "open"},
    )
    assert CapabilityProposalValidator().validate(proposal) is proposal


def test_application_managed_approval_metadata_cannot_be_submitted():
    proposal = make_proposal(approved_by="forged-admin")
    with pytest.raises(CapabilityProposalValidationError):
        CapabilityProposalValidator().validate(proposal)
