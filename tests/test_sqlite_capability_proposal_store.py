from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.llm.capability_proposal import (
    CapabilityProposal,
    CapabilityProposalStateError,
    ProposalCategory,
    ProposalStatus,
)
from agent.llm.sqlite_capability_proposal_store import SQLiteCapabilityProposalStore


def make_proposal(proposal_id="proposal-1", capability_id="cap.read", action_name="read_data"):
    return CapabilityProposal(
        proposal_id=proposal_id,
        capability_id=capability_id,
        action_name=action_name,
        description="Read application-owned data.",
        category=ProposalCategory.READ_ONLY,
        risk_level="low",
        permissions=("read:records",),
        preconditions={"source": "ready"},
        effects={},
        parameters={"limit": 10},
    )


def test_persists_across_store_instances(tmp_path):
    path = tmp_path / "proposals.sqlite3"
    first = SQLiteCapabilityProposalStore(path)
    saved = first.submit(make_proposal())
    second = SQLiteCapabilityProposalStore(path)
    loaded = second.get(saved.proposal_id)
    assert loaded == saved
    assert loaded.preconditions["source"] == "ready"
    assert loaded.parameters["limit"] == 10


def test_transition_metadata_survives_restart(tmp_path):
    path = tmp_path / "proposals.sqlite3"
    store = SQLiteCapabilityProposalStore(path)
    store.submit(make_proposal())
    approved = store.transition(
        "proposal-1",
        expected_status=ProposalStatus.PENDING,
        new_status=ProposalStatus.APPROVED,
        actor="admin-1",
    )
    assert approved.approved_by == "admin-1"
    reopened = SQLiteCapabilityProposalStore(path).get("proposal-1")
    assert reopened.status is ProposalStatus.APPROVED
    assert reopened.version == 2
    assert reopened.approved_by == "admin-1"


def test_rejected_proposal_releases_unique_capability_and_action(tmp_path):
    store = SQLiteCapabilityProposalStore(tmp_path / "proposals.sqlite3")
    store.submit(make_proposal())
    store.transition(
        "proposal-1",
        expected_status=ProposalStatus.PENDING,
        new_status=ProposalStatus.REJECTED,
        actor="admin-1",
        reason="Not needed",
    )
    replacement = make_proposal("proposal-2")
    assert store.submit(replacement).status is ProposalStatus.PENDING


def test_unresolved_capability_and_action_names_are_unique(tmp_path):
    store = SQLiteCapabilityProposalStore(tmp_path / "proposals.sqlite3")
    store.submit(make_proposal())
    with pytest.raises(CapabilityProposalStateError):
        store.submit(make_proposal("proposal-2", capability_id="cap.other"))
    with pytest.raises(CapabilityProposalStateError):
        store.submit(
            make_proposal(
                "proposal-3",
                capability_id="cap.other",
                action_name="read_data",
            )
        )


def test_invalid_transition_does_not_change_record(tmp_path):
    store = SQLiteCapabilityProposalStore(tmp_path / "proposals.sqlite3")
    store.submit(make_proposal())
    with pytest.raises(CapabilityProposalStateError):
        store.transition(
            "proposal-1",
            expected_status=ProposalStatus.PENDING,
            new_status=ProposalStatus.REGISTERED,
            actor="admin-1",
        )
    assert store.get("proposal-1").version == 1


def test_list_all_returns_persisted_records_in_stable_order(tmp_path):
    store = SQLiteCapabilityProposalStore(tmp_path / "proposals.sqlite3")
    store.submit(make_proposal())
    store.submit(make_proposal("proposal-2", "cap.second", "read_second"))
    assert [item.proposal_id for item in store.list_all()] == [
        "proposal-1",
        "proposal-2",
    ]


def test_concurrent_duplicate_submission_only_one_wins(tmp_path):
    path = tmp_path / "proposals.sqlite3"
    proposals = [make_proposal(f"proposal-{index}") for index in range(8)]

    def submit(proposal):
        try:
            return SQLiteCapabilityProposalStore(path).submit(proposal).proposal_id
        except CapabilityProposalStateError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(submit, proposals))

    assert sum(value != "conflict" for value in outcomes) == 1
    assert len(SQLiteCapabilityProposalStore(path).list_all()) == 1


def test_concurrent_store_initialization_preserves_all_distinct_submissions(tmp_path):
    path = tmp_path / "parallel-startup" / "proposals.sqlite3"
    proposals = [
        make_proposal(
            f"parallel-{index}",
            capability_id=f"cap.parallel.{index}",
            action_name=f"read_parallel_{index}",
        )
        for index in range(8)
    ]

    def initialize_and_submit(proposal):
        store = SQLiteCapabilityProposalStore(path, timeout=15.0)
        return store.submit(proposal).proposal_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(initialize_and_submit, proposals))

    assert set(outcomes) == {proposal.proposal_id for proposal in proposals}
    assert len(SQLiteCapabilityProposalStore(path).list_all()) == 8


def test_nested_metadata_is_immutable_after_read(tmp_path):
    store = SQLiteCapabilityProposalStore(tmp_path / "proposals.sqlite3")
    store.submit(make_proposal())
    loaded = store.get("proposal-1")
    with pytest.raises(TypeError):
        loaded.preconditions["source"] = "tampered"


def test_create_generates_identifier_and_persists(tmp_path):
    store = SQLiteCapabilityProposalStore(tmp_path / "proposals.sqlite3")
    created = store.create(
        capability_id="cap.created",
        action_name="read_created",
        description="Read created data.",
        category=ProposalCategory.READ_ONLY,
        risk_level="low",
        permissions=("read:created",),
    )
    assert created.proposal_id
    assert store.get(created.proposal_id) == created
