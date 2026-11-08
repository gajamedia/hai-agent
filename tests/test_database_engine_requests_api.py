from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from tests.test_table_reports_api import _Attach

URL = "/api/v1/database-engine-requests"
ASK, REVIEW = "database:engine:request", "database:engine:review"


def _who(actor, *permissions):
    return AuthenticatedPrincipal(actor_id=actor, roles=(), permissions=permissions,
                                  issuer="test", subject=actor, is_authenticated=True)


@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    registry = DatabaseRuntimeRegistry(
        data_root=data, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: None)
    clients = []

    def client(actor, *permissions):
        app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                         database_runtime_registry_factory=lambda _a: registry)
        app.add_middleware(_Attach, principal=_who(actor, *permissions))
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c
    yield client
    for c in clients:
        c.__exit__(None, None, None)
    registry.shutdown()


def _ask(c, engine="Oracle 19c", reason="We run finance on it"):
    return c.post(URL, json={"engine": engine, "reason": reason})


def test_a_user_can_request_an_engine_and_sees_only_their_own(world):
    alice, bob = world("alice", ASK), world("bob", ASK)
    created = _ask(alice)
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "open" and body["requested_by"] == "alice" and body["engine"] == "Oracle 19c"
    _ask(bob, "SQL Server")
    mine = alice.get(URL).json()
    assert mine["scope"] == "own" and [r["requested_by"] for r in mine["requests"]] == ["alice"]


def test_reviewers_see_everything_and_can_move_a_request_through_its_life(world):
    alice, admin = world("alice", ASK), world("root", REVIEW)
    rid = _ask(alice).json()["id"]
    everything = admin.get(URL).json()
    assert everything["scope"] == "all" and len(everything["requests"]) == 1
    ok = admin.patch(f"{URL}/{rid}", json={"status": "accepted", "note": "Planned for Q1"})
    assert ok.status_code == 200 and ok.json()["reviewed_by"] == "root" and ok.json()["note"] == "Planned for Q1"
    assert admin.patch(f"{URL}/{rid}", json={"status": "delivered"}).json()["status"] == "delivered"
    assert admin.patch(f"{URL}/{rid}", json={"status": "rejected"}).status_code == 409  # final


def test_a_requester_cannot_review_their_own_request(world):
    alice = world("alice", ASK)
    rid = _ask(alice).json()["id"]
    assert alice.patch(f"{URL}/{rid}", json={"status": "accepted"}).status_code == 403


def test_permissions_and_authentication_are_enforced(world, tmp_path):
    assert _ask(world("nobody")).status_code == 403
    assert world("nobody").get(URL).status_code == 403
    assert world("nobody").patch(f"{URL}/x", json={"status": "accepted"}).status_code == 403
    app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                     database_runtime_registry_factory=lambda _a: DatabaseRuntimeRegistry(
                         data_root=tmp_path, work_dir=tmp_path / "w2",
                         authorization_policy=AuthorizationPolicy(), llm_provider_factory=lambda: None))
    with TestClient(app) as anonymous:  # no principal attached
        assert anonymous.post(URL, json={"engine": "Oracle", "reason": "x"}).status_code == 401


def test_unconfigured_server_answers_503():
    app = create_app()
    app.add_middleware(_Attach, principal=_who("alice", ASK))
    with TestClient(app) as c:
        assert c.get(URL).status_code in (503,)


@pytest.mark.parametrize("body", [
    {}, {"engine": "Oracle"}, {"reason": "x"}, {"engine": 5, "reason": "x"},
    {"engine": "Oracle", "reason": "   "}, {"engine": "x" * 41, "reason": "x"},
    {"engine": "Ora<script>", "reason": "x"}, {"engine": "Oracle", "reason": "x" * 501},
    {"engine": "Oracle", "reason": "bad\x00char"}, {"engine": "Oracle", "reason": "x", "extra": 1},
    {"engine": "Oracle", "reason": "x", "status": "accepted"}])
def test_invalid_requests_are_rejected(world, body):
    assert world("alice", ASK).post(URL, json=body).status_code == 422


def test_already_supported_engines_are_refused_with_conflict(world):
    alice = world("alice", ASK)
    for name in ("PostgreSQL", "postgres", "MYSQL", "mariadb", "SQLite"):
        assert _ask(alice, name).status_code == 409


def test_open_requests_are_capped_per_user(world):
    alice = world("alice", ASK)
    for i in range(5):
        assert _ask(alice, f"Engine {i}").status_code == 201
    assert _ask(alice, "Engine 6").status_code == 409
    assert _ask(world("bob", ASK), "Engine 6").status_code == 201  # per user, not global


def test_text_is_stored_verbatim_as_data_and_never_interpreted(world):
    alice = world("alice", ASK)
    note = "'; DROP TABLE engine_requests; -- {{7*7}} ${jndi:x} 管理"
    assert _ask(alice, "Oracle 19c", note).status_code == 201
    assert alice.get(URL).json()["requests"][0]["reason"] == note
    assert _ask(alice, "Another One", "still works").status_code == 201


@pytest.mark.parametrize("body,status", [
    ({"status": "open"}, 422), ({"status": "weird"}, 422), ({}, 422), ({"status": "accepted", "x": 1}, 422)])
def test_review_input_is_validated(world, body, status):
    alice, admin = world("alice", ASK), world("root", REVIEW)
    rid = _ask(alice).json()["id"]
    assert admin.patch(f"{URL}/{rid}", json=body).status_code == status


def test_reviewing_an_unknown_id_is_404_and_skipping_a_step_is_409(world):
    alice, admin = world("alice", ASK), world("root", REVIEW)
    assert admin.patch(f"{URL}/nope", json={"status": "accepted"}).status_code == 404
    rid = _ask(alice).json()["id"]
    assert admin.patch(f"{URL}/{rid}", json={"status": "delivered"}).status_code == 409  # not accepted yet
