"""The Keycloak roles must grant exactly what the data-report routes demand.

Without this mapping every dashboard user would get 403, because the policy maps
roles to permissions through an exact table.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.database.capability_gateway import DEFAULT_PERMISSION as GATEWAY_READ
from api.routes import database_connections, database_engine_requests, table_reports
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationDenied
from api.security.authorization_config import create_environment_authorization_policy

REALM = Path(__file__).resolve().parents[1] / "keycloak" / "realm" / "hai-agent-realm.json"
CLIENT = "hai-agent-api"
ENV = {"HAI_AUTH_ENABLED": "true", "HAI_AUTH_PROVIDER": "keycloak",
       "HAI_KEYCLOAK_API_CLIENT_ID": CLIENT}
USER, MANAGER, REVIEWER = ("data-reports-user", "database-connection-manager",
                           "database-engine-reviewer")

RUN = table_reports._PERMISSION
MANAGE = database_connections.PERMISSION
ASK = database_engine_requests.REQUEST_PERMISSION
REVIEW = database_engine_requests.REVIEW_PERMISSION


def _principal(*roles: str) -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        actor_id="u", roles=tuple(f"client:{CLIENT}:{r}" for r in roles), permissions=(),
        issuer="test", subject="u", is_authenticated=True)


def _allowed(policy, principal, permission) -> bool:
    try:
        policy.require(principal, permission)
        return True
    except AuthorizationDenied:
        return False


@pytest.fixture(scope="module")
def policy():
    return create_environment_authorization_policy(environ=ENV)


def test_a_report_user_can_run_reports_read_data_and_ask_for_engines(policy):
    p = _principal(USER)
    assert all(_allowed(policy, p, perm) for perm in (RUN, GATEWAY_READ, ASK))


def test_a_report_user_cannot_manage_connections_or_review_requests(policy):
    p = _principal(USER)
    assert not _allowed(policy, p, MANAGE) and not _allowed(policy, p, REVIEW)


def test_each_management_role_grants_only_its_own_permission(policy):
    assert _allowed(policy, _principal(MANAGER), MANAGE)
    assert not any(_allowed(policy, _principal(MANAGER), x) for x in (RUN, GATEWAY_READ, ASK, REVIEW))
    # a reviewer can review and also list requests (the central gate maps GET to "ask")
    assert _allowed(policy, _principal(REVIEWER), REVIEW) and _allowed(policy, _principal(REVIEWER), ASK)
    assert not any(_allowed(policy, _principal(REVIEWER), x) for x in (RUN, GATEWAY_READ, MANAGE))


def test_a_manager_alone_cannot_read_business_data(policy):
    assert not _allowed(policy, _principal(MANAGER), GATEWAY_READ)


def test_a_role_from_another_client_grants_nothing(policy):
    foreign = AuthenticatedPrincipal(
        actor_id="u", roles=(f"client:other-client:{USER}", USER), permissions=(),
        issuer="test", subject="u", is_authenticated=True)
    assert not any(_allowed(policy, foreign, x) for x in (RUN, GATEWAY_READ, ASK, MANAGE, REVIEW))


def test_the_realm_declares_the_roles_the_policy_maps():
    realm = json.loads(REALM.read_text(encoding="utf-8"))
    declared = {r["name"] for r in realm["roles"]["client"][CLIENT]}
    assert {USER, MANAGER, REVIEWER} <= declared


def test_the_realm_keeps_the_dev_user_read_only_so_data_roles_are_granted_explicitly():
    realm = json.loads(REALM.read_text(encoding="utf-8"))
    dev = next(u for u in realm["users"] if u["username"] == "hai.dev")
    assert not ({USER, MANAGER, REVIEWER} & set(dev["clientRoles"][CLIENT]))


def test_every_route_permission_string_is_in_the_central_catalog():
    from api.security.authorization import API_ROUTE_PERMISSIONS
    values = set(API_ROUTE_PERMISSIONS.values())
    assert {RUN, MANAGE, ASK, REVIEW, GATEWAY_READ} <= values


def test_the_realm_clients_include_the_basic_scope_that_carries_the_sub_claim():
    """Keycloak 25+ puts `sub` in the `basic` scope; the API rejects tokens without it."""
    realm = json.loads(REALM.read_text(encoding="utf-8"))
    for client in realm["clients"]:
        assert "basic" in client["defaultClientScopes"], client["clientId"]


WRITER, APPROVER = "data-writer", "data-change-approver"
PLAN_P, EXEC_P, APPROVE_P = "data:change:plan", "data:change:execute", "data:change:approve"


def test_changing_data_is_its_own_pair_of_roles_and_reading_never_grants_it(policy):
    assert all(_allowed(policy, _principal(WRITER), p) for p in (PLAN_P, EXEC_P))
    assert not _allowed(policy, _principal(WRITER), APPROVE_P)           # a writer cannot approve
    assert _allowed(policy, _principal(APPROVER), APPROVE_P)
    assert not any(_allowed(policy, _principal(APPROVER), p) for p in (PLAN_P, EXEC_P))
    for role in (USER, MANAGER, REVIEWER):
        assert not any(_allowed(policy, _principal(role), p) for p in (PLAN_P, EXEC_P, APPROVE_P))


def test_the_realm_declares_the_data_change_roles():
    realm = json.loads(REALM.read_text(encoding="utf-8"))
    declared = {r["name"] for r in realm["roles"]["client"][CLIENT]}
    assert {WRITER, APPROVER} <= declared
    dev = next(u for u in realm["users"] if u["username"] == "hai.dev")
    assert not ({WRITER, APPROVER} & set(dev["clientRoles"][CLIENT]))
