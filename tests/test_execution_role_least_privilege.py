from __future__ import annotations

import json
from pathlib import Path

from api.security.authorization import API_ROUTE_PERMISSIONS
from api.security.authorization_config import _role_map_for_keycloak


ROOT = Path(__file__).resolve().parents[1]
REALM_PATH = ROOT / "keycloak" / "realm" / "hai-agent-realm.json"


def test_execution_operator_is_limited_to_action_and_plan_execution():
    roles = _role_map_for_keycloak("hai-agent-api")
    assert roles["client:hai-agent-api:execution-operator"] == (
        API_ROUTE_PERMISSIONS["execution_action_execute"],
        API_ROUTE_PERMISSIONS["execution_plan_execute"],
    )


def test_execution_state_write_has_a_separate_role():
    roles = _role_map_for_keycloak("hai-agent-api")
    assert roles["client:hai-agent-api:execution-state-operator"] == (
        API_ROUTE_PERMISSIONS["execution_state_write"],
    )


def test_execution_reset_has_a_separate_role():
    roles = _role_map_for_keycloak("hai-agent-api")
    assert roles["client:hai-agent-api:execution-reset-operator"] == (
        API_ROUTE_PERMISSIONS["execution_state_reset"],
    )


def test_execution_roles_do_not_overlap_privileged_permissions():
    roles = _role_map_for_keycloak("hai-agent-api")
    execute = set(roles["client:hai-agent-api:execution-operator"])
    state_write = set(roles["client:hai-agent-api:execution-state-operator"])
    reset = set(roles["client:hai-agent-api:execution-reset-operator"])

    assert execute.isdisjoint(state_write)
    assert execute.isdisjoint(reset)
    assert state_write.isdisjoint(reset)


def test_realm_declares_separate_execution_state_and_reset_roles():
    realm = json.loads(REALM_PATH.read_text(encoding="utf-8"))
    client_roles = {
        item["name"]: item
        for item in realm["roles"]["client"]["hai-agent-api"]
    }
    assert {
        "execution-operator",
        "execution-state-operator",
        "execution-reset-operator",
    }.issubset(client_roles)
    assert "cannot write or reset" in client_roles["execution-operator"]["description"]


def test_development_user_remains_read_scoped():
    realm = json.loads(REALM_PATH.read_text(encoding="utf-8"))
    user = next(item for item in realm["users"] if item["username"] == "hai.dev")
    assert set(user["clientRoles"]["hai-agent-api"]) == {
        "proposal-reader",
        "lifecycle-reader",
        "api-reader",
    }
