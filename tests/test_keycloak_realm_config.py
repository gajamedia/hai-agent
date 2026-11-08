from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REALM_PATH = ROOT / "keycloak" / "realm" / "hai-agent-realm.json"


def load_realm():
    return json.loads(REALM_PATH.read_text(encoding="utf-8"))


def test_realm_import_declares_expected_realm_and_two_clients():
    realm = load_realm()
    assert realm["realm"] == "hai-agent"
    clients = {item["clientId"]: item for item in realm["clients"]}
    assert set(clients) == {"hai-agent-api", "hai-agent-web"}


def test_frontend_client_uses_pkce_s256_and_no_password_grant():
    realm = load_realm()
    web = next(item for item in realm["clients"] if item["clientId"] == "hai-agent-web")
    assert web["publicClient"] is True
    assert web["standardFlowEnabled"] is True
    assert web["implicitFlowEnabled"] is False
    assert web["directAccessGrantsEnabled"] is False
    assert web["attributes"]["pkce.code.challenge.method"] == "S256"


def test_frontend_access_token_contains_backend_audience():
    realm = load_realm()
    web = next(item for item in realm["clients"] if item["clientId"] == "hai-agent-web")
    audience = [
        mapper for mapper in web["protocolMappers"]
        if mapper["protocolMapper"] == "oidc-audience-mapper"
    ]
    assert len(audience) == 1
    assert audience[0]["config"]["included.client.audience"] == "hai-agent-api"
    assert audience[0]["config"]["access.token.claim"] == "true"
    assert audience[0]["config"]["id.token.claim"] == "false"


def test_api_resource_client_has_no_browser_or_password_grant():
    realm = load_realm()
    api = next(item for item in realm["clients"] if item["clientId"] == "hai-agent-api")
    assert api["standardFlowEnabled"] is False
    assert api["implicitFlowEnabled"] is False
    assert api["directAccessGrantsEnabled"] is False
    assert api["serviceAccountsEnabled"] is False


def test_development_user_password_is_environment_placeholder():
    realm = load_realm()
    user = next(item for item in realm["users"] if item["username"] == "hai.dev")
    assert user["credentials"][0]["value"] == "${HAI_KEYCLOAK_DEV_USER_PASSWORD}"
    assert user["credentials"][0]["temporary"] is True


def test_development_user_has_only_read_scoped_client_roles():
    realm = load_realm()
    user = next(item for item in realm["users"] if item["username"] == "hai.dev")
    roles = set(user["clientRoles"]["hai-agent-api"])

    # api-reader grants only read permissions in authorization_config.py.
    # The developer account may inspect API state, but receives no operator,
    # reconciliation, proposal mutation, or lifecycle mutation role.
    assert roles == {
        "proposal-reader",
        "lifecycle-reader",
        "api-reader",
    }
    from api.security.authorization import API_ROUTE_PERMISSIONS
    from api.security.authorization_config import _role_map_for_keycloak

    role_permissions = _role_map_for_keycloak("hai-agent-api")
    assert role_permissions["client:hai-agent-api:api-reader"] == (
        API_ROUTE_PERMISSIONS["execution_state_read"],
        API_ROUTE_PERMISSIONS["strategy_registry_read"],
        API_ROUTE_PERMISSIONS["telemetry_capability_read"],
        API_ROUTE_PERMISSIONS["autonomous_batch_read"],
    )

    assert roles.isdisjoint({
        "execution-operator",
        "execution-reconciler",
        "cognitive-operator",
        "autonomous-testing-operator",
        "autonomous-testing-reconciler",
        "proposal-submitter",
        "proposal-approver",
        "proposal-rejector",
        "proposal-register",
        "lifecycle-activator",
        "lifecycle-disabler",
        "lifecycle-revoker",
    })


def test_keycloak_compose_binds_only_to_loopback_and_uses_pinned_image():
    compose_text = (ROOT / "docker-compose.keycloak.yml").read_text(encoding="utf-8")
    assert "quay.io/keycloak/keycloak:26.8.1" in compose_text
    assert '"127.0.0.1:8080:8080"' in compose_text
    assert "start-dev" in compose_text
    assert "--import-realm" in compose_text
