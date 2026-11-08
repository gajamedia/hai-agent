from api.security.authorization_config import _role_map_for_keycloak


def test_keycloak_role_map_contains_execution_and_testing_permissions():
    roles = _role_map_for_keycloak("hai-agent-api")
    assert "execution:state:read" in roles["client:hai-agent-api:api-reader"]
    assert "execution:reconciliation:write" in roles["client:hai-agent-api:execution-reconciler"]
    assert "autonomous-testing:batch:create" in roles["client:hai-agent-api:autonomous-testing-operator"]
    assert "autonomous-testing:batch:run" in roles["client:hai-agent-api:autonomous-testing-operator"]
