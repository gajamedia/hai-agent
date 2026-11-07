from api.security.authorization import API_ROUTE_PERMISSIONS
from api.security.authentication_middleware import required_permission_for_route
from api.security.authorization_config import _role_map_for_keycloak


def test_canonical_invocation_permission_is_declared():
    assert API_ROUTE_PERMISSIONS["canonical_capability_invoke"] == "capability:canonical:invoke"



def test_canonical_invocation_route_is_explicitly_mapped():
    assert required_permission_for_route(
        "POST", "/api/v1/trusted-capability-invocations"
    ) == "capability:canonical:invoke"


def test_keycloak_role_is_least_privilege():
    roles = _role_map_for_keycloak("hai-agent-api")
    permission = "capability:canonical:invoke"
    assert roles["client:hai-agent-api:canonical-capability-invoker"] == (permission,)
    for role, grants in roles.items():
        if role != "client:hai-agent-api:canonical-capability-invoker":
            assert permission not in grants
