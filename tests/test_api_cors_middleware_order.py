from fastapi.middleware.cors import CORSMiddleware

from api.main import create_app


def test_cors_is_outermost_user_middleware():
    application = create_app()

    middleware_classes = [
        middleware.cls for middleware in application.user_middleware
    ]

    assert middleware_classes[0] is CORSMiddleware
    assert middleware_classes[1].__name__ == (
        "APIAuthenticationAuthorizationMiddleware"
    )


def test_a_browser_may_send_delete_from_the_frontend_origin():
    """Disconnecting a database and cancelling a plan are DELETE requests; without this the browser reports
    "Failed to fetch" because the preflight is refused."""
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    response = client.options(
        "/api/v1/database-connections/dbsales",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "DELETE",
                 "Access-Control-Request-Headers": "authorization"})
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]
    refused = client.options(
        "/api/v1/database-connections/dbsales",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "PATCH"})
    assert refused.status_code == 400                                  # other methods stay refused
