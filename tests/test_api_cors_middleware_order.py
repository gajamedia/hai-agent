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
