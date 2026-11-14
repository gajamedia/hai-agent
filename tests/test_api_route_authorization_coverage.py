"""Every API route must have an explicit authorization contract.

With HAI_AUTH_ENABLED=true the middleware answers 403 `authorization_route_unmapped` for
any /api/v1 route missing from its table, so a new route that is not listed there works
in tests (which attach a principal directly) and then fails in a real deployment. This
test fails the build instead.
"""
from __future__ import annotations

import re

import pytest

from api.main import create_app
from api.security.authentication_middleware import _CAPABILITY_PREFIXES, required_permission_for_route

# Routes that intentionally carry no central permission. Capability routes do their own
# per-operation checks inside the handler (see _CAPABILITY_PREFIXES).
EXEMPT_PREFIXES = _CAPABILITY_PREFIXES


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "some-id", path)


def _api_routes():
    """Every (method, path) under /api/v1, read from the OpenAPI schema.

    Not from ``app.routes``: newer FastAPI wraps included routers in ``_IncludedRouter`` objects, so
    walking ``app.routes`` for ``APIRoute`` finds none and this guard would pass without checking anything.
    """
    paths = create_app().openapi()["paths"]
    for path, operations in paths.items():
        if path.startswith("/api/v1"):
            for method in sorted(m.upper() for m in operations if m.upper() in _METHODS):
                yield method, path


_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def test_the_guard_actually_sees_the_api():
    """If route discovery silently finds nothing, every other test here passes for no reason."""
    routes = set(_api_routes())
    assert len(routes) > 40
    for expected in [("POST", "/api/v1/table-reports/plans"), ("POST", "/api/v1/data-changes/plans"),
                     ("GET", "/api/v1/tasks"), ("GET", "/api/v1/tasks/{task_id}")]:
        assert expected in routes


def test_every_non_capability_api_route_is_mapped_to_a_permission():
    unmapped = [
        f"{method} {path}" for method, path in _api_routes()
        if not path.startswith(EXEMPT_PREFIXES)
        and required_permission_for_route(method, _concrete(path)) is None
    ]
    assert not unmapped, "routes without an authorization contract: " + ", ".join(unmapped)


def test_the_data_report_routes_are_checked_against_the_expected_permissions():
    expected = {
        ("POST", "/api/v1/table-reports"): "report:table:run",
        ("POST", "/api/v1/table-reports/plans"): "report:table:run",
        ("POST", "/api/v1/table-reports/plans/p1/run"): "report:table:run",
        ("DELETE", "/api/v1/table-reports/plans/p1"): "report:table:run",
        ("GET", "/api/v1/table-reports/reports/w1/download"): "report:table:run",
        ("POST", "/api/v1/database-connections"): "database:connection:manage",
        ("GET", "/api/v1/database-connections"): "database:connection:manage",
        ("GET", "/api/v1/database-connections/profiles"): "database:connection:manage",
        ("GET", "/api/v1/database-connections/sales/schema"): "database:connection:manage",
        ("POST", "/api/v1/database-connections/sales/approve"): "database:connection:manage",
        ("DELETE", "/api/v1/database-connections/sales"): "database:connection:manage",
        ("POST", "/api/v1/database-connections/servers"): "database:connection:manage",
        ("POST", "/api/v1/database-connections/servers/inspect"): "database:connection:manage",
        ("POST", "/api/v1/database-connections/servers/auto"): "database:connection:manage",
        ("DELETE", "/api/v1/database-connections/servers/srv"): "database:connection:manage",
        ("POST", "/api/v1/database-engine-requests"): "database:engine:request",
        ("GET", "/api/v1/database-engine-requests"): "database:engine:request",
        ("PATCH", "/api/v1/database-engine-requests/r1"): "database:engine:review",
    }
    for (method, path), permission in expected.items():
        assert required_permission_for_route(method, path) == permission, (method, path)


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/v1/table-reports"), ("PUT", "/api/v1/table-reports/plans"),
    ("GET", "/api/v1/database-connections/a/b/schema"),
    ("DELETE", "/api/v1/database-engine-requests/r1"),
    ("POST", "/api/v1/table-reports/plans/p1/run/extra"),
    ("GET", "/api/v1/table-reports/reports/../../secret/download"),
])
def test_lookalike_paths_and_methods_stay_unmapped(method, path):
    assert required_permission_for_route(method, path) is None
