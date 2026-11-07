"""Grant (or revoke) the data-report roles for a user in a running Keycloak.

The realm file deliberately keeps the local dev user read-only, so the roles that let
someone connect databases and run reports are granted explicitly with this script:

    $env:KC_ADMIN_PASSWORD = "<your Keycloak admin password>"
    python scripts/keycloak_data_report_roles.py --user hai.dev

Roles (client "hai-agent-api"):
    data-reports-user            run reports, read approved data, request new engines
    database-connection-manager  connect databases and approve tables
    database-engine-reviewer     review requests for new engines (not granted by default)

Idempotent. The admin password is read from KC_ADMIN_PASSWORD (or prompted) and is never
printed. Plain HTTP is accepted only for loopback addresses.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

DEFAULT_ROLES = ("data-reports-user", "database-connection-manager")
KNOWN_ROLES = {
    "data-reports-user": "Create and run database reports and request new database engines.",
    "database-connection-manager": "Connect databases and approve which tables the agent may use.",
    "database-engine-reviewer": "Review requests for new database engines.",
    "data-writer": "Plan and execute data changes (insert, update, delete) after re-confirming identity.",
    "data-change-approver": "Approve data changes made by someone else (second pair of eyes).",
}
LOOPBACK = {"localhost", "127.0.0.1", "::1"}

Http = Callable[[str, str, "dict[str, str] | None", Any], tuple[int, Any]]


class KeycloakAdminError(RuntimeError):
    pass


def _real_http(method: str, url: str, headers: dict[str, str] | None, body: Any) -> tuple[int, Any]:
    data = None
    request_headers = dict(headers or {})
    if isinstance(body, dict) and request_headers.get("Content-Type") == "application/x-www-form-urlencoded":
        data = urllib.parse.urlencode(body).encode()
    elif body is not None:
        data = json.dumps(body).encode()
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, (json.loads(raw) if raw else None)
        except ValueError:
            return exc.code, None


def _check_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise KeycloakAdminError("base URL must be http(s)://host[:port]")
    if parsed.scheme == "http" and parsed.hostname not in LOOPBACK:
        raise KeycloakAdminError("plain HTTP is only allowed for localhost; use https")
    return base_url.rstrip("/")


def run(*, base_url: str, realm: str, client_id: str, username: str, roles: list[str],
        admin_user: str, admin_password: str, revoke: bool = False, http: Http = _real_http) -> dict:
    base = _check_url(base_url)
    unknown = [r for r in roles if r not in KNOWN_ROLES]
    if unknown:
        raise KeycloakAdminError(f"unknown role(s): {', '.join(unknown)}")

    status, token = http("POST", f"{base}/realms/master/protocol/openid-connect/token",
                         {"Content-Type": "application/x-www-form-urlencoded"},
                         {"grant_type": "password", "client_id": "admin-cli",
                          "username": admin_user, "password": admin_password})
    if status != 200 or not isinstance(token, dict) or "access_token" not in token:
        raise KeycloakAdminError("could not sign in as the Keycloak admin (check user/password)")
    auth = {"Authorization": f"Bearer {token['access_token']}"}
    admin = f"{base}/admin/realms/{urllib.parse.quote(realm)}"

    status, clients = http("GET", f"{admin}/clients?clientId={urllib.parse.quote(client_id)}", auth, None)
    if status != 200 or not clients:
        raise KeycloakAdminError(f"client {client_id!r} was not found in realm {realm!r}")
    client_uuid = clients[0]["id"]

    status, users = http("GET", f"{admin}/users?username={urllib.parse.quote(username)}&exact=true", auth, None)
    if status != 200 or not users:
        raise KeycloakAdminError(f"user {username!r} was not found in realm {realm!r}")
    user_id = users[0]["id"]

    created, representations = [], []
    for name in roles:
        status, role = http("GET", f"{admin}/clients/{client_uuid}/roles/{urllib.parse.quote(name)}", auth, None)
        if status == 404 and not revoke:
            status, _ = http("POST", f"{admin}/clients/{client_uuid}/roles", auth,
                             {"name": name, "description": KNOWN_ROLES[name]})
            if status not in (201, 204):
                raise KeycloakAdminError(f"could not create role {name!r}")
            created.append(name)
            status, role = http("GET", f"{admin}/clients/{client_uuid}/roles/{urllib.parse.quote(name)}",
                                auth, None)
        if status == 200 and role:
            representations.append({"id": role["id"], "name": role["name"]})

    mapping = f"{admin}/users/{user_id}/role-mappings/clients/{client_uuid}"
    if representations:
        status, _ = http("DELETE" if revoke else "POST", mapping, auth, representations)
        if status not in (200, 204):
            raise KeycloakAdminError("could not update the user's roles")
    status, current = http("GET", mapping, auth, None)
    return {"user": username, "created_roles": created, "revoked": revoke,
            "roles_now": sorted(r["name"] for r in (current or []))}


def main(argv: list[str] | None = None, http: Http = _real_http) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default=os.getenv("KC_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--realm", default="hai-agent")
    parser.add_argument("--client-id", default="hai-agent-api")
    parser.add_argument("--user", required=True, help="Keycloak username to change")
    parser.add_argument("--roles", default=",".join(DEFAULT_ROLES),
                        help=f"comma separated; known: {', '.join(KNOWN_ROLES)}")
    parser.add_argument("--admin-user", default=os.getenv("KC_ADMIN_USER", "admin"))
    parser.add_argument("--revoke", action="store_true", help="remove the roles instead of granting")
    args = parser.parse_args(argv)
    password = os.getenv("KC_ADMIN_PASSWORD") or getpass.getpass("Keycloak admin password: ")
    try:
        result = run(base_url=args.base_url, realm=args.realm, client_id=args.client_id,
                     username=args.user, roles=[r.strip() for r in args.roles.split(",") if r.strip()],
                     admin_user=args.admin_user, admin_password=password, revoke=args.revoke, http=http)
    except KeycloakAdminError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    action = "revoked from" if result["revoked"] else "granted to"
    print(f"Roles {action} {result['user']}. Created roles: {result['created_roles'] or 'none'}.")
    print(f"{result['user']} now has: {', '.join(result['roles_now']) or '(no client roles)'}")
    print("Sign out and sign in again in the dashboard so the new roles appear in your token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
