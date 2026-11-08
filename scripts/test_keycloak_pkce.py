from __future__ import annotations

import argparse
import base64
import hashlib
import json
import queue
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


KEYCLOAK_BASE_URL = "http://localhost:8080"
REALM = "hai-agent"
CLIENT_ID = "hai-agent-web"
REDIRECT_URI = "http://127.0.0.1:8765/callback"

API_BASE_URL = "http://127.0.0.1:8000"
CALLBACK_TIMEOUT = 180
REQUEST_TIMEOUT = 15

# The identifiers are deliberately synthetic. PBAC must reject the request
# before looking up the resource. No real proposal/capability is modified.
PROBE_ID = "pbac-diagnostic-nonexistent-resource"


def base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def create_pkce_pair() -> tuple[str, str]:
    verifier = base64url(secrets.token_bytes(32))
    challenge = base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


class CallbackHandler(BaseHTTPRequestHandler):
    result_queue: queue.Queue | None = None
    expected_state: str | None = None

    def log_message(self, format: str, *args: Any) -> None:
        # Never log authorization codes or callback query strings.
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_error(404)
            return

        params = urllib.parse.parse_qs(parsed.query)
        returned_state = params.get("state", [""])[0]
        code = params.get("code", [""])[0]
        error = params.get("error", [""])[0]

        if returned_state != self.expected_state:
            self._respond(400, b"Invalid OAuth state.")
            if self.result_queue is not None:
                self.result_queue.put({"error": "OAuth state mismatch"})
            return

        if error:
            # Do not expose arbitrary provider error descriptions in logs.
            self._respond(400, b"Keycloak authorization failed.")
            if self.result_queue is not None:
                self.result_queue.put({"error": "Keycloak authorization failed"})
            return

        if not code:
            self._respond(400, b"Authorization code missing.")
            if self.result_queue is not None:
                self.result_queue.put({"error": "Authorization code missing"})
            return

        self._respond(
            200,
            b"""<!doctype html><html><body style="font-family:Arial;text-align:center;padding:50px">
            <h2>Keycloak authentication successful</h2>
            <p>You may close this browser window.</p></body></html>""",
            content_type="text/html; charset=utf-8",
        )
        if self.result_queue is not None:
            self.result_queue.put({"code": code})

    def _respond(
        self,
        status: int,
        body: bytes,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def exchange_code(
    code: str,
    verifier: str,
    token_endpoint: str,
    client_id: str,
) -> dict[str, Any]:
    data = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        token_endpoint,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Token endpoint returned an invalid response.")
    return payload


def decode_jwt_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None

    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    return claims if isinstance(claims, dict) else None


def inspect_jwt(token: str, token_name: str) -> dict[str, Any] | None:
    print(f"\n=== {token_name} ===")
    claims = decode_jwt_payload(token)
    if claims is None:
        print("JWT payload could not be decoded.")
        return None

    subject = claims.get("sub")
    checks = {
        "sub_exists": "sub" in claims,
        "sub_is_string": isinstance(subject, str),
        "sub_non_empty": isinstance(subject, str) and bool(subject.strip()),
    }
    for name, result in checks.items():
        print(f"{name}: {result}")

    print("Claim names:", sorted(claims.keys()))
    for field in ("iss", "aud", "azp", "typ", "exp", "iat"):
        if field in claims:
            print(f"{field}: {claims[field]}")

    return claims


def inspect_access_token_roles(claims: dict[str, Any] | None) -> None:
    print("\n=== ACCESS TOKEN ROLE DIAGNOSTIC ===")
    if claims is None:
        print("Unavailable.")
        return

    realm_access = claims.get("realm_access")
    if not isinstance(realm_access, dict):
        realm_access = {}
    resource_access = claims.get("resource_access")
    if not isinstance(resource_access, dict):
        resource_access = {}
    api_access = resource_access.get("hai-agent-api")
    if not isinstance(api_access, dict):
        api_access = {}

    print("Realm roles:", realm_access.get("roles", []))
    print("API client roles:", api_access.get("roles", []))


def inspect_userinfo(access_token: str, userinfo_endpoint: str) -> bool:
    print("\n=== USERINFO ===")
    request = urllib.request.Request(
        userinfo_endpoint,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        print("HTTP Status:", exc.code)
        print("UserInfo rejected the request.")
        return False
    except urllib.error.URLError as exc:
        print("UserInfo connection failed:", exc.reason)
        return False

    print("HTTP Status:", status)
    try:
        claims = json.loads(body)
    except json.JSONDecodeError:
        print("UserInfo returned invalid JSON.")
        return False

    subject = claims.get("sub") if isinstance(claims, dict) else None
    valid = isinstance(subject, str) and bool(subject.strip())
    print("sub exists and is non-empty:", valid)
    if isinstance(claims, dict):
        print("Claim names:", sorted(claims.keys()))
    # Do not print subject, email, name, or other personal identity values.
    return status == 200 and valid


def request_api(
    method: str,
    url: str,
    access_token: str | None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    data = None

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return 0, {"transport_error": str(exc.reason)}

    try:
        parsed: Any = json.loads(body) if body else {}
    except json.JSONDecodeError:
        parsed = {"raw_response": body[:1000]}
    return status, parsed


def error_code(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if isinstance(detail, dict):
        value = detail.get("error")
        return value if isinstance(value, str) else None
    value = body.get("error")
    return value if isinstance(value, str) else None


def run_api_case(
    name: str,
    method: str,
    path: str,
    token: str | None,
    expected_statuses: tuple[int, ...],
    *,
    payload: dict[str, Any] | None = None,
    blocked_statuses: tuple[int, ...] = (),
) -> str:
    url = API_BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    status, body = request_api(method, url, token, payload)
    code = error_code(body)

    if status in expected_statuses:
        result = "PASS"
    elif status in blocked_statuses:
        result = "BLOCKED"
    else:
        result = "FAIL"

    suffix = f" error={code}" if code else ""
    print(
        f"[{result:7}] {method:4} {path} "
        f"-> HTTP {status}{suffix}"
    )
    return result


def run_pbAC_diagnostics(api_base_url: str, access_token: str) -> tuple[int, int, int]:
    global API_BASE_URL
    API_BASE_URL = api_base_url

    print("\n========================================")
    print("PBAC END-TO-END API DIAGNOSTIC")
    print("========================================")
    print("API base URL:", API_BASE_URL)
    print("Using synthetic resource ID:", PROBE_ID)
    print("Mutation probes are expected to be denied before resource lookup.")

    cases = [
        # Read permissions expected for hai.dev.
        ("proposal list read", "GET", "/api/v1/capability-proposals", (200,), ()),
        ("lifecycle list read", "GET", "/api/v1/capabilities", (200,), (503,)),
        ("lifecycle inspection read", "GET", "/api/v1/capabilities/inspection", (200,), (503,)),

        # Proposal router authorizes before validation or manager lookup.
        (
            "proposal submission denied",
            "POST",
            "/api/v1/capability-proposals",
            (403,),
            (),
        ),
        (
            "proposal approval denied",
            "POST",
            f"/api/v1/capability-proposals/{PROBE_ID}/approve",
            (403,),
            (),
        ),
        (
            "proposal rejection denied",
            "POST",
            f"/api/v1/capability-proposals/{PROBE_ID}/reject",
            (403,),
            (),
        ),
        (
            "proposal registration denied",
            "POST",
            f"/api/v1/capability-proposals/{PROBE_ID}/register",
            (403,),
            (),
        ),

        # Lifecycle routes resolve the lifecycle manager before PBAC.
        # A 503 means the environment is not ready to prove the PBAC result.
        (
            "lifecycle activation denied",
            "POST",
            f"/api/v1/capabilities/{PROBE_ID}/activate",
            (403,),
            (503,),
        ),
        (
            "lifecycle disable denied",
            "POST",
            f"/api/v1/capabilities/{PROBE_ID}/disable",
            (403,),
            (503,),
        ),
        (
            "lifecycle revoke denied",
            "POST",
            f"/api/v1/capabilities/{PROBE_ID}/revoke",
            (403,),
            (503,),
        ),
    ]

    pass_count = 0
    fail_count = 0
    blocked_count = 0

    proposal_payload = {
        "capability_id": "pbac-diagnostic-probe",
        "action_name": "pbac_diagnostic_probe",
        "description": "Synthetic proposal used only to verify PBAC denial.",
        "category": "read_only",
        "risk_level": "low",
        "permissions": [],
        "preconditions": {},
        "effects": {},
        "parameters": {},
    }
    rejection_payload = {"reason": "PBAC diagnostic only; must be denied."}

    for name, method, path, expected, blocked in cases:
        payload = None
        if name == "proposal submission denied":
            payload = proposal_payload
        elif name == "proposal rejection denied":
            payload = rejection_payload

        result = run_api_case(
            name,
            method,
            path,
            access_token,
            expected,
            payload=payload,
            blocked_statuses=blocked,
        )
        if result == "PASS":
            pass_count += 1
        elif result == "BLOCKED":
            blocked_count += 1
        else:
            fail_count += 1

    # Confirm that a protected read route rejects a request without a token.
    print("\n=== UNAUTHENTICATED REQUEST CHECK ===")
    unauth_result = run_api_case(
        "missing bearer token rejected",
        "GET",
        "/api/v1/capability-proposals",
        None,
        (401,),
    )
    if unauth_result == "PASS":
        pass_count += 1
    elif unauth_result == "BLOCKED":
        blocked_count += 1
    else:
        fail_count += 1

    print("\n=== PBAC SUMMARY ===")
    print("PASS:", pass_count)
    print("FAIL:", fail_count)
    print("BLOCKED:", blocked_count)
    return pass_count, fail_count, blocked_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keycloak PKCE, token, API authentication, and PBAC diagnostic."
    )
    parser.add_argument("--keycloak-base-url", default=KEYCLOAK_BASE_URL)
    parser.add_argument("--realm", default=REALM)
    parser.add_argument("--client-id", default=CLIENT_ID)
    parser.add_argument("--api-base-url", default=API_BASE_URL)
    parser.add_argument(
        "--skip-pbac",
        action="store_true",
        help="Only run PKCE, token, and UserInfo diagnostics.",
    )
    args = parser.parse_args()

    keycloak_base = args.keycloak_base_url.rstrip("/")
    issuer = f"{keycloak_base}/realms/{args.realm}"
    authorization_endpoint = f"{issuer}/protocol/openid-connect/auth"
    token_endpoint = f"{issuer}/protocol/openid-connect/token"
    userinfo_endpoint = f"{issuer}/protocol/openid-connect/userinfo"

    verifier, challenge = create_pkce_pair()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    callback_queue: queue.Queue = queue.Queue()

    CallbackHandler.result_queue = callback_queue
    CallbackHandler.expected_state = state

    try:
        server = HTTPServer(("127.0.0.1", 8765), CallbackHandler)
    except OSError as exc:
        print("Could not bind callback server on 127.0.0.1:8765:", exc)
        return 1

    server.timeout = CALLBACK_TIMEOUT
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    params = {
        "client_id": args.client_id,
        "response_type": "code",
        "scope": "openid profile email",
        "redirect_uri": REDIRECT_URI,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    login_url = authorization_endpoint + "?" + urllib.parse.urlencode(params)

    print("\n=== KEYCLOAK PKCE + PBAC DIAGNOSTIC ===")
    print("Keycloak:", keycloak_base)
    print("Realm:", args.realm)
    print("Client:", args.client_id)
    print("Callback:", REDIRECT_URI)
    print("\nOpening browser. Login with the intended test account.")

    try:
        webbrowser.open(login_url)
        callback_result = callback_queue.get(timeout=CALLBACK_TIMEOUT)
    except queue.Empty:
        print("ERROR: Login callback timeout.")
        return 1
    except Exception as exc:
        print("ERROR: Could not complete browser login:", exc)
        return 1
    finally:
        server.server_close()

    if "error" in callback_result:
        print("AUTHENTICATION FAILED:", callback_result["error"])
        return 1

    print("Authorization code received; exchanging it for tokens.")
    token_response: dict[str, Any] | None = None
    access_token: str | None = None
    id_token: str | None = None

    try:
        token_response = exchange_code(
            callback_result["code"],
            verifier,
            token_endpoint,
            args.client_id,
        )
        access_token = token_response.get("access_token")
        id_token = token_response.get("id_token")
    except urllib.error.HTTPError as exc:
        print("TOKEN EXCHANGE FAILED. HTTP Status:", exc.code)
        print("Response body omitted to avoid accidental credential disclosure.")
        return 1
    except (urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        print("TOKEN EXCHANGE FAILED:", str(exc))
        return 1

    if not isinstance(access_token, str) or not access_token:
        print("ERROR: Access token was not returned.")
        return 1

    print("\nTOKEN EXCHANGE SUCCESSFUL")
    print("Token type:", token_response.get("token_type"))
    print("Expires in:", token_response.get("expires_in"))
    print("Access token: [HIDDEN]")
    print("ID token: [HIDDEN]" if id_token else "ID token: NOT RETURNED")

    access_claims = inspect_jwt(access_token, "ACCESS TOKEN")
    inspect_access_token_roles(access_claims)

    token_checks_ok = (
        access_claims is not None
        and isinstance(access_claims.get("sub"), str)
        and bool(access_claims.get("sub", "").strip())
        and access_claims.get("typ") == "Bearer"
        and access_claims.get("iss") == issuer
        and (
            "hai-agent-api" in access_claims.get("aud", [])
            if isinstance(access_claims.get("aud"), list)
            else access_claims.get("aud") == "hai-agent-api"
        )
    )

    id_token_ok = True
    if isinstance(id_token, str) and id_token:
        id_claims = inspect_jwt(id_token, "ID TOKEN")
        id_token_ok = (
            id_claims is not None
            and isinstance(id_claims.get("sub"), str)
            and bool(id_claims.get("sub", "").strip())
            and id_claims.get("nonce") == nonce
        )
        print("ID token nonce matches:", id_token_ok)
    else:
        id_token_ok = False
        print("\nID TOKEN: NOT RETURNED")

    userinfo_ok = inspect_userinfo(access_token, userinfo_endpoint)

    if not token_checks_ok or not id_token_ok or not userinfo_ok:
        print("\nTOKEN / USERINFO DIAGNOSTIC FAILED.")
        access_token = None
        id_token = None
        token_response = None
        return 1

    print("\nTOKEN AND USERINFO DIAGNOSTICS: PASS")

    if args.skip_pbac:
        print("PBAC diagnostic skipped by command-line option.")
        access_token = None
        id_token = None
        token_response = None
        return 0

    pass_count, fail_count, blocked_count = run_pbAC_diagnostics(
        args.api_base_url,
        access_token,
    )

    # Release references; never print or persist token values.
    access_token = None
    id_token = None
    token_response = None

    print("\n=== FINAL RESULT ===")
    if fail_count:
        print("FAIL: One or more PBAC/API expectations were not met.")
        return 1
    if blocked_count:
        print("PARTIAL: No unexpected denial was observed, but some checks were blocked.")
        return 2
    print(f"SUCCESS: All {pass_count} diagnostic checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
