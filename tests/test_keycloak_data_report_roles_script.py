from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "keycloak_data_report_roles.py"
spec = importlib.util.spec_from_file_location("keycloak_data_report_roles", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class FakeKeycloak:
    """Just enough of the admin REST API for the script."""

    def __init__(self, roles=("api-reader",), user_roles=("api-reader",), admin_ok=True):
        self.roles = {n: {"id": f"id-{n}", "name": n} for n in roles}
        self.user_roles = set(user_roles)
        self.admin_ok, self.calls, self.leaked = admin_ok, [], []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        self.leaked.append(json.dumps(body, default=str) if method != "POST" or "token" not in url else "")
        if url.endswith("/protocol/openid-connect/token"):
            ok = self.admin_ok and body["password"] == "right"
            return (200, {"access_token": "T"}) if ok else (401, None)
        assert headers and headers["Authorization"] == "Bearer T"
        if re.search(r"/clients\?clientId=", url):
            return 200, [{"id": "client-uuid"}]
        if re.search(r"/users\?username=hai.dev", url):
            return 200, [{"id": "user-uuid"}]
        if re.search(r"/users\?username=", url):
            return 200, []
        role = re.search(r"/clients/client-uuid/roles/([^/]+)$", url)
        if role and method == "GET":
            return (200, self.roles[role.group(1)]) if role.group(1) in self.roles else (404, None)
        if url.endswith("/clients/client-uuid/roles") and method == "POST":
            self.roles[body["name"]] = {"id": f"id-{body['name']}", "name": body["name"]}
            return 201, None
        if url.endswith("/users/user-uuid/role-mappings/clients/client-uuid"):
            if method == "GET":
                return 200, [self.roles[n] for n in sorted(self.user_roles)]
            names = {r["name"] for r in body}
            self.user_roles = (self.user_roles | names) if method == "POST" else (self.user_roles - names)
            return 204, None
        raise AssertionError(f"unexpected call {method} {url}")


def _run(fake, **kw):
    args = dict(base_url="http://localhost:8080", realm="hai-agent", client_id="hai-agent-api",
                username="hai.dev", roles=list(tool.DEFAULT_ROLES), admin_user="admin",
                admin_password="right", http=fake)
    args.update(kw)
    return tool.run(**args)


def test_missing_roles_are_created_and_granted():
    fake = FakeKeycloak()
    result = _run(fake)
    assert set(result["created_roles"]) == set(tool.DEFAULT_ROLES)
    assert set(result["roles_now"]) == {"api-reader", *tool.DEFAULT_ROLES}


def test_running_twice_changes_nothing_the_second_time():
    fake = FakeKeycloak()
    _run(fake)
    again = _run(fake)
    assert again["created_roles"] == [] and set(again["roles_now"]) == {"api-reader", *tool.DEFAULT_ROLES}


def test_existing_roles_are_not_recreated():
    fake = FakeKeycloak(roles=("api-reader", *tool.DEFAULT_ROLES))
    assert _run(fake)["created_roles"] == []
    assert not any(m == "POST" and u.endswith("/roles") for m, u in fake.calls)


def test_revoke_removes_only_the_named_roles():
    fake = FakeKeycloak(roles=("api-reader", *tool.DEFAULT_ROLES), user_roles=("api-reader", *tool.DEFAULT_ROLES))
    result = _run(fake, revoke=True, roles=["data-reports-user"])
    assert set(result["roles_now"]) == {"api-reader", "database-connection-manager"}


def test_revoking_never_creates_roles():
    fake = FakeKeycloak()
    _run(fake, revoke=True)
    assert not any(m == "POST" and u.endswith("/roles") for m, u in fake.calls)


def test_the_reviewer_role_is_only_granted_when_asked_for_explicitly():
    assert "database-engine-reviewer" not in tool.DEFAULT_ROLES
    fake = FakeKeycloak()
    assert "database-engine-reviewer" in _run(fake, roles=["database-engine-reviewer"])["roles_now"]


def test_a_wrong_admin_password_is_reported_without_echoing_it(capsys):
    fake = FakeKeycloak(admin_ok=False)
    with pytest.raises(tool.KeycloakAdminError) as info:
        _run(fake, admin_password="hunter2-secret")
    assert "hunter2-secret" not in str(info.value)


def test_unknown_user_and_unknown_role_are_refused():
    with pytest.raises(tool.KeycloakAdminError, match="not found"):
        _run(FakeKeycloak(), username="nobody")
    with pytest.raises(tool.KeycloakAdminError, match="unknown role"):
        _run(FakeKeycloak(), roles=["realm-admin"])


@pytest.mark.parametrize("url", ["http://keycloak.example.com", "ftp://localhost", "localhost:8080", ""])
def test_plain_http_to_a_remote_host_and_odd_schemes_are_refused(url):
    with pytest.raises(tool.KeycloakAdminError):
        _run(FakeKeycloak(), base_url=url)


def test_https_remote_hosts_and_loopback_http_are_accepted():
    assert tool._check_url("https://keycloak.example.com/") == "https://keycloak.example.com"
    assert tool._check_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_main_prints_a_summary_without_the_password(monkeypatch, capsys):
    monkeypatch.setenv("KC_ADMIN_PASSWORD", "right")
    assert tool.main(["--user", "hai.dev"], http=FakeKeycloak()) == 0
    out = capsys.readouterr().out
    assert "granted to hai.dev" in out and "right" not in out and "sign in again" in out.lower()
    monkeypatch.setenv("KC_ADMIN_PASSWORD", "wrong")
    assert tool.main(["--user", "hai.dev"], http=FakeKeycloak()) == 1
    assert "wrong" not in capsys.readouterr().err
