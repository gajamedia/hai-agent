"""The role that may create databases grants nothing else, and nothing else grants it."""
from __future__ import annotations

import json

from tests.test_data_report_roles import (  # noqa: F401  (policy is a module-scoped fixture there)
    APPROVE_P, APPROVER, CLIENT, EXEC_P, MANAGER, MANAGE, PLAN_P, REALM, REVIEWER, RUN, USER, WRITER,
    _allowed, _principal, policy,
)

CREATOR = "data-schema-creator"
CREATE = "database:create:manage"


def test_creating_databases_is_its_own_role_and_grants_nothing_else(policy):
    assert _allowed(policy, _principal(CREATOR), CREATE)
    assert not any(_allowed(policy, _principal(CREATOR), p) for p in (PLAN_P, EXEC_P, APPROVE_P, RUN, MANAGE))
    for role in (USER, MANAGER, REVIEWER, WRITER, APPROVER):
        assert not _allowed(policy, _principal(role), CREATE)


def test_the_realm_declares_the_role_and_keeps_the_dev_user_without_it():
    realm = json.loads(REALM.read_text(encoding="utf-8"))
    assert CREATOR in {r["name"] for r in realm["roles"]["client"][CLIENT]}
    dev = next(u for u in realm["users"] if u["username"] == "hai.dev")
    assert CREATOR not in dev["clientRoles"][CLIENT]
