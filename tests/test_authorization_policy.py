from __future__ import annotations

import pytest

from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import (
    CAPABILITY_PERMISSIONS,
    AuthorizationDenied,
    AuthorizationPolicy,
    AuthorizationPolicyConfigurationError,
)


READ = CAPABILITY_PERMISSIONS["proposal_read"]
SUBMIT = CAPABILITY_PERMISSIONS["proposal_submit"]
APPROVE = CAPABILITY_PERMISSIONS["proposal_approve"]
ACTIVATE = CAPABILITY_PERMISSIONS["lifecycle_activate"]


def principal(actor_id="user-1", roles=(), permissions=()):
    return AuthenticatedPrincipal(
        actor_id=actor_id,
        roles=tuple(roles),
        permissions=tuple(permissions),
    )


def test_policy_defaults_to_deny():
    policy = AuthorizationPolicy()

    decision = policy.evaluate(principal(), READ)

    assert decision.allowed is False
    assert decision.reason == "no_matching_permission"


def test_exact_principal_permission_is_allowed():
    policy = AuthorizationPolicy()
    actor = principal(permissions=(READ,))

    decision = policy.evaluate(actor, READ)

    assert decision.allowed is True
    assert decision.reason == "explicit_principal_permission"
    assert decision.actor_id == "user-1"


def test_permission_prefix_does_not_grant_child_permission():
    policy = AuthorizationPolicy()
    actor = principal(permissions=("capability:proposal",))

    assert policy.is_allowed(actor, READ) is False


def test_role_grants_only_explicitly_mapped_permissions():
    policy = AuthorizationPolicy(
        role_permissions={
            "proposal-reviewer": (READ, APPROVE),
        }
    )
    actor = principal(roles=("proposal-reviewer",))

    assert policy.is_allowed(actor, READ) is True
    assert policy.is_allowed(actor, APPROVE) is True
    assert policy.is_allowed(actor, SUBMIT) is False


def test_admin_role_has_no_implicit_privileges():
    policy = AuthorizationPolicy()
    actor = principal(roles=("admin",))

    assert policy.is_allowed(actor, READ) is False
    assert policy.is_allowed(actor, APPROVE) is False
    assert policy.is_allowed(actor, ACTIVATE) is False


def test_explicit_deny_overrides_principal_permission_and_role_grant():
    policy = AuthorizationPolicy(
        role_permissions={"admin": (READ, APPROVE)},
        denied_permissions=frozenset({APPROVE}),
    )
    actor = principal(
        roles=("admin",),
        permissions=(APPROVE,),
    )

    decision = policy.evaluate(actor, APPROVE)

    assert decision.allowed is False
    assert decision.reason == "explicit_policy_deny"


def test_missing_or_untrusted_principal_is_denied():
    policy = AuthorizationPolicy()

    decision = policy.evaluate(None, READ)

    assert decision.allowed is False
    assert decision.actor_id is None
    assert decision.reason == "missing_or_untrusted_principal"


def test_require_returns_decision_when_allowed():
    policy = AuthorizationPolicy()
    actor = principal(permissions=(READ,))

    decision = policy.require(actor, READ)

    assert decision.allowed is True
    assert decision.permission == READ


def test_require_raises_for_denied_permission():
    policy = AuthorizationPolicy()

    with pytest.raises(AuthorizationDenied) as exc:
        policy.require(principal(), READ)

    assert exc.value.permission == READ
    assert exc.value.actor_id == "user-1"


@pytest.mark.parametrize(
    "permission",
    [
        "",
        "READ",
        "capability",
        "capability:*",
        "*",
        "capability:proposal:read:*",
        " capability:proposal:read ",
    ],
)
def test_permission_input_must_be_canonical(permission):
    policy = AuthorizationPolicy()

    if permission == " capability:proposal:read ":
        # Surrounding whitespace is normalized intentionally.
        assert policy.is_allowed(
            principal(permissions=(READ,)),
            permission,
        )
    else:
        with pytest.raises(AuthorizationPolicyConfigurationError):
            policy.evaluate(principal(), permission)


@pytest.mark.parametrize(
    "role_permissions",
    [
        {"": (READ,)},
        {"Proposal-Reviewer": (READ,)},
        {"admin": ("*",)},
        {"admin": ("capability:proposal",)},
        {"admin": "capability:proposal:read"},
    ],
)
def test_malformed_role_policy_is_rejected(role_permissions):
    with pytest.raises(AuthorizationPolicyConfigurationError):
        AuthorizationPolicy(role_permissions=role_permissions)


def test_policy_copies_role_map_and_permission_collections():
    original = {"reviewer": [READ]}
    policy = AuthorizationPolicy(role_permissions=original)
    original["reviewer"].append(APPROVE)

    actor = principal(roles=("reviewer",))
    assert policy.is_allowed(actor, READ)
    assert policy.is_allowed(actor, APPROVE) is False


def test_capability_permission_catalog_contains_all_expected_operations():
    assert dict(CAPABILITY_PERMISSIONS) == {
        "proposal_read": "capability:proposal:read",
        "proposal_submit": "capability:proposal:submit",
        "proposal_approve": "capability:proposal:approve",
        "proposal_reject": "capability:proposal:reject",
        "proposal_register": "capability:proposal:register",
        "lifecycle_read": "capability:lifecycle:read",
        "lifecycle_activate": "capability:lifecycle:activate",
        "lifecycle_disable": "capability:lifecycle:disable",
        "lifecycle_revoke": "capability:lifecycle:revoke",
        "environment_workspace_read": "environment:workspace:read",
    }
