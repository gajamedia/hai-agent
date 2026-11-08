from dataclasses import FrozenInstanceError

import pytest

from api.security.authentication import AuthenticatedPrincipal
from api.security.trusted_execution_context import (
    TrustedExecutionContext,
    create_trusted_execution_context,
)


def test_context_requires_exact_verified_principal_type():
    class PrincipalSubclass(AuthenticatedPrincipal):
        pass

    with pytest.raises(TypeError, match="exact AuthenticatedPrincipal"):
        TrustedExecutionContext(PrincipalSubclass(actor_id="user-1"))


def test_context_generates_opaque_execution_id():
    principal = AuthenticatedPrincipal(actor_id="user-1", permissions=("capability:read:execute",))
    first = create_trusted_execution_context(principal)
    second = create_trusted_execution_context(principal)

    assert first.principal is principal
    assert first.execution_id
    assert len(first.execution_id) == 32
    assert first.execution_id != second.execution_id


def test_context_is_immutable():
    context = TrustedExecutionContext(AuthenticatedPrincipal(actor_id="user-1"))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        context.execution_id = "caller-controlled"


def test_context_does_not_accept_caller_supplied_execution_id():
    principal = AuthenticatedPrincipal(actor_id="user-1")
    with pytest.raises(TypeError):
        TrustedExecutionContext(principal, "caller-controlled")
