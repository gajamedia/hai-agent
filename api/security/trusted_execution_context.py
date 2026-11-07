from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .authentication import AuthenticatedPrincipal


@dataclass(frozen=True, slots=True, init=False)
class TrustedExecutionContext:
    """Immutable, request-scoped identity context for trusted execution.

    Instances can only be created from the application's verified
    ``AuthenticatedPrincipal``. The execution ID is minted here; callers
    cannot provide or override it. The object contains no credentials.

    Keep this object out of LLM prompts, plan steps, request metadata, and
    world state. Pass it only as an explicit runtime keyword argument.
    """

    principal: AuthenticatedPrincipal
    execution_id: str = field(init=False)

    def __init__(self, principal: AuthenticatedPrincipal) -> None:
        if type(principal) is not AuthenticatedPrincipal:
            raise TypeError(
                "principal must be the exact AuthenticatedPrincipal type."
            )
        if principal.is_authenticated is not True:
            raise ValueError("principal must represent an authenticated identity.")

        object.__setattr__(self, "principal", principal)
        object.__setattr__(self, "execution_id", uuid4().hex)


def create_trusted_execution_context(
    principal: AuthenticatedPrincipal,
) -> TrustedExecutionContext:
    """Create a trusted context from the verified request principal."""
    return TrustedExecutionContext(principal)
