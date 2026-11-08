from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from agent.llm.capability_proposal import (
    CapabilityProposal,
    CapabilityProposalValidationError,
    ProposalCategory,
    ProposalStatus,
)


class CapabilityProposalValidator:
    """Validate declarative proposal metadata without accepting executable code."""

    ACTION_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
    CAPABILITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
    ALLOWED_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
    MAX_JSON_BYTES = 16_384
    MAX_DEPTH = 8

    @classmethod
    def _validate_json_value(cls, value: Any, *, depth: int = 0) -> None:
        if depth > cls.MAX_DEPTH:
            raise CapabilityProposalValidationError("Proposal metadata is too deeply nested.")
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                raise CapabilityProposalValidationError("Non-finite numbers are not allowed.")
            return
        if isinstance(value, list):
            for item in value:
                cls._validate_json_value(item, depth=depth + 1)
            return
        if isinstance(value, tuple):
            for item in value:
                cls._validate_json_value(item, depth=depth + 1)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or not key.strip():
                    raise CapabilityProposalValidationError("Metadata keys must be non-empty strings.")
                cls._validate_json_value(item, depth=depth + 1)
            return
        raise CapabilityProposalValidationError(
            f"Unsupported metadata value type: {type(value).__name__}."
        )

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        return value

    @classmethod
    def _validate_mapping(cls, value: Any, field_name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise CapabilityProposalValidationError(f"{field_name} must be a mapping.")
        normalized = cls._to_jsonable(value)
        cls._validate_json_value(normalized)
        try:
            encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CapabilityProposalValidationError(
                f"{field_name} must contain JSON-compatible data only."
            ) from exc
        if len(encoded) > cls.MAX_JSON_BYTES:
            raise CapabilityProposalValidationError(f"{field_name} exceeds the metadata size limit.")
        return normalized

    def validate(self, proposal: CapabilityProposal) -> CapabilityProposal:
        if not isinstance(proposal, CapabilityProposal):
            raise CapabilityProposalValidationError("proposal must be a CapabilityProposal.")
        if proposal.status != ProposalStatus.PENDING:
            raise CapabilityProposalValidationError("New proposals must have PENDING status.")
        if (
            not isinstance(proposal.proposal_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", proposal.proposal_id)
        ):
            raise CapabilityProposalValidationError("Invalid proposal_id.")
        if proposal.version != 1:
            raise CapabilityProposalValidationError("New proposals must start at version 1.")
        if any(
            value is not None
            for value in (
                proposal.approved_by,
                proposal.rejected_by,
                proposal.rejection_reason,
                proposal.registered_by,
            )
        ):
            raise CapabilityProposalValidationError(
                "Approval and registration metadata are application-managed."
            )
        if not isinstance(proposal.capability_id, str) or not self.CAPABILITY_ID_PATTERN.fullmatch(proposal.capability_id):
            raise CapabilityProposalValidationError("Invalid capability_id.")
        if not isinstance(proposal.action_name, str) or not self.ACTION_NAME_PATTERN.fullmatch(proposal.action_name):
            raise CapabilityProposalValidationError("Invalid action_name.")
        if not isinstance(proposal.description, str) or not proposal.description.strip():
            raise CapabilityProposalValidationError("description cannot be empty.")
        if len(proposal.description) > 500:
            raise CapabilityProposalValidationError("description exceeds 500 characters.")
        if not isinstance(proposal.category, ProposalCategory):
            raise CapabilityProposalValidationError("category must be a ProposalCategory.")
        if (
            not isinstance(proposal.risk_level, str)
            or proposal.risk_level not in self.ALLOWED_RISK_LEVELS
        ):
            raise CapabilityProposalValidationError("Invalid risk_level.")
        if isinstance(proposal.permissions, (str, bytes)):
            raise CapabilityProposalValidationError("permissions must be a sequence of strings.")
        seen: set[str] = set()
        for permission in proposal.permissions:
            if not isinstance(permission, str) or not permission or permission != permission.strip():
                raise CapabilityProposalValidationError("Invalid permission.")
            if permission in seen:
                raise CapabilityProposalValidationError(f"Duplicate permission: {permission}")
            seen.add(permission)

        preconditions = self._validate_mapping(proposal.preconditions, "preconditions")
        effects = self._validate_mapping(proposal.effects, "effects")
        self._validate_mapping(proposal.parameters, "parameters")

        if proposal.category == ProposalCategory.READ_ONLY:
            if effects:
                raise CapabilityProposalValidationError(
                    "Read-only proposals cannot declare effects."
                )
            invalid_permissions = [
                item for item in proposal.permissions
                if not item.startswith(("read:", "observe:"))
            ]
            if invalid_permissions:
                raise CapabilityProposalValidationError(
                    "Read-only proposals may only request read:* or observe:* permissions."
                )
        elif proposal.category == ProposalCategory.STATE_CHANGING:
            # State-changing proposals may describe effects, but they are never
            # eligible for registration in this stage.
            pass

        return proposal


__all__ = ["CapabilityProposalValidator"]
