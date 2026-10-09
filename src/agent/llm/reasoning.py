from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .context import AgentContext
from .interpretation import LLMInterpretation
from .contracts import LLMResponse
from .service import LLMService


_DEFAULT_SYSTEM_PROMPT = """You are the reasoning layer of an autonomous agent.
Your job is to reason from the supplied observation interpretation and agent context.
Do not execute actions, modify the world model, modify memory, or create a step-by-step action plan.
Separate what is concluded from alternatives, risks, and information gaps.
Return ONLY a single JSON object with this schema:
{
  "summary": "brief reasoning summary",
  "conclusions": ["supported conclusions"],
  "alternatives": ["plausible alternative explanations or interpretations"],
  "risks": ["relevant risks or failure modes"],
  "information_gaps": ["important missing information"],
  "recommended_direction": "high-level direction for later planning, or null",
  "confidence": 0.0
}
The confidence must be a number from 0.0 to 1.0.
"recommended_direction" must be high-level and must NOT be a step-by-step action plan.
Do not wrap the JSON in markdown fences.
"""


class LLMReasoningError(RuntimeError):
    """Base error for the LLM reasoning layer."""


class LLMReasoningParseError(LLMReasoningError):
    """Raised when the LLM response cannot be parsed as reasoning JSON."""


class LLMReasoningSchemaError(LLMReasoningError):
    """Raised when parsed JSON does not satisfy the reasoning schema."""


@dataclass(frozen=True)
class LLMReasoning:
    """Structured, side-effect-free reasoning produced by an LLM."""

    summary: str
    conclusions: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    information_gaps: tuple[str, ...] = ()
    recommended_direction: str | None = None
    confidence: float = 0.0
    provider: str | None = None
    model: str | None = None
    raw_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "conclusions": list(self.conclusions),
            "alternatives": list(self.alternatives),
            "risks": list(self.risks),
            "information_gaps": list(self.information_gaps),
            "recommended_direction": self.recommended_direction,
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
        }


class LLMReasoningParser:
    """Parse and validate the canonical reasoning JSON contract."""

    REQUIRED_KEYS = {
        "summary",
        "conclusions",
        "alternatives",
        "risks",
        "information_gaps",
        "recommended_direction",
        "confidence",
    }

    @classmethod
    def parse(
        cls,
        content: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> LLMReasoning:
        if not isinstance(content, str) or not content.strip():
            raise LLMReasoningParseError("LLM reasoning content is empty.")

        payload = cls._decode_json(content)
        if not isinstance(payload, Mapping):
            raise LLMReasoningSchemaError("Reasoning must be a JSON object.")

        missing = cls.REQUIRED_KEYS.difference(payload.keys())
        if missing:
            names = ", ".join(sorted(missing))
            raise LLMReasoningSchemaError(
                f"Reasoning is missing required fields: {names}"
            )

        summary = payload["summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise LLMReasoningSchemaError("summary must be a non-empty string.")

        conclusions = cls._string_sequence(payload["conclusions"], "conclusions")
        alternatives = cls._string_sequence(payload["alternatives"], "alternatives")
        risks = cls._string_sequence(payload["risks"], "risks")
        information_gaps = cls._string_sequence(
            payload["information_gaps"],
            "information_gaps",
        )

        recommended_direction = payload["recommended_direction"]
        if recommended_direction is not None:
            if not isinstance(recommended_direction, str):
                raise LLMReasoningSchemaError(
                    "recommended_direction must be a string or null."
                )
            recommended_direction = recommended_direction.strip() or None

        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise LLMReasoningSchemaError("confidence must be a number from 0.0 to 1.0.")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise LLMReasoningSchemaError("confidence must be a number from 0.0 to 1.0.")

        return LLMReasoning(
            summary=summary.strip(),
            conclusions=conclusions,
            alternatives=alternatives,
            risks=risks,
            information_gaps=information_gaps,
            recommended_direction=recommended_direction,
            confidence=confidence,
            provider=provider,
            model=model,
            raw_content=content,
        )

    @classmethod
    def _decode_json(cls, content: str) -> Any:
        text = content.strip()
        candidates = [text]

        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            candidates.append(fenced.group(1).strip())

        object_match = cls._extract_json_object(text)
        if object_match is not None:
            candidates.append(object_match)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise LLMReasoningParseError(
            "LLM response does not contain valid reasoning JSON."
        )

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        return None

    @staticmethod
    def _string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise LLMReasoningSchemaError(
                f"{field_name} must be an array of strings."
            )

        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMReasoningSchemaError(
                    f"{field_name} must contain only strings."
                )
            if item.strip():
                result.append(item.strip())
        return tuple(result)


class LLMReasoner:
    """Ask an injected LLM service to reason from interpretation + context."""

    def __init__(
        self,
        llm_service: LLMService,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        temperature: float | None = 0.0,
        max_tokens: int | None = 1400,
    ) -> None:
        if llm_service is None:
            raise ValueError("llm_service is required")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")

        self.llm_service = llm_service
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_response: LLMResponse | None = None
        self.last_reasoning: LLMReasoning | None = None

    def reason(
        self,
        interpretation: LLMInterpretation,
        context: AgentContext,
        *,
        observation: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMReasoning:
        if not isinstance(interpretation, LLMInterpretation):
            raise TypeError("interpretation must be an LLMInterpretation")
        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        interpretation_text = json.dumps(
            interpretation.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        observation_text = self._render_observation(observation)

        prompt = (
            "Reason about the supplied interpretation in the context of the agent.\n\n"
            "### INTERPRETATION\n"
            f"{interpretation_text}\n\n"
            "### OBSERVATION\n"
            f"{observation_text}\n\n"
            "### AGENT CONTEXT\n"
            f"{context.to_prompt()}\n\n"
            "Focus on consequences, competing explanations, risks, and missing information. "
            "Do not produce a step-by-step action plan.\n\n"
            "Return the required JSON object only."
        )

        response = self.llm_service.complete_text(
            prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata={
                "operation": "reasoning",
                **dict(metadata or {}),
            },
        )

        reasoning = LLMReasoningParser.parse(
            response.content,
            provider=response.provider,
            model=response.model,
        )

        self.last_response = response
        self.last_reasoning = reasoning
        return reasoning

    @staticmethod
    def _render_observation(observation: Any) -> str:
        if observation is None:
            return "<not provided>"
        if isinstance(observation, str):
            return observation
        try:
            return json.dumps(
                observation,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            return str(observation)

    def clear_trace(self) -> None:
        self.last_response = None
        self.last_reasoning = None


__all__ = [
    "LLMReasoning",
    "LLMReasoningError",
    "LLMReasoningParseError",
    "LLMReasoningParser",
    "LLMReasoningSchemaError",
    "LLMReasoner",
]
