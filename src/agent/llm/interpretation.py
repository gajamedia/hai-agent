from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .context import AgentContext
from .contracts import LLMResponse
from .service import LLMService


logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = """You are the interpretation layer of an autonomous agent.
Your job is to interpret the supplied observation using the supplied agent context.
Do not execute actions and do not modify the world model.
Return ONLY a single JSON object with this schema:
{
  "summary": "brief interpretation",
  "facts": ["observed or strongly supported facts"],
  "entities": {"name": "relevant attributes"},
  "uncertainties": ["things that remain uncertain"],
  "implications": ["likely implications of the observation"],
  "confidence": 0.0
}
The confidence must be a number from 0.0 to 1.0.
Do not wrap the JSON in markdown fences.
"""


class LLMInterpretationError(RuntimeError):
    """Base error for the LLM interpretation layer."""


class LLMInterpretationParseError(LLMInterpretationError):
    """Raised when the LLM response cannot be parsed as interpretation JSON."""


class LLMInterpretationSchemaError(LLMInterpretationError):
    """Raised when parsed JSON does not satisfy the interpretation schema."""


@dataclass(frozen=True)
class LLMInterpretation:
    """Structured, side-effect-free interpretation produced by an LLM."""

    summary: str
    facts: tuple[str, ...] = ()
    entities: Mapping[str, Any] = field(default_factory=dict)
    uncertainties: tuple[str, ...] = ()
    implications: tuple[str, ...] = ()
    confidence: float = 0.0
    provider: str | None = None
    model: str | None = None
    raw_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "facts": list(self.facts),
            "entities": dict(self.entities),
            "uncertainties": list(self.uncertainties),
            "implications": list(self.implications),
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
        }


class LLMInterpretationParser:
    """Parse and validate the canonical interpretation JSON contract."""

    REQUIRED_KEYS = {"summary", "facts", "entities", "uncertainties", "implications", "confidence"}

    @classmethod
    def parse(
        cls,
        content: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> LLMInterpretation:
        if not isinstance(content, str) or not content.strip():
            raise LLMInterpretationParseError("LLM interpretation content is empty.")

        payload = cls._decode_json(content)
        if not isinstance(payload, Mapping):
            raise LLMInterpretationSchemaError("Interpretation must be a JSON object.")

        missing = cls.REQUIRED_KEYS.difference(payload.keys())
        if missing:
            names = ", ".join(sorted(missing))
            raise LLMInterpretationSchemaError(
                f"Interpretation is missing required fields: {names}"
            )

        summary = payload["summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise LLMInterpretationSchemaError("summary must be a non-empty string.")

        facts = cls._string_sequence(payload["facts"], "facts")
        uncertainties = cls._string_sequence(payload["uncertainties"], "uncertainties")
        implications = cls._string_sequence(payload["implications"], "implications")

        entities = payload["entities"]
        if not isinstance(entities, Mapping):
            raise LLMInterpretationSchemaError("entities must be a JSON object.")

        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise LLMInterpretationSchemaError("confidence must be a number from 0.0 to 1.0.")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise LLMInterpretationSchemaError("confidence must be a number from 0.0 to 1.0.")

        return LLMInterpretation(
            summary=summary.strip(),
            facts=facts,
            entities=dict(entities),
            uncertainties=uncertainties,
            implications=implications,
            confidence=confidence,
            provider=provider,
            model=model,
            raw_content=content,
        )

    @classmethod
    def _decode_json(cls, content: str) -> Any:
        """Decode JSON robustly without weakening interpretation schema checks."""
        text = content.strip().lstrip("\ufeff")
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

        errors: list[json.JSONDecodeError] = []
        for candidate in dict.fromkeys(candidates):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                errors.append(exc)
            # Some providers emit literal newlines/control characters inside a
            # JSON string. This compatibility mode only affects decoding; the
            # parsed object still goes through the strict schema validation.
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError as exc:
                errors.append(exc)

        if errors:
            error = min(errors, key=lambda exc: (exc.pos, -len(exc.doc)))
            logger.warning(
                "LLM interpretation JSON decode failed: candidates=%d, chars=%d, "
                "line=%d, column=%d, position=%d, reason=%s",
                len(list(dict.fromkeys(candidates))), len(text), error.lineno,
                error.colno, error.pos, error.msg,
            )
        raise LLMInterpretationParseError(
            "LLM response does not contain valid interpretation JSON."
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
            raise LLMInterpretationSchemaError(
                f"{field_name} must be an array of strings."
            )

        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMInterpretationSchemaError(
                    f"{field_name} must contain only strings."
                )
            if item.strip():
                result.append(item.strip())
        return tuple(result)


class LLMInterpreter:
    """Ask an injected LLM service to interpret an observation."""

    def __init__(
        self,
        llm_service: LLMService,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        temperature: float | None = 0.0,
        max_tokens: int | None = 1200,
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
        self.last_interpretation: LLMInterpretation | None = None

    def interpret(
        self,
        observation: Any,
        context: AgentContext,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMInterpretation:
        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        observation_text = self._render_observation(observation)
        prompt = (
            "Interpret the following observation in the context of the agent.\n\n"
            "### OBSERVATION\n"
            f"{observation_text}\n\n"
            "### AGENT CONTEXT\n"
            f"{context.to_prompt()}\n\n"
            "Return the required JSON object only."
        )

        request_metadata = {
            "operation": "interpretation",
            **dict(metadata or {}),
        }
        response = self.llm_service.complete_text(
            prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata=request_metadata,
        )

        try:
            interpretation = LLMInterpretationParser.parse(
                response.content,
                provider=response.provider,
                model=response.model,
            )
        except (LLMInterpretationParseError, LLMInterpretationSchemaError) as parse_error:
            logger.warning(
                "LLM interpretation response failed parsing/schema validation; "
                "requesting one bounded correction (provider=%s, model=%s, error=%s)",
                response.provider,
                response.model,
                str(parse_error),
            )
            correction_prompt = (
                prompt
                + "\n\n### OUTPUT VALIDATION FAILED\n"
                + "Regenerate the entire interpretation as exactly one valid JSON object.\n"
                + "Required fields: summary (non-empty string), facts (array of strings), "
                + "entities (object), uncertainties (array of strings), implications "
                + "(array of strings), confidence (number from 0.0 to 1.0).\n"
                + "Do not include markdown fences or text outside JSON. Do not invent facts.\n"
                + "Validation error: " + str(parse_error)
                + "\nReturn only the corrected JSON object."
            )
            response = self.llm_service.complete_text(
                correction_prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                metadata={
                    **request_metadata,
                    "interpretation_retry": 1,
                    "interpretation_retry_reason": (
                        "json_parse" if isinstance(parse_error, LLMInterpretationParseError)
                        else "schema_validation"
                    ),
                },
            )
            # A second invalid response is intentionally surfaced; never return
            # an unvalidated or fabricated interpretation.
            interpretation = LLMInterpretationParser.parse(
                response.content,
                provider=response.provider,
                model=response.model,
            )

        self.last_response = response
        self.last_interpretation = interpretation
        return interpretation

    @staticmethod
    def _render_observation(observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        try:
            return json.dumps(observation, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(observation)

    def clear_trace(self) -> None:
        self.last_response = None
        self.last_interpretation = None


__all__ = [
    "LLMInterpretation",
    "LLMInterpretationError",
    "LLMInterpretationParseError",
    "LLMInterpretationParser",
    "LLMInterpretationSchemaError",
    "LLMInterpreter",
]
