from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .context import AgentContext
from .contracts import LLMResponse
from .service import LLMService
from ..learning.metacognition import MetacognitiveReflection


_DEFAULT_SYSTEM_PROMPT = """You are the metacognition layer of an autonomous agent.
Analyze one completed or partially observed episode and improve the agent's self-reflection.
Treat the deterministic reflection as evidence, not as unquestionable truth.
Do not invent observations, outcomes, causes, or capabilities that are not supported by the episode, context, or deterministic reflection.
Clearly separate evidence from inference and hypothesis.
Identify uncertainty and alternative explanations when evidence is weak or ambiguous.
Do not execute actions, call tools, modify memory, modify the world model, or claim that an action was performed.
Return ONLY one JSON object with this schema:
{
  "diagnosis": "best-supported diagnosis",
  "lesson": "what the agent should learn",
  "next_step": "what should be checked or considered next",
  "evidence_for": ["evidence supporting the diagnosis"],
  "evidence_against": ["evidence weakening the diagnosis or indicating a problem"],
  "alternative_explanations": ["plausible alternative explanations"],
  "uncertainties": ["important unresolved uncertainties"],
  "change_required": false,
  "confidence": 0.0,
  "alignment": "supports"
}
The confidence must be a number from 0.0 to 1.0.
The alignment must be one of: supports, refines, contradicts, insufficient_evidence.
"""

_ALLOWED_ALIGNMENTS = {
    "supports",
    "refines",
    "contradicts",
    "insufficient_evidence",
}


class LLMMetacognitionError(RuntimeError):
    """Base error for the LLM-assisted metacognition layer."""


class LLMMetacognitionParseError(LLMMetacognitionError):
    """Raised when the LLM response cannot be parsed as JSON."""


class LLMMetacognitionSchemaError(LLMMetacognitionError):
    """Raised when parsed JSON violates the metacognition contract."""


@dataclass(frozen=True)
class LLMMetacognitiveReflection:
    """LLM-generated meta-level reflection; it has no side effects."""

    diagnosis: str
    lesson: str
    next_step: str
    evidence_for: tuple[str, ...] = ()
    evidence_against: tuple[str, ...] = ()
    alternative_explanations: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    change_required: bool = False
    confidence: float = 0.0
    alignment: str = "insufficient_evidence"
    provider: str | None = None
    model: str | None = None
    raw_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "lesson": self.lesson,
            "next_step": self.next_step,
            "evidence_for": list(self.evidence_for),
            "evidence_against": list(self.evidence_against),
            "alternative_explanations": list(self.alternative_explanations),
            "uncertainties": list(self.uncertainties),
            "change_required": self.change_required,
            "confidence": self.confidence,
            "alignment": self.alignment,
            "provider": self.provider,
            "model": self.model,
        }


@dataclass(frozen=True)
class LLMMetacognitionResult:
    """Combined deterministic + LLM reflection for transparent provenance."""

    deterministic_reflection: MetacognitiveReflection
    llm_reflection: LLMMetacognitiveReflection
    provider: str | None = None
    model: str | None = None
    raw_content: str | None = None

    @property
    def confidence(self) -> float:
        return self.llm_reflection.confidence

    @property
    def alignment(self) -> str:
        return self.llm_reflection.alignment

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic_reflection": _dataclass_to_dict(
                self.deterministic_reflection
            ),
            "llm_reflection": self.llm_reflection.to_dict(),
            "provider": self.provider,
            "model": self.model,
        }


class LLMMetacognitionParser:
    """Parse and validate the canonical LLM metacognition JSON contract."""

    REQUIRED_KEYS = {
        "diagnosis",
        "lesson",
        "next_step",
        "evidence_for",
        "evidence_against",
        "alternative_explanations",
        "uncertainties",
        "change_required",
        "confidence",
        "alignment",
    }

    @classmethod
    def parse(
        cls,
        content: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> LLMMetacognitiveReflection:
        if not isinstance(content, str) or not content.strip():
            raise LLMMetacognitionParseError(
                "LLM metacognition content is empty."
            )

        payload = cls._decode_json(content)
        if not isinstance(payload, Mapping):
            raise LLMMetacognitionSchemaError(
                "Metacognition response must be a JSON object."
            )

        missing = cls.REQUIRED_KEYS.difference(payload.keys())
        if missing:
            names = ", ".join(sorted(missing))
            raise LLMMetacognitionSchemaError(
                f"Metacognition response is missing required fields: {names}"
            )

        diagnosis = cls._required_string(payload["diagnosis"], "diagnosis")
        lesson = cls._required_string(payload["lesson"], "lesson")
        next_step = cls._required_string(payload["next_step"], "next_step")

        evidence_for = cls._string_sequence(
            payload["evidence_for"], "evidence_for"
        )
        evidence_against = cls._string_sequence(
            payload["evidence_against"], "evidence_against"
        )
        alternative_explanations = cls._string_sequence(
            payload["alternative_explanations"],
            "alternative_explanations",
        )
        uncertainties = cls._string_sequence(
            payload["uncertainties"], "uncertainties"
        )

        change_required = payload["change_required"]
        if not isinstance(change_required, bool):
            raise LLMMetacognitionSchemaError(
                "change_required must be a boolean."
            )

        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            raise LLMMetacognitionSchemaError(
                "confidence must be a number from 0.0 to 1.0."
            )
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise LLMMetacognitionSchemaError(
                "confidence must be a number from 0.0 to 1.0."
            )

        alignment = payload["alignment"]
        if not isinstance(alignment, str) or alignment not in _ALLOWED_ALIGNMENTS:
            raise LLMMetacognitionSchemaError(
                "alignment must be one of: supports, refines, contradicts, insufficient_evidence."
            )

        return LLMMetacognitiveReflection(
            diagnosis=diagnosis,
            lesson=lesson,
            next_step=next_step,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            alternative_explanations=alternative_explanations,
            uncertainties=uncertainties,
            change_required=change_required,
            confidence=confidence,
            alignment=alignment,
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

        raise LLMMetacognitionParseError(
            "LLM response does not contain valid metacognition JSON."
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
    def _required_string(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise LLMMetacognitionSchemaError(
                f"{field_name} must be a non-empty string."
            )
        return value.strip()

    @staticmethod
    def _string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise LLMMetacognitionSchemaError(
                f"{field_name} must be an array of strings."
            )

        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMMetacognitionSchemaError(
                    f"{field_name} must contain only strings."
                )
            if item.strip():
                result.append(item.strip())
        return tuple(result)


class LLMMetacognitionEngine:
    """Use an LLM to deepen a deterministic metacognitive reflection."""

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
        self.last_result: LLMMetacognitionResult | None = None

    def reflect(
        self,
        episode: Any,
        deterministic_reflection: MetacognitiveReflection,
        context: AgentContext,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMMetacognitionResult:
        if not isinstance(deterministic_reflection, MetacognitiveReflection):
            raise TypeError(
                "deterministic_reflection must be a MetacognitiveReflection"
            )
        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        episode_text = self._render_json(episode)
        baseline_text = self._render_json(
            _dataclass_to_dict(deterministic_reflection)
        )

        prompt = (
            "Deepen the deterministic metacognitive reflection for the supplied episode.\n\n"
            "### EPISODE\n"
            f"{episode_text}\n\n"
            "### DETERMINISTIC REFLECTION\n"
            f"{baseline_text}\n\n"
            "### AGENT CONTEXT\n"
            f"{context.to_prompt()}\n\n"
            "Use the available evidence to refine self-understanding. Do not invent causes. "
            "Explicitly surface supporting evidence, counter-evidence, alternative explanations, "
            "and unresolved uncertainties. Do not produce an action plan.\n\n"
            "Return the required JSON object only."
        )

        response = self.llm_service.complete_text(
            prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata={
                "operation": "metacognition",
                **dict(metadata or {}),
            },
        )

        llm_reflection = LLMMetacognitionParser.parse(
            response.content,
            provider=response.provider,
            model=response.model,
        )

        result = LLMMetacognitionResult(
            deterministic_reflection=deterministic_reflection,
            llm_reflection=llm_reflection,
            provider=response.provider,
            model=response.model,
            raw_content=response.content,
        )

        self.last_response = response
        self.last_result = result
        return result

    @staticmethod
    def _render_json(value: Any) -> str:
        if value is None:
            return "<not provided>"
        if isinstance(value, str):
            return value
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            return str(value)

    def clear_trace(self) -> None:
        self.last_response = None
        self.last_result = None



def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    if not hasattr(value, "__dataclass_fields__"):
        return {"value": value}
    result: dict[str, Any] = {}
    for field_name in value.__dataclass_fields__:
        result[field_name] = getattr(value, field_name)
    return result


__all__ = [
    "LLMMetacognitiveReflection",
    "LLMMetacognitionEngine",
    "LLMMetacognitionError",
    "LLMMetacognitionParseError",
    "LLMMetacognitionParser",
    "LLMMetacognitionResult",
    "LLMMetacognitionSchemaError",
]
