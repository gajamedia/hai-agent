from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .context import AgentContext
from .contracts import LLMResponse
from .reasoning import LLMReasoning
from .service import LLMService


_DEFAULT_SYSTEM_PROMPT = """You are the planning layer of an autonomous agent.
Create a structured, minimal, goal-directed plan from the supplied goal, agent context, and reasoning.
Do not execute actions, call tools, modify the world model, modify memory, or claim that any action has already happened.
Every step must describe an intended action, not an execution result.
Keep the plan finite and ordered.
Separate rationale, assumptions, constraints, and stop conditions from the action steps.
Return ONLY one JSON object with this schema:
{
  "goal": "the goal being planned for",
  "rationale": "why this plan is appropriate",
  "steps": [
    {
      "index": 1,
      "action": "intended action",
      "purpose": "why this step exists",
      "preconditions": ["conditions that should hold before the step"],
      "expected_observation": "what observation would indicate what happened",
      "success_criteria": "how to judge the step as successful",
      "risk": "main risk of this step or null"
    }
  ],
  "assumptions": ["assumptions made by the planner"],
  "constraints": ["constraints that the plan must obey"],
  "stop_conditions": ["conditions under which planning/execution should stop"],
  "confidence": 0.0
}
The confidence must be a number from 0.0 to 1.0.
The plan must contain at least one step.
Do not wrap the JSON in markdown fences.
"""


class LLMPlanningError(RuntimeError):
    """Base error for the LLM planning layer."""


class LLMPlanningParseError(LLMPlanningError):
    """Raised when the LLM response cannot be parsed as planning JSON."""


class LLMPlanningSchemaError(LLMPlanningError):
    """Raised when parsed JSON does not satisfy the planning schema."""


@dataclass(frozen=True)
class LLMPlanStep:
    """One intended, not-yet-executed plan step."""

    index: int
    action: str
    purpose: str
    preconditions: tuple[str, ...] = ()
    expected_observation: str | None = None
    success_criteria: str | None = None
    risk: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "purpose": self.purpose,
            "preconditions": list(self.preconditions),
            "expected_observation": self.expected_observation,
            "success_criteria": self.success_criteria,
            "risk": self.risk,
        }


@dataclass(frozen=True)
class LLMPlan:
    """Structured plan produced by an LLM; side-effect free."""

    goal: str
    rationale: str
    steps: tuple[LLMPlanStep, ...]
    assumptions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    confidence: float = 0.0
    provider: str | None = None
    model: str | None = None
    raw_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "rationale": self.rationale,
            "steps": [step.to_dict() for step in self.steps],
            "assumptions": list(self.assumptions),
            "constraints": list(self.constraints),
            "stop_conditions": list(self.stop_conditions),
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
        }

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self.steps)


class LLMPlanParser:
    """Parse and validate the canonical planning JSON contract."""

    REQUIRED_KEYS = {
        "goal",
        "rationale",
        "steps",
        "assumptions",
        "constraints",
        "stop_conditions",
        "confidence",
    }
    STEP_REQUIRED_KEYS = {
        "index",
        "action",
        "purpose",
        "preconditions",
        "expected_observation",
        "success_criteria",
        "risk",
    }

    @classmethod
    def parse(
        cls,
        content: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_steps: int = 32,
    ) -> LLMPlan:
        if not isinstance(content, str) or not content.strip():
            raise LLMPlanningParseError("LLM planning content is empty.")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        payload = cls._decode_json(content)
        if not isinstance(payload, Mapping):
            raise LLMPlanningSchemaError("Plan must be a JSON object.")

        missing = cls.REQUIRED_KEYS.difference(payload.keys())
        if missing:
            names = ", ".join(sorted(missing))
            raise LLMPlanningSchemaError(
                f"Plan is missing required fields: {names}"
            )

        goal = cls._required_string(payload["goal"], "goal")
        rationale = cls._required_string(payload["rationale"], "rationale")

        raw_steps = payload["steps"]
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes, bytearray)):
            raise LLMPlanningSchemaError("steps must be an array of objects.")
        if not raw_steps:
            raise LLMPlanningSchemaError("Plan must contain at least one step.")
        if len(raw_steps) > max_steps:
            raise LLMPlanningSchemaError(
                f"Plan cannot contain more than {max_steps} steps."
            )

        steps: list[LLMPlanStep] = []
        seen_indices: set[int] = set()
        for position, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, Mapping):
                raise LLMPlanningSchemaError(
                    f"steps[{position - 1}] must be an object."
                )
            missing_step = cls.STEP_REQUIRED_KEYS.difference(raw_step.keys())
            if missing_step:
                names = ", ".join(sorted(missing_step))
                raise LLMPlanningSchemaError(
                    f"steps[{position - 1}] is missing required fields: {names}"
                )

            index = raw_step["index"]
            if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
                raise LLMPlanningSchemaError(
                    f"steps[{position - 1}].index must be a positive integer."
                )
            if index in seen_indices:
                raise LLMPlanningSchemaError("Plan step indices must be unique.")
            seen_indices.add(index)

            action = cls._required_string(raw_step["action"], "action", position - 1)
            purpose = cls._required_string(raw_step["purpose"], "purpose", position - 1)
            preconditions = cls._string_sequence(
                raw_step["preconditions"],
                f"steps[{position - 1}].preconditions",
            )
            expected_observation = cls._optional_string(
                raw_step["expected_observation"],
                f"steps[{position - 1}].expected_observation",
            )
            success_criteria = cls._optional_string(
                raw_step["success_criteria"],
                f"steps[{position - 1}].success_criteria",
            )
            risk = cls._optional_string(
                raw_step["risk"],
                f"steps[{position - 1}].risk",
            )

            steps.append(
                LLMPlanStep(
                    index=index,
                    action=action,
                    purpose=purpose,
                    preconditions=preconditions,
                    expected_observation=expected_observation,
                    success_criteria=success_criteria,
                    risk=risk,
                )
            )

        indices = [step.index for step in steps]
        if indices != list(range(1, len(steps) + 1)):
            raise LLMPlanningSchemaError("Plan step indices must start at 1 and be sequential.")

        assumptions = cls._string_sequence(payload["assumptions"], "assumptions")
        constraints = cls._string_sequence(payload["constraints"], "constraints")
        stop_conditions = cls._string_sequence(payload["stop_conditions"], "stop_conditions")

        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise LLMPlanningSchemaError("confidence must be a number from 0.0 to 1.0.")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise LLMPlanningSchemaError("confidence must be a number from 0.0 to 1.0.")

        return LLMPlan(
            goal=goal,
            rationale=rationale,
            steps=tuple(steps),
            assumptions=assumptions,
            constraints=constraints,
            stop_conditions=stop_conditions,
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

        raise LLMPlanningParseError(
            "LLM response does not contain valid planning JSON."
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
    def _required_string(value: Any, field_name: str, step_index: int | None = None) -> str:
        label = field_name if step_index is None else f"steps[{step_index}].{field_name}"
        if not isinstance(value, str) or not value.strip():
            raise LLMPlanningSchemaError(f"{label} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _optional_string(value: Any, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise LLMPlanningSchemaError(f"{field_name} must be a string or null.")
        return value.strip() or None

    @staticmethod
    def _string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise LLMPlanningSchemaError(f"{field_name} must be an array of strings.")
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMPlanningSchemaError(
                    f"{field_name} must contain only strings."
                )
            if item.strip():
                result.append(item.strip())
        return tuple(result)


class LLMPlanner:
    """Ask an injected LLM service to create a plan from reasoning + context."""

    def __init__(
        self,
        llm_service: LLMService,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        temperature: float | None = 0.0,
        max_tokens: int | None = 1800,
        max_steps: int = 12,
    ) -> None:
        if llm_service is None:
            raise ValueError("llm_service is required")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        self.llm_service = llm_service
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.last_response: LLMResponse | None = None
        self.last_plan: LLMPlan | None = None

    def plan(
        self,
        reasoning: LLMReasoning,
        context: AgentContext,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlan:
        if not isinstance(reasoning, LLMReasoning):
            raise TypeError("reasoning must be an LLMReasoning")
        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        reasoning_text = json.dumps(
            reasoning.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        prompt = (
            "Create a finite plan for the supplied goal using the agent context and reasoning.\n\n"
            "### REASONING\n"
            f"{reasoning_text}\n\n"
            "### AGENT CONTEXT\n"
            f"{context.to_prompt()}\n\n"
            "Plan only; do not execute or report execution results. "
            f"Use at most {self.max_steps} ordered steps.\n\n"
            "Return the required JSON object only."
        )

        response = self.llm_service.complete_text(
            prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata={
                "operation": "planning",
                "max_steps": self.max_steps,
                **dict(metadata or {}),
            },
        )

        plan = LLMPlanParser.parse(
            response.content,
            provider=response.provider,
            model=response.model,
            max_steps=self.max_steps,
        )

        self.last_response = response
        self.last_plan = plan
        return plan

    def clear_trace(self) -> None:
        self.last_response = None
        self.last_plan = None


__all__ = [
    "LLMPlan",
    "LLMPlanParser",
    "LLMPlanStep",
    "LLMPlanner",
    "LLMPlanningError",
    "LLMPlanningParseError",
    "LLMPlanningSchemaError",
]
