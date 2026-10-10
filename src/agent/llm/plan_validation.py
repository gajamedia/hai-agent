# from __future__ import annotations

# from dataclasses import dataclass
# from math import isfinite
# from typing import Any, Iterable, Mapping


# class LLMPlanValidationError(ValueError):
#     """Base error for malformed LLM plans."""


# @dataclass(frozen=True)
# class PlanValidationIssue:
#     code: str
#     message: str
#     step_index: int | None = None


# @dataclass(frozen=True)
# class LLMPlanValidationResult:
#     valid: bool
#     issues: tuple[PlanValidationIssue, ...]
#     checked_step_count: int
#     executable_step_indices: tuple[int, ...]

#     @property
#     def errors(self) -> tuple[PlanValidationIssue, ...]:
#         return self.issues



# def _read(value: Any, *names: str, default: Any = None) -> Any:
#     if value is None:
#         return default
#     if isinstance(value, Mapping):
#         for name in names:
#             if name in value:
#                 return value[name]
#         return default
#     for name in names:
#         if hasattr(value, name):
#             return getattr(value, name)
#     return default


# def _text(value: Any) -> str:
#     if value is None:
#         return ""
#     if isinstance(value, str):
#         return value.strip()
#     nested = _read(value, "description", "name", "goal", default=None)
#     if nested is not None and nested is not value:
#         return str(nested).strip()
#     return str(value).strip()


# def _bounded_number(value: Any, field_name: str) -> float:
#     try:
#         number = float(value)
#     except (TypeError, ValueError) as exc:
#         raise LLMPlanValidationError(
#             f"{field_name} must be numeric."
#         ) from exc
#     if not isfinite(number) or number < 0.0 or number > 1.0:
#         raise LLMPlanValidationError(
#             f"{field_name} must be between 0 and 1."
#         )
#     return number


# def _extract_steps(plan: Any) -> list[Any]:
#     steps = _read(plan, "steps", default=None)
#     if steps is None:
#         raise LLMPlanValidationError("LLM plan must contain steps.")
#     if isinstance(steps, (str, bytes)):
#         raise LLMPlanValidationError("LLM plan steps must be iterable.")
#     try:
#         return list(steps)
#     except TypeError as exc:
#         raise LLMPlanValidationError(
#             "LLM plan steps must be iterable."
#         ) from exc


# def _extract_action_names(available_actions: Any) -> set[str]:
#     if available_actions is None:
#         return set()
#     if isinstance(available_actions, Mapping):
#         return {str(key).strip() for key in available_actions.keys()}

#     names: set[str] = set()
#     for item in available_actions:
#         if isinstance(item, str):
#             names.add(item.strip())
#             continue
#         name = _read(item, "name", "action_name", default=None)
#         if name is not None:
#             names.add(str(name).strip())
#     return names


# class LLMPlanValidator:
#     """
#     Validate an LLM-generated plan before any execution boundary.

#     This validator is deliberately provider-neutral and Agent-neutral.
#     It never executes actions and never mutates Agent state.
#     """

#     def __init__(
#         self,
#         *,
#         min_confidence: float = 0.0,
#         max_step_risk: float = 1.0,
#         max_plan_steps: int | None = None,
#         require_goal_match: bool = True,
#         require_success_criteria: bool = False,
#     ) -> None:
#         self.min_confidence = _bounded_number(min_confidence, "min_confidence")
#         self.max_step_risk = _bounded_number(max_step_risk, "max_step_risk")
#         if max_plan_steps is not None and max_plan_steps < 1:
#             raise ValueError("max_plan_steps must be >= 1 when provided.")
#         self.max_plan_steps = max_plan_steps
#         self.require_goal_match = require_goal_match
#         self.require_success_criteria = require_success_criteria

#     def validate(
#         self,
#         plan: Any,
#         *,
#         goal: Any = None,
#         available_actions: Any = None,
#     ) -> LLMPlanValidationResult:
#         issues: list[PlanValidationIssue] = []

#         steps = _extract_steps(plan)
#         if not steps:
#             issues.append(
#                 PlanValidationIssue(
#                     code="empty_plan",
#                     message="LLM plan must contain at least one step.",
#                 )
#             )

#         if self.max_plan_steps is not None and len(steps) > self.max_plan_steps:
#             issues.append(
#                 PlanValidationIssue(
#                     code="too_many_steps",
#                     message=(
#                         f"LLM plan contains {len(steps)} steps; "
#                         f"maximum is {self.max_plan_steps}."
#                     ),
#                 )
#             )

#         confidence_value = _read(plan, "confidence", default=1.0)
#         try:
#             confidence = _bounded_number(confidence_value, "plan confidence")
#         except LLMPlanValidationError as exc:
#             issues.append(
#                 PlanValidationIssue(
#                     code="invalid_confidence",
#                     message=str(exc),
#                 )
#             )
#             confidence = 0.0

#         if confidence < self.min_confidence:
#             issues.append(
#                 PlanValidationIssue(
#                     code="low_confidence",
#                     message=(
#                         f"Plan confidence {confidence:.3f} is below "
#                         f"minimum {self.min_confidence:.3f}."
#                     ),
#                 )
#             )

#         if self.require_goal_match and goal is not None:
#             plan_goal = _text(_read(plan, "goal", default=None))
#             expected_goal = _text(goal)
#             if not plan_goal:
#                 issues.append(
#                     PlanValidationIssue(
#                         code="missing_goal",
#                         message="LLM plan does not declare a goal.",
#                     )
#                 )
#             elif plan_goal.casefold() != expected_goal.casefold():
#                 issues.append(
#                     PlanValidationIssue(
#                         code="goal_mismatch",
#                         message="LLM plan goal does not match the active goal.",
#                     )
#                 )

#         action_names = _extract_action_names(available_actions)
#         if available_actions is not None and not action_names:
#             issues.append(
#                 PlanValidationIssue(
#                     code="no_available_actions",
#                     message="No executable actions are available for validation.",
#                 )
#             )

#         seen_indices: set[int] = set()
#         executable: list[int] = []

#         for position, step in enumerate(steps, start=1):
#             declared_index = _read(step, "index", "step_index", default=position)
#             try:
#                 numeric_index = int(declared_index)
#             except (TypeError, ValueError):
#                 numeric_index = position
#                 issues.append(
#                     PlanValidationIssue(
#                         code="invalid_step_index",
#                         message="Step index must be an integer.",
#                         step_index=position,
#                     )
#                 )

#             if numeric_index in seen_indices:
#                 issues.append(
#                     PlanValidationIssue(
#                         code="duplicate_step_index",
#                         message=f"Duplicate step index {numeric_index}.",
#                         step_index=position,
#                     )
#                 )
#             seen_indices.add(numeric_index)

#             action_name = _text(_read(step, "action", "action_name", default=None))
#             if not action_name:
#                 issues.append(
#                     PlanValidationIssue(
#                         code="missing_action",
#                         message="Each plan step must declare an action.",
#                         step_index=position,
#                     )
#                 )
#                 continue

#             if action_names and action_name not in action_names:
#                 issues.append(
#                     PlanValidationIssue(
#                         code="unknown_action",
#                         message=f"Action '{action_name}' is not available to the Agent.",
#                         step_index=position,
#                     )
#                 )
#             else:
#                 executable.append(position)

#             risk_value = _read(step, "risk", default=0.0)
#             try:
#                 risk = _bounded_number(risk_value, f"step {position} risk")
#             except LLMPlanValidationError as exc:
#                 issues.append(
#                     PlanValidationIssue(
#                         code="invalid_step_risk",
#                         message=str(exc),
#                         step_index=position,
#                     )
#                 )
#                 risk = 1.0

#             if risk > self.max_step_risk:
#                 issues.append(
#                     PlanValidationIssue(
#                         code="risk_above_limit",
#                         message=(
#                             f"Step risk {risk:.3f} exceeds "
#                             f"limit {self.max_step_risk:.3f}."
#                         ),
#                         step_index=position,
#                     )
#                 )

#             if self.require_success_criteria:
#                 criteria = _read(step, "success_criteria", default=None)
#                 if criteria is None or _text(criteria) == "":
#                     issues.append(
#                         PlanValidationIssue(
#                             code="missing_success_criteria",
#                             message="Step requires success criteria.",
#                             step_index=position,
#                         )
#                     )

#         valid = not issues
#         return LLMPlanValidationResult(
#             valid=valid,
#             issues=tuple(issues),
#             checked_step_count=len(steps),
#             executable_step_indices=tuple(executable if valid else ()),
#         )

#==============================================
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


class LLMPlanValidationError(ValueError):
    """Base error for malformed LLM plans."""


@dataclass(frozen=True)
class PlanValidationIssue:
    code: str
    message: str
    step_index: int | None = None


@dataclass(frozen=True)
class LLMPlanValidationResult:
    valid: bool
    issues: tuple[PlanValidationIssue, ...]
    checked_step_count: int
    executable_step_indices: tuple[int, ...]

    @property
    def errors(self) -> tuple[PlanValidationIssue, ...]:
        return self.issues


def _read(value: Any, *names: str, default: Any = None) -> Any:
    if value is None:
        return default

    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default

    for name in names:
        if hasattr(value, name):
            return getattr(value, name)

    return default


def _text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    nested = _read(value, "description", "name", "goal", default=None)

    if nested is not None and nested is not value:
        return str(nested).strip()

    return str(value).strip()


def _bounded_number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMPlanValidationError(
            f"{field_name} must be numeric."
        ) from exc

    if not isfinite(number) or number < 0.0 or number > 1.0:
        raise LLMPlanValidationError(
            f"{field_name} must be between 0 and 1."
        )

    return number


def _extract_steps(plan: Any) -> list[Any]:
    steps = _read(plan, "steps", default=None)

    if steps is None:
        raise LLMPlanValidationError("LLM plan must contain steps.")

    if isinstance(steps, (str, bytes)):
        raise LLMPlanValidationError("LLM plan steps must be iterable.")

    try:
        return list(steps)
    except TypeError as exc:
        raise LLMPlanValidationError(
            "LLM plan steps must be iterable."
        ) from exc


def _extract_action_names(available_actions: Any) -> set[str]:
    if available_actions is None:
        return set()

    if isinstance(available_actions, Mapping):
        return {str(key).strip() for key in available_actions.keys()}

    names: set[str] = set()

    for item in available_actions:
        if isinstance(item, str):
            names.add(item.strip())
            continue

        name = _read(item, "name", "action_name", default=None)

        if name is not None:
            names.add(str(name).strip())

    return names


class LLMPlanValidator:
    """
    Validate an LLM-generated plan before any execution boundary.

    This validator is provider-neutral and Agent-neutral.
    It never executes actions and never mutates Agent state.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.0,
        max_step_risk: float = 1.0,
        max_plan_steps: int | None = None,
        require_goal_match: bool = True,
        require_success_criteria: bool = False,
    ) -> None:
        self.min_confidence = _bounded_number(
            min_confidence,
            "min_confidence",
        )
        self.max_step_risk = _bounded_number(
            max_step_risk,
            "max_step_risk",
        )

        if max_plan_steps is not None and max_plan_steps < 1:
            raise ValueError("max_plan_steps must be >= 1 when provided.")

        self.max_plan_steps = max_plan_steps
        self.require_goal_match = require_goal_match
        self.require_success_criteria = require_success_criteria

    def validate(
        self,
        plan: Any,
        *,
        goal: Any = None,
        available_actions: Any = None,
    ) -> LLMPlanValidationResult:
        issues: list[PlanValidationIssue] = []
        steps = _extract_steps(plan)

        if not steps:
            issues.append(
                PlanValidationIssue(
                    code="empty_plan",
                    message="LLM plan must contain at least one step.",
                )
            )

        if (
            self.max_plan_steps is not None
            and len(steps) > self.max_plan_steps
        ):
            issues.append(
                PlanValidationIssue(
                    code="too_many_steps",
                    message=(
                        f"LLM plan contains {len(steps)} steps; "
                        f"maximum is {self.max_plan_steps}."
                    ),
                )
            )

        confidence_value = _read(plan, "confidence", default=1.0)

        try:
            confidence = _bounded_number(
                confidence_value,
                "plan confidence",
            )
        except LLMPlanValidationError as exc:
            issues.append(
                PlanValidationIssue(
                    code="invalid_confidence",
                    message=str(exc),
                )
            )
            confidence = 0.0

        if confidence < self.min_confidence:
            issues.append(
                PlanValidationIssue(
                    code="low_confidence",
                    message=(
                        f"Plan confidence {confidence:.3f} is below "
                        f"minimum {self.min_confidence:.3f}."
                    ),
                )
            )

        if self.require_goal_match and goal is not None:
            plan_goal = _text(_read(plan, "goal", default=None))
            expected_goal = _text(goal)

            if not plan_goal:
                issues.append(
                    PlanValidationIssue(
                        code="missing_goal",
                        message="LLM plan does not declare a goal.",
                    )
                )
            elif plan_goal.casefold() != expected_goal.casefold():
                issues.append(
                    PlanValidationIssue(
                        code="goal_mismatch",
                        message=(
                            "LLM plan goal does not match the active goal."
                        ),
                    )
                )

        action_names = _extract_action_names(available_actions)

        if available_actions is not None and not action_names:
            issues.append(
                PlanValidationIssue(
                    code="no_available_actions",
                    message=(
                        "No executable actions are available for validation."
                    ),
                )
            )

        seen_indices: set[int] = set()
        executable: list[int] = []

        for position, step in enumerate(steps, start=1):
            declared_index = _read(
                step,
                "index",
                "step_index",
                default=position,
            )

            try:
                numeric_index = int(declared_index)
            except (TypeError, ValueError):
                numeric_index = position
                issues.append(
                    PlanValidationIssue(
                        code="invalid_step_index",
                        message="Step index must be an integer.",
                        step_index=position,
                    )
                )

            if numeric_index in seen_indices:
                issues.append(
                    PlanValidationIssue(
                        code="duplicate_step_index",
                        message=f"Duplicate step index {numeric_index}.",
                        step_index=position,
                    )
                )

            seen_indices.add(numeric_index)

            action_name = _text(
                _read(step, "action", "action_name", default=None)
            )

            if not action_name:
                issues.append(
                    PlanValidationIssue(
                        code="missing_action",
                        message="Each plan step must declare an action.",
                        step_index=position,
                    )
                )
                continue

            if action_names and action_name not in action_names:
                issues.append(
                    PlanValidationIssue(
                        code="unknown_action",
                        message=(
                            f"Action '{action_name}' is not available "
                            "to the Agent."
                        ),
                        step_index=position,
                    )
                )
            else:
                executable.append(position)

            risk_value = _read(step, "risk", default=0.0)

            if risk_value is None:
                # None is permitted by LLMPlanStep.risk.
                risk = None
            elif isinstance(risk_value, str):
                if isinstance(step, Mapping):
                    # Dictionary input uses risk as a numeric score.
                    issues.append(
                        PlanValidationIssue(
                            code="invalid_step_risk",
                            message=(
                                f"step {position} risk must be numeric."
                            ),
                            step_index=position,
                        )
                    )
                # Object input may use a descriptive string, as
                # permitted by LLMPlanStep.risk.
                risk = None
            else:
                try:
                    risk = _bounded_number(
                        risk_value,
                        f"step {position} risk",
                    )
                except LLMPlanValidationError as exc:
                    issues.append(
                        PlanValidationIssue(
                            code="invalid_step_risk",
                            message=str(exc),
                            step_index=position,
                        )
                    )
                    risk = None

            if risk is not None and risk > self.max_step_risk:
                issues.append(
                    PlanValidationIssue(
                        code="risk_above_limit",
                        message=(
                            f"Step risk {risk:.3f} exceeds "
                            f"limit {self.max_step_risk:.3f}."
                        ),
                        step_index=position,
                    )
                )

            if self.require_success_criteria:
                criteria = _read(
                    step,
                    "success_criteria",
                    default=None,
                )

                if criteria is None or _text(criteria) == "":
                    issues.append(
                        PlanValidationIssue(
                            code="missing_success_criteria",
                            message="Step requires success criteria.",
                            step_index=position,
                        )
                    )

        valid = not issues

        return LLMPlanValidationResult(
            valid=valid,
            issues=tuple(issues),
            checked_step_count=len(steps),
            executable_step_indices=(
                tuple(executable) if valid else ()
            ),
        )

