from typing import Any, Dict, Optional

from agent.core.action import Action


class FailureAnalysis:

    def __init__(
        self,
        action: str,
        problem: str,
        cause: str,
        lesson: str,
        condition: Any = None,
    ):
        self.action = action
        self.problem = problem
        self.cause = cause
        self.lesson = lesson
        self.condition = condition


class FailureAnalyzer:

    def analyze(
        self,
        action: Action,
        world_state: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Optional[FailureAnalysis]:

        if result.get("success", False):
            return None

        failed_conditions = []

        for key, expected in action.preconditions.items():

            actual = world_state.get(key)

            if actual != expected:

                failed_conditions.append(
                    {
                        "key": key,
                        "expected": expected,
                        "actual": actual,
                    }
                )

        if failed_conditions:

            first = failed_conditions[0]

            problem = (
                f"Action '{action.name}' failed "
                f"because precondition "
                f"'{first['key']}' was not satisfied."
            )

            cause = (
                f"{first['key']} was "
                f"{first['actual']!r}, "
                f"expected {first['expected']!r}."
            )

            lesson = (
                f"Before executing '{action.name}', "
                f"verify that "
                f"{first['key']} == "
                f"{first['expected']!r}."
            )

            return FailureAnalysis(
                action=action.name,
                problem=problem,
                cause=cause,
                lesson=lesson,
                condition=first,
            )

        message = result.get(
            "message",
            "Unknown failure.",
        )

        return FailureAnalysis(
            action=action.name,
            problem=message,
            cause="The action reported a failure.",
            lesson=(
                f"Investigate the environment "
                f"before retrying '{action.name}'."
            ),
        )
