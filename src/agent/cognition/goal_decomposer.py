from typing import Callable, Dict, List, Optional

from agent.core.goal import Goal


class GoalDecomposer:

    def __init__(
        self,
        rules: Optional[
            Dict[str, Callable[[Goal], List[Goal]]]
        ] = None,
    ):
        self.rules = rules or {}

    def register_rule(
        self,
        goal_description: str,
        decomposer: Callable[[Goal], List[Goal]],
    ) -> None:

        self.rules[
            goal_description
        ] = decomposer

    def decompose(
        self,
        goal: Goal,
    ) -> Goal:

        if goal.description not in self.rules:
            return goal

        sub_goals = self.rules[
            goal.description
        ](goal)

        for sub_goal in sub_goals:
            goal.add_sub_goal(
                sub_goal
            )

        return goal

    def decompose_recursive(
        self,
        goal: Goal,
    ) -> Goal:

        self.decompose(goal)

        for sub_goal in goal.sub_goals:
            self.decompose_recursive(
                sub_goal
            )

        return goal
