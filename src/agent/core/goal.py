from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Goal:

    description: str

    priority: int = 1

    completed: bool = False

    failed: bool = False

    target_state: Dict[str, Any] = field(
        default_factory=dict
    )

    sub_goals: List["Goal"] = field(
        default_factory=list
    )

    dependencies: List["Goal"] = field(
        default_factory=list
    )

    def complete(self) -> None:
        self.completed = True
        self.failed = False

    def fail(self) -> None:
        self.failed = True
        self.completed = False

    @property
    def is_active(self) -> bool:
        return (
            not self.completed
            and not self.failed
        )

    # =================================================
    # Target State
    # =================================================

    def is_satisfied(
        self,
        world_state: Dict[str, Any],
    ) -> bool:

        if not self.target_state:
            return False

        return all(
            world_state.get(key) == value
            for key, value
            in self.target_state.items()
        )

    # =================================================
    # Sub Goals
    # =================================================

    def add_sub_goal(
        self,
        goal: "Goal",
    ) -> None:
        self.sub_goals.append(goal)

    def remove_sub_goal(
        self,
        description: str,
    ) -> bool:

        for index, goal in enumerate(
            self.sub_goals
        ):
            if goal.description == description:
                del self.sub_goals[index]
                return True

        return False

    def get_sub_goal(
        self,
        description: str,
    ) -> "Goal | None":

        for goal in self.sub_goals:
            if goal.description == description:
                return goal

        return None

    @property
    def has_sub_goals(self) -> bool:
        return len(self.sub_goals) > 0

    @property
    def all_sub_goals_completed(
        self,
    ) -> bool:

        if not self.sub_goals:
            return False

        return all(
            goal.completed
            for goal in self.sub_goals
        )

    @property
    def all_sub_goals_failed(
        self,
    ) -> bool:

        if not self.sub_goals:
            return False

        return all(
            goal.failed
            for goal in self.sub_goals
        )

    def update_status_from_sub_goals(
        self,
    ) -> None:

        if not self.sub_goals:
            return

        if self.all_sub_goals_completed:
            self.complete()
            return

        if self.all_sub_goals_failed:
            self.fail()

    def flatten(
        self,
    ) -> List["Goal"]:

        result: List["Goal"] = []

        for sub_goal in self.sub_goals:
            result.append(sub_goal)

            result.extend(
                sub_goal.flatten()
            )

        return result

    # =================================================
    # Dependencies
    # =================================================

    def add_dependency(
        self,
        goal: "Goal",
    ) -> None:

        if goal is self:
            raise ValueError(
                "A goal cannot depend on itself."
            )

        if goal not in self.dependencies:
            self.dependencies.append(
                goal
            )

    def remove_dependency(
        self,
        goal: "Goal",
    ) -> bool:

        if goal not in self.dependencies:
            return False

        self.dependencies.remove(
            goal
        )

        return True

    @property
    def has_dependencies(self) -> bool:
        return len(
            self.dependencies
        ) > 0

    def dependencies_satisfied(
        self,
    ) -> bool:

        return all(
            dependency.completed
            for dependency
            in self.dependencies
        )

    def can_execute(
        self,
    ) -> bool:

        if self.completed or self.failed:
            return False

        return self.dependencies_satisfied()

    def dependency_graph(
        self,
    ) -> Dict[int, List["Goal"]]:

        graph: Dict[
            int,
            List["Goal"],
        ] = {}

        visited = set()

        def visit(current: "Goal"):

            identity = id(current)

            if identity in visited:
                return

            visited.add(identity)

            graph[identity] = list(
                current.dependencies
            )

            for dependency in (
                current.dependencies
            ):
                visit(dependency)

        visit(self)

        return graph


    def has_dependency_cycle(
        self,
    ) -> bool:

        visiting = set()
        visited = set()

        def visit(current: "Goal") -> bool:

            identity = id(current)

            if identity in visiting:
                return True

            if identity in visited:
                return False

            visiting.add(identity)

            for dependency in (
                current.dependencies
            ):
                if visit(dependency):
                    return True

            visiting.remove(identity)
            visited.add(identity)

            return False

        return visit(self)

    def validate_dependencies(
        self,
    ) -> None:

        if self.has_dependency_cycle():
            raise ValueError(
                "Goal dependency cycle detected."
            )

    def topological_order(
        self,
    ) -> List["Goal"]:

        goals = self.flatten()

        if not goals:
            return []

        visited = set()
        visiting = set()
        result: List["Goal"] = []

        def visit(goal: "Goal") -> None:

            identity = id(goal)

            if identity in visiting:
                raise ValueError(
                    "Goal dependency cycle detected."
                )

            if identity in visited:
                return

            visiting.add(identity)

            for dependency in goal.dependencies:
                visit(dependency)

            visiting.remove(identity)
            visited.add(identity)

            result.append(goal)

        for goal in goals:
            visit(goal)

        return result
