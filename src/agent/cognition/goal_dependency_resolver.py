from typing import List

from agent.core.goal import Goal


class GoalDependencyResolver:

    def resolve(
        self,
        goal: Goal,
    ) -> List[Goal]:

        goal.validate_dependencies()

        return goal.topological_order()
