from typing import Any, Dict, List, Optional

from agent.core.action import Action


class Planner:

    def __init__(self, actions: List[Action]):
        self.actions = actions

    def choose_action(
        self,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
    ) -> Optional[Action]:

        plan = self.plan(world_state, goal_state)

        if not plan:
            return None

        return plan[0]

    def plan(
        self,
        world_state: Dict[str, Any],
        goal_state: Dict[str, Any],
    ) -> List[Action]:

        state = world_state.copy()
        plan: List[Action] = []

        success = self._achieve_goals(
            state=state,
            goals=goal_state,
            plan=plan,
            visited=set(),
        )

        if not success:
            return []

        return plan

    def _achieve_goals(
        self,
        state: Dict[str, Any],
        goals: Dict[str, Any],
        plan: List[Action],
        visited: set,
    ) -> bool:

        for key, expected_value in goals.items():

            if state.get(key) == expected_value:
                continue

            if not self._achieve_goal(
                state=state,
                key=key,
                expected_value=expected_value,
                plan=plan,
                visited=visited,
            ):
                return False

        return True

    def _achieve_goal(
        self,
        state: Dict[str, Any],
        key: str,
        expected_value: Any,
        plan: List[Action],
        visited: set,
    ) -> bool:

        # Already achieved
        if state.get(key) == expected_value:
            return True

        goal_signature = (key, expected_value)

        if goal_signature in visited:
            return False

        visited.add(goal_signature)

        # Find actions capable of producing the desired state
        candidates = [
            action
            for action in self.actions
            if action.effects.get(key) == expected_value
        ]

        for action in candidates:

            simulated_state = state.copy()
            simulated_plan = list(plan)

            # Solve action preconditions first
            if not self._achieve_goals(
                state=simulated_state,
                goals=action.preconditions,
                plan=simulated_plan,
                visited=set(visited),
            ):
                continue

            # Preconditions are satisfied.
            # Apply the action effects to the simulated world.
            for effect_key, effect_value in action.effects.items():
                simulated_state[effect_key] = effect_value

            simulated_plan.append(action)

            # Verify that our original goal is now satisfied
            if simulated_state.get(key) == expected_value:

                state.clear()
                state.update(simulated_state)

                plan.clear()
                plan.extend(simulated_plan)

                return True

        return False
