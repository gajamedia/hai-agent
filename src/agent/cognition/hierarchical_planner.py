from typing import Dict, List, Optional, Set

from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.goal_dependency_resolver import GoalDependencyResolver


class HierarchicalPlanner:
    def __init__(
        self,
        planner: Planner,
        decomposer: GoalDecomposer,
        procedural_store=None,
        causal_store=None,
    ):
        self.planner = planner
        self.decomposer = decomposer
        self.dependency_resolver = GoalDependencyResolver()

        # Optional long-term learning sources.
        #
        # Existing behavior remains unchanged when these are None.
        self.procedural_store = procedural_store
        self.causal_store = causal_store

    def plan(
        self,
        goal: Goal,
        world_state: Dict,
    ) -> List[Action]:

        root = self.decomposer.decompose_recursive(goal)

        # Validate dependency cycles across the whole hierarchy.
        for current_goal in self._all_goals(root):
            current_goal.validate_dependencies()

        leaf_goals = self._ordered_leaf_goals(root)

        state = dict(world_state)
        result: List[Action] = []

        for leaf_goal in leaf_goals:
            goal_state = self._goal_to_state(leaf_goal)

            # Abstract leaf without a target cannot be planned directly.
            if not goal_state:
                continue

            # ---------------------------------------------------------
            # 1. Try learned procedural experience first.
            # ---------------------------------------------------------
            learned_actions = self._try_procedural_plan(
                goal=leaf_goal,
                world_state=state,
            )

            if learned_actions is not None:
                actions = learned_actions
            else:
                # -----------------------------------------------------
                # 2. Fallback to classical symbolic planner.
                # -----------------------------------------------------
                actions = self.planner.plan(
                    world_state=state,
                    goal_state=goal_state,
                )

            if not actions:
                return []

            for action in actions:
                result.append(action)
                state.update(action.effects)

        return result

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def _all_goals(self, goal: Goal) -> List[Goal]:
        """
        Return the root and every descendant goal.
        """
        return [goal] + goal.flatten()

    def _leaf_goals(self, goal: Goal) -> List[Goal]:
        """
        Return executable leaf goals while preserving hierarchy order.
        """
        if not goal.has_sub_goals:
            return [goal]

        result: List[Goal] = []

        for sub_goal in goal.sub_goals:
            result.extend(
                self._leaf_goals(sub_goal)
            )

        return result

    def _ordered_leaf_goals(
        self,
        root: Goal,
    ) -> List[Goal]:
        """
        Convert goal-level dependencies into executable leaf dependencies.

        Example:

            unlock_door
                depends_on obtain_key
                    contains take_key

        becomes:

            take_key -> unlock_door
        """

        leaf_goals = self._leaf_goals(root)

        if len(leaf_goals) <= 1:
            return leaf_goals

        leaf_ids = {
            id(leaf)
            for leaf in leaf_goals
        }

        dependency_edges: Dict[int, Set[int]] = {
            id(leaf): set()
            for leaf in leaf_goals
        }

        for current_goal in self._all_goals(root):

            current_leaves = [
                leaf
                for leaf in self._leaf_goals(current_goal)
                if id(leaf) in leaf_ids
            ]

            if not current_leaves:
                continue

            for dependency in current_goal.dependencies:

                dependency_leaves = [
                    leaf
                    for leaf in self._leaf_goals(dependency)
                    if id(leaf) in leaf_ids
                ]

                for current_leaf in current_leaves:
                    current_leaf_id = id(current_leaf)

                    for dependency_leaf in dependency_leaves:
                        dependency_leaf_id = id(dependency_leaf)

                        if dependency_leaf_id != current_leaf_id:
                            dependency_edges[
                                current_leaf_id
                            ].add(
                                dependency_leaf_id
                            )

        return self._stable_topological_sort(
            leaf_goals,
            dependency_edges,
        )

    def _stable_topological_sort(
        self,
        leaf_goals: List[Goal],
        dependency_edges: Dict[int, Set[int]],
    ) -> List[Goal]:

        result: List[Goal] = []
        completed: Set[int] = set()
        remaining = list(leaf_goals)

        while remaining:

            candidates = [
                leaf
                for leaf in remaining
                if dependency_edges[
                    id(leaf)
                ].issubset(completed)
            ]

            if not candidates:
                raise ValueError(
                    "Leaf goal dependency cycle detected."
                )

            # Preserve hierarchy order for independent goals.
            selected = candidates[0]

            result.append(selected)
            completed.add(id(selected))
            remaining.remove(selected)

        return result

    # ------------------------------------------------------------------
    # Procedural Memory integration
    # ------------------------------------------------------------------

    def _try_procedural_plan(
        self,
        goal: Goal,
        world_state: Dict,
    ) -> Optional[List[Action]]:
        """
        Try to reuse a previously learned successful procedure.

        Returns:
            List[Action] when a valid learned procedure exists.
            None when procedural memory is unavailable or unsuitable.
        """

        if self.procedural_store is None:
            return None

        skill = self._goal_to_skill(goal)

        memory = self._find_procedural_memory(skill)

        if memory is None:
            return None

        success_count = getattr(
            memory,
            "success_count",
            0,
        )

        failure_count = getattr(
            memory,
            "failure_count",
            0,
        )

        # Do not trust a procedure that has no successful evidence.
        if success_count <= 0:
            return None

        # A procedure with more failures than successes is not reliable
        # enough to override symbolic planning.
        if failure_count > success_count:
            return None

        steps = list(
            getattr(memory, "steps", [])
        )

        if not steps:
            return None

        actions = self._resolve_action_steps(steps)

        if actions is None:
            return None

        # Validate the learned sequence against current state.
        if not self._validate_action_sequence(
            actions,
            world_state,
            goal.target_state,
        ):
            return None

        # Use causal memory as an additional consistency check.
        if not self._validate_causal_sequence(
            actions,
            world_state,
        ):
            return None

        return actions

    def _goal_to_skill(
        self,
        goal: Goal,
    ) -> str:
        """
        Use the same normalization convention used by ProceduralLearning.
        """
        return (
            goal.description
            .strip()
            .lower()
            .replace(" ", "_")
        )

    def _find_procedural_memory(
        self,
        skill: str,
    ):
        """
        Retrieve a procedural memory without requiring a specific
        store implementation.

        Supports stores exposing either:
            get(skill)
        or:
            all()
        """

        getter = getattr(
            self.procedural_store,
            "get",
            None,
        )

        if callable(getter):
            memory = getter(skill)

            if memory is not None:
                return memory

        all_method = getattr(
            self.procedural_store,
            "all",
            None,
        )

        if not callable(all_method):
            return None

        for memory in all_method():
            memory_skill = getattr(
                memory,
                "skill",
                None,
            )

            if memory_skill == skill:
                return memory

        return None

    def _resolve_action_steps(
        self,
        steps: List[str],
    ) -> Optional[List[Action]]:
        """
        Convert learned action names into registered Action objects.
        """

        registered_actions = {
            action.name: action
            for action in getattr(
                self.planner,
                "actions",
                [],
            )
        }

        result: List[Action] = []

        for step in steps:
            action = registered_actions.get(step)

            if action is None:
                return None

            result.append(action)

        return result

    def _validate_action_sequence(
        self,
        actions: List[Action],
        world_state: Dict,
        target_state: Dict,
    ) -> bool:
        """
        Simulate a learned procedure without actually executing it.
        """

        state = dict(world_state)

        for action in actions:

            if not action.can_execute(state):
                return False

            state.update(action.effects)

        if not target_state:
            return True

        return all(
            state.get(key) == expected
            for key, expected in target_state.items()
        )

    # ------------------------------------------------------------------
    # Causal Memory integration
    # ------------------------------------------------------------------

    def _validate_causal_sequence(
        self,
        actions: List[Action],
        world_state: Dict,
    ) -> bool:
        """
        Use known causal relationships as a safety/consistency check.

        Important:
        Absence of causal memory does NOT invalidate a procedure.

        Only known contradictory causal evidence can reject it.
        """

        if self.causal_store is None:
            return True

        if len(actions) <= 1:
            return True

        causal_memories = self._all_causal_memories()

        if not causal_memories:
            return True

        state = dict(world_state)

        for previous, current in zip(
            actions,
            actions[1:],
        ):

            relevant = [
                memory
                for memory in causal_memories
                if self._causal_matches(
                    memory,
                    previous.name,
                    current.name,
                )
            ]

            if not relevant:
                # No learned causal evidence.
                # Do not block the procedure.
                state.update(previous.effects)
                continue

            # If there is explicit causal evidence, ensure at least
            # one non-contradicted relationship is applicable.
            applicable = False

            for memory in relevant:

                contradiction_count = getattr(
                    memory,
                    "contradiction_count",
                    0,
                )

                evidence_count = getattr(
                    memory,
                    "evidence_count",
                    0,
                )

                if contradiction_count > evidence_count:
                    continue

                condition = getattr(
                    memory,
                    "condition",
                    None,
                )

                if self._condition_matches(
                    condition,
                    state,
                ):
                    applicable = True
                    break

                if condition in (
                    None,
                    {},
                ):
                    applicable = True
                    break

            if not applicable:
                return False

            state.update(previous.effects)

        return True

    def _all_causal_memories(self) -> List:
        all_method = getattr(
            self.causal_store,
            "all",
            None,
        )

        if not callable(all_method):
            return []

        return list(
            all_method()
        )

    def _causal_matches(
        self,
        memory,
        cause: str,
        effect: str,
    ) -> bool:
        memory_cause = getattr(
            memory,
            "cause",
            None,
        )

        memory_effect = getattr(
            memory,
            "effect",
            None,
        )

        relation = getattr(
            memory,
            "relation",
            None,
        )

        return (
            memory_cause == cause
            and memory_effect == effect
            and relation == "enables"
        )

    def _condition_matches(
        self,
        condition,
        state: Dict,
    ) -> bool:

        if not isinstance(
            condition,
            dict,
        ):
            return True

        return all(
            state.get(key) == expected
            for key, expected in condition.items()
        )

    # ------------------------------------------------------------------
    # Goal -> planner target
    # ------------------------------------------------------------------

    def _goal_to_state(
        self,
        goal: Goal,
    ) -> Dict:
        """
        Convert a leaf Goal into the planner target state.
        """
        return dict(
            goal.target_state
        )
