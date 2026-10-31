from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

from agent.core.action import Action


class DynamicWorkspaceActionError(RuntimeError):
    """Base error for dynamic workspace action generation."""


class DynamicWorkspaceValidationError(DynamicWorkspaceActionError):
    """Raised when a generated action is invalid or unsupported."""


class DynamicWorkspaceRegistrationError(DynamicWorkspaceActionError):
    """Raised when an action cannot be registered safely."""


CapabilityHandler = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class DynamicActionSpec:
    name: str
    description: str
    capability_id: str
    preconditions: Mapping[str, Any] = field(default_factory=dict)
    effects: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicActionBuildReport:
    plan_actions: tuple[str, ...]
    existing_actions: tuple[str, ...]
    generated_actions: tuple[str, ...]
    registered_actions: tuple[str, ...]


def _observe_capability(
    *,
    environment: Any,
    parameters: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Trusted, read-only capability that observes the current environment."""
    observer = getattr(environment, "observe", None)

    if not callable(observer):
        raise DynamicWorkspaceValidationError(
            "Environment does not expose observe()."
        )

    observation = observer()
    if not isinstance(observation, Mapping):
        raise DynamicWorkspaceValidationError(
            "Environment observation must be a mapping."
        )

    return {
        "success": True,
        "observation": dict(observation),
        "message": "Environment observation completed.",
    }


class DynamicWorkspaceActionBuilder:
    """
    Build declarative Action objects from explicitly trusted capabilities.

    This class never evaluates LLM-provided Python or imports capabilities
    dynamically. The Agent remains the canonical action registry and its
    register_action() method remains the registration entry point.

    The builder lock serializes calls made through this builder instance.
    It cannot coordinate unrelated code that mutates agent.actions directly.
    """

    _NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
    _RESERVED_HANDLER_ARGUMENTS = frozenset({"environment", "parameters"})

    def __init__(
        self,
        agent: Any,
        *,
        capabilities: Mapping[str, CapabilityHandler] | None = None,
        allow_existing: bool = True,
    ) -> None:
        if agent is None:
            raise DynamicWorkspaceValidationError("agent cannot be None.")

        registry = getattr(agent, "actions", None)
        if not isinstance(registry, MutableMapping):
            raise DynamicWorkspaceValidationError(
                "Agent must expose a mutable actions registry."
            )

        if not callable(getattr(agent, "register_action", None)):
            raise DynamicWorkspaceValidationError(
                "Agent must expose register_action()."
            )

        self.agent = agent
        self.allow_existing = bool(allow_existing)
        self._registration_lock = RLock()

        self.capabilities: dict[str, CapabilityHandler] = {
            "environment.observe": _observe_capability,
        }

        if capabilities is not None:
            if not isinstance(capabilities, Mapping):
                raise DynamicWorkspaceValidationError(
                    "capabilities must be a mapping."
                )
            for capability_id, handler in capabilities.items():
                self.register_capability(capability_id, handler)

    def register_capability(
        self,
        capability_id: str,
        handler: CapabilityHandler,
    ) -> None:
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise DynamicWorkspaceValidationError(
                "capability_id must be a non-empty string."
            )
        if capability_id != capability_id.strip():
            raise DynamicWorkspaceValidationError(
                "capability_id cannot contain surrounding whitespace."
            )
        if not callable(handler):
            raise DynamicWorkspaceValidationError(
                "Capability handler must be callable."
            )
        if capability_id in self.capabilities:
            raise DynamicWorkspaceRegistrationError(
                f"Capability already registered: {capability_id}"
            )
        self.capabilities[capability_id] = handler

    @classmethod
    def _validate_name(cls, name: str, field_name: str) -> None:
        if not isinstance(name, str) or not cls._NAME_PATTERN.fullmatch(name):
            raise DynamicWorkspaceValidationError(
                f"Invalid {field_name}: {name!r}"
            )

    @staticmethod
    def _validate_mapping(
        value: Mapping[str, Any],
        field_name: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise DynamicWorkspaceValidationError(
                f"{field_name} must be a mapping."
            )

        result = dict(value)
        for key in result:
            if not isinstance(key, str) or not key.strip():
                raise DynamicWorkspaceValidationError(
                    f"{field_name} contains an invalid key."
                )
        return result

    def validate_spec(self, spec: DynamicActionSpec) -> None:
        if not isinstance(spec, DynamicActionSpec):
            raise DynamicWorkspaceValidationError(
                "spec must be a DynamicActionSpec."
            )

        self._validate_name(spec.name, "action name")

        if not isinstance(spec.description, str) or not spec.description.strip():
            raise DynamicWorkspaceValidationError(
                "Action description cannot be empty."
            )

        if (
            not isinstance(spec.capability_id, str)
            or spec.capability_id not in self.capabilities
        ):
            raise DynamicWorkspaceValidationError(
                f"Unsupported capability: {spec.capability_id!r}"
            )

        self._validate_mapping(spec.preconditions, "preconditions")
        self._validate_mapping(spec.effects, "effects")
        self._validate_mapping(spec.parameters, "parameters")

    def build_action(self, spec: DynamicActionSpec) -> Action:
        self.validate_spec(spec)

        capability = self.capabilities[spec.capability_id]
        # Snapshot configuration so later caller mutation cannot silently
        # alter the behavior of an already-built Action.
        fixed_parameters = self._validate_mapping(
            spec.parameters, "parameters"
        )
        preconditions = self._validate_mapping(
            spec.preconditions, "preconditions"
        )
        effects = self._validate_mapping(spec.effects, "effects")

        def generated_handler(
            *,
            environment: Any = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if environment is None:
                raise DynamicWorkspaceActionError(
                    "Action execution requires an environment."
                )

            reserved = self._RESERVED_HANDLER_ARGUMENTS.intersection(kwargs)
            if reserved:
                names = ", ".join(sorted(reserved))
                raise DynamicWorkspaceValidationError(
                    f"Capability arguments contain reserved name(s): {names}."
                )

            result = capability(
                environment=environment,
                parameters=dict(fixed_parameters),
                **kwargs,
            )

            if not isinstance(result, Mapping):
                raise DynamicWorkspaceActionError(
                    "Capability must return a mapping."
                )
            if "success" not in result:
                raise DynamicWorkspaceActionError(
                    "Capability result must contain 'success'."
                )
            return dict(result)

        return Action(
            name=spec.name,
            description=spec.description,
            handler=generated_handler,
            preconditions=preconditions,
            effects=effects,
        )

    @staticmethod
    def _extract_plan_actions(plan: Any) -> tuple[str, ...]:
        steps = getattr(plan, "steps", None)
        if not steps:
            raise DynamicWorkspaceValidationError(
                "Plan must contain at least one step."
            )

        try:
            iterator = iter(steps)
        except TypeError as exc:
            raise DynamicWorkspaceValidationError(
                "Plan steps must be iterable."
            ) from exc

        actions: list[str] = []
        for step in iterator:
            action_name = getattr(step, "action", None)
            if (
                not isinstance(action_name, str)
                or not action_name.strip()
                or action_name != action_name.strip()
            ):
                raise DynamicWorkspaceValidationError(
                    "Every plan step must have a valid action name."
                )
            actions.append(action_name)

        if not actions:
            raise DynamicWorkspaceValidationError(
                "Plan must contain at least one step."
            )
        return tuple(actions)

    def _restore_registry(
        self,
        registry: MutableMapping[str, Any],
        snapshot: dict[str, Any],
    ) -> None:
        registry.clear()
        registry.update(snapshot)

        # Agent.register_action() reconfigures the planner after mutation.
        # Reconfigure after rollback so the planner reflects the restored
        # canonical registry.
        configure_planner = getattr(self.agent, "configure_planner", None)
        if callable(configure_planner):
            configure_planner()

    def ensure_plan_actions(
        self,
        plan: Any,
        specifications: Mapping[str, DynamicActionSpec],
    ) -> DynamicActionBuildReport:
        """
        Ensure every distinct action referenced by the plan is registered.

        All missing specifications are validated and built before the first
        registry mutation. If a registration fails, the original registry is
        restored and the planner is synchronized again when supported.
        """
        if not isinstance(specifications, Mapping):
            raise DynamicWorkspaceValidationError(
                "specifications must be a mapping."
            )

        plan_actions = self._extract_plan_actions(plan)
        unique_plan_actions = tuple(dict.fromkeys(plan_actions))

        with self._registration_lock:
            registry = self.agent.actions
            existing_actions: list[str] = []
            missing_actions: list[str] = []

            for name in unique_plan_actions:
                if name in registry:
                    if not self.allow_existing:
                        raise DynamicWorkspaceRegistrationError(
                            f"Action already exists: {name}"
                        )
                    existing_actions.append(name)
                else:
                    missing_actions.append(name)

            built_actions: list[Action] = []
            for name in missing_actions:
                spec = specifications.get(name)
                if spec is None:
                    raise DynamicWorkspaceValidationError(
                        f"Missing specification for plan action: {name}"
                    )
                if not isinstance(spec, DynamicActionSpec):
                    raise DynamicWorkspaceValidationError(
                        f"Specification for '{name}' must be a DynamicActionSpec."
                    )
                if spec.name != name:
                    raise DynamicWorkspaceValidationError(
                        f"Specification key '{name}' does not match "
                        f"action name '{spec.name}'."
                    )
                built_actions.append(self.build_action(spec))

            if not built_actions:
                return DynamicActionBuildReport(
                    plan_actions=plan_actions,
                    existing_actions=tuple(existing_actions),
                    generated_actions=(),
                    registered_actions=(),
                )

            # Snapshot only after all validation/build operations succeeded.
            # Keep the same registry object because Agent and other modules
            # may retain references to it.
            snapshot = dict(registry)
            registered_names: list[str] = []

            try:
                for action in built_actions:
                    # Protect against a registry mutation that occurred after
                    # the initial inspection (within the builder's own flow).
                    if action.name in registry:
                        raise DynamicWorkspaceRegistrationError(
                            f"Action appeared during registration: {action.name}"
                        )

                    self.agent.register_action(action)

                    # Agent.register_action currently assigns by name. Verify
                    # the canonical registry contains precisely our object.
                    if registry.get(action.name) is not action:
                        raise DynamicWorkspaceRegistrationError(
                            f"Agent did not retain generated action: {action.name}"
                        )
                    registered_names.append(action.name)

            except Exception as exc:
                try:
                    self._restore_registry(registry, snapshot)
                except Exception as rollback_exc:
                    raise DynamicWorkspaceRegistrationError(
                        "Action registration failed and registry rollback "
                        "could not fully resynchronize the planner."
                    ) from rollback_exc

                if isinstance(exc, DynamicWorkspaceActionError):
                    raise
                raise DynamicWorkspaceRegistrationError(
                    "Action registration failed; registry was restored."
                ) from exc

            generated_names = tuple(action.name for action in built_actions)
            return DynamicActionBuildReport(
                plan_actions=plan_actions,
                existing_actions=tuple(existing_actions),
                generated_actions=generated_names,
                registered_actions=tuple(registered_names),
            )


__all__ = [
    "DynamicActionSpec",
    "DynamicActionBuildReport",
    "DynamicWorkspaceActionBuilder",
    "DynamicWorkspaceActionError",
    "DynamicWorkspaceValidationError",
    "DynamicWorkspaceRegistrationError",
]
