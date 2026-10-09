from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from agent.learning.strategy import Strategy
from agent.self.assessment import SelfAssessment


@dataclass
class SelfModel:

    capabilities: Dict[str, Any] = field(
        default_factory=dict
    )

    limitations: List[str] = field(
        default_factory=list
    )

    known_successes: List[str] = field(
        default_factory=list
    )

    known_failures: List[str] = field(
        default_factory=list
    )

    learned_strategies: Dict[
        str,
        Strategy,
    ] = field(
        default_factory=dict
    )

    confidence: float = 0.5

    current_condition: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    assessments: Dict[str, SelfAssessment] = field(
        default_factory=dict
    )    

    def register_capability(
        self,
        name: str,
        capability: Any,
    ) -> None:

        self.capabilities[name] = capability

    def add_limitation(
        self,
        limitation: str,
    ) -> None:

        if limitation not in self.limitations:
            self.limitations.append(
                limitation
            )

    def record_success(
        self,
        description: str,
    ) -> None:

        if description not in self.known_successes:
            self.known_successes.append(
                description
            )

    def record_failure(
        self,
        description: str,
    ) -> None:

        if description not in self.known_failures:
            self.known_failures.append(
                description
            )

    def register_strategy(
        self,
        strategy: Strategy,
    ) -> None:

        self.learned_strategies[
            strategy.name
        ] = strategy

    def update_condition(
        self,
        condition: Dict[str, Any],
    ) -> None:

        self.current_condition.update(
            condition
        )

    def get_capability(
        self,
        name: str,
    ) -> Optional[Any]:

        return self.capabilities.get(name)

    def register_assessment(
        self,
        key: str,
        assessment: SelfAssessment,
    ) -> None:

        self.assessments[key] = assessment


    def get_assessment(
        self,
        key: str,
    ) -> Optional[SelfAssessment]:

        return self.assessments.get(key)