import json
from typing import Any, Optional

from agent.memory.persistence.database import (
    SQLiteDatabase,
)
from agent.self.assessment import SelfAssessment
from agent.self.capability import Capability
from agent.self.model import SelfModel
from agent.learning.strategy import Strategy


class SelfModelRepository:

    MODEL_ID = 1

    def __init__(
        self,
        database: SQLiteDatabase,
    ):
        self.database = database
        self._initialize()

    def _initialize(self) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS self_models (
                    model_id INTEGER PRIMARY KEY,

                    capabilities_json TEXT NOT NULL,
                    limitations_json TEXT NOT NULL,
                    known_successes_json TEXT NOT NULL,
                    known_failures_json TEXT NOT NULL,
                    learned_strategies_json TEXT NOT NULL,

                    confidence REAL NOT NULL,

                    current_condition_json TEXT NOT NULL,

                    assessments_json TEXT NOT NULL
                )
                """
            )

            connection.commit()

    def save(
        self,
        model: SelfModel,
    ) -> None:

        capabilities = {}

        for name, capability in (
            model.capabilities.items()
        ):
            capabilities[name] = {
                "name": capability.name,
                "confidence": capability.confidence,
                "success_count": (
                    capability.success_count
                ),
                "failure_count": (
                    capability.failure_count
                ),
                "evidence": list(
                    capability.evidence
                ),
            }

        strategies = {}

        for name, strategy in (
            model.learned_strategies.items()
        ):
            strategies[name] = {
                "name": strategy.name,
                "goal": strategy.goal,
                "steps": list(
                    strategy.steps
                ),
                "success_count": (
                    strategy.success_count
                ),
                "failure_count": (
                    strategy.failure_count
                ),
                "total_actions": (
                    strategy.total_actions
                ),
                "reflection_penalty": (
                    strategy.reflection_penalty
                ),
                "source_experiences": list(
                    strategy.source_experiences
                ),
            }

        assessments = {}

        for key, assessment in (
            model.assessments.items()
        ):
            assessments[key] = {
                "confidence": (
                    assessment.confidence
                ),
                "uncertainty": (
                    assessment.uncertainty
                ),
                "risk": (
                    assessment.risk
                ),
            }

        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO self_models (
                    model_id,
                    capabilities_json,
                    limitations_json,
                    known_successes_json,
                    known_failures_json,
                    learned_strategies_json,
                    confidence,
                    current_condition_json,
                    assessments_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(model_id)
                DO UPDATE SET
                    capabilities_json =
                        excluded.capabilities_json,
                    limitations_json =
                        excluded.limitations_json,
                    known_successes_json =
                        excluded.known_successes_json,
                    known_failures_json =
                        excluded.known_failures_json,
                    learned_strategies_json =
                        excluded.learned_strategies_json,
                    confidence =
                        excluded.confidence,
                    current_condition_json =
                        excluded.current_condition_json,
                    assessments_json =
                        excluded.assessments_json
                """,
                (
                    self.MODEL_ID,
                    json.dumps(
                        capabilities,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        model.limitations,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        model.known_successes,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        model.known_failures,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        strategies,
                        ensure_ascii=False,
                    ),
                    model.confidence,
                    json.dumps(
                        model.current_condition,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        assessments,
                        ensure_ascii=False,
                    ),
                ),
            )

            connection.commit()

    def get(
        self,
    ) -> Optional[SelfModel]:

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM self_models
                WHERE model_id = ?
                """,
                (self.MODEL_ID,),
            ).fetchone()

        if row is None:
            return None

        return self._from_row(row)

    def exists(self) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM self_models
                WHERE model_id = ?
                """,
                (self.MODEL_ID,),
            ).fetchone()

        return row is not None

    def clear(self) -> None:

        with self.database.connect() as connection:
            connection.execute(
                """
                DELETE FROM self_models
                WHERE model_id = ?
                """,
                (self.MODEL_ID,),
            )

            connection.commit()

    def _from_row(
        self,
        row: Any,
    ) -> SelfModel:

        capabilities_data = json.loads(
            row["capabilities_json"]
        )

        capabilities = {}

        for name, data in (
            capabilities_data.items()
        ):
            capabilities[name] = Capability(
                name=data["name"],
                confidence=data["confidence"],
                success_count=data[
                    "success_count"
                ],
                failure_count=data[
                    "failure_count"
                ],
                evidence=list(
                    data["evidence"]
                ),
            )

        strategies_data = json.loads(
            row["learned_strategies_json"]
        )

        strategies = {}

        for name, data in (
            strategies_data.items()
        ):
            strategies[name] = Strategy(
                name=data["name"],
                goal=data["goal"],
                steps=list(
                    data["steps"]
                ),
                success_count=data[
                    "success_count"
                ],
                failure_count=data[
                    "failure_count"
                ],
                total_actions=data[
                    "total_actions"
                ],
                reflection_penalty=data[
                    "reflection_penalty"
                ],
                source_experiences=list(
                    data["source_experiences"]
                ),
            )

        assessments_data = json.loads(
            row["assessments_json"]
        )

        assessments = {}

        for key, data in (
            assessments_data.items()
        ):
            assessments[key] = SelfAssessment(
                confidence=data[
                    "confidence"
                ],
                uncertainty=data[
                    "uncertainty"
                ],
                risk=data["risk"],
            )

        return SelfModel(
            capabilities=capabilities,
            limitations=json.loads(
                row["limitations_json"]
            ),
            known_successes=json.loads(
                row["known_successes_json"]
            ),
            known_failures=json.loads(
                row["known_failures_json"]
            ),
            learned_strategies=strategies,
            confidence=row["confidence"],
            current_condition=json.loads(
                row["current_condition_json"]
            ),
            assessments=assessments,
        )
