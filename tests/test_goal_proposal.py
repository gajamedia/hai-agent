from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.goal_proposal import (
    GoalCandidate,
    GoalOrigin,
    GoalProposalConfigurationError,
    GoalProposalEngine,
    GoalProposalInputError,
    GoalProposalBatch,
    GoalSignal,
)


class TestGoalSignal:
    def test_defaults(self):
        signal = GoalSignal("Improve system")
        assert signal.origin is GoalOrigin.WORLD_GAP
        assert signal.urgency == 0.5

    def test_empty_description_rejected(self):
        with pytest.raises(ValueError):
            GoalSignal(" ")

    def test_scores_are_bounded(self):
        with pytest.raises(ValueError):
            GoalSignal("x", urgency=1.1)

    def test_metadata_is_copied(self):
        metadata = {"source": "world"}
        signal = GoalSignal("x", metadata=metadata)
        metadata["later"] = True
        assert signal.metadata == {"source": "world"}

    def test_origin_from_string(self):
        signal = GoalSignal("x", origin="memory")
        assert signal.origin is GoalOrigin.MEMORY

    def test_invalid_origin_rejected(self):
        with pytest.raises(ValueError):
            GoalSignal("x", origin="invalid")

    def test_parent_id_empty_becomes_none(self):
        signal = GoalSignal("x", parent_goal_id=" ")
        assert signal.parent_goal_id is None

    def test_candidate_id_trimmed(self):
        signal = GoalSignal("x", candidate_id="  c1  ")
        assert signal.candidate_id == "c1"


class TestGoalCandidate:
    def test_manager_kwargs(self):
        candidate = GoalCandidate(
            candidate_id="c1",
            description="Do thing",
            origin=GoalOrigin.EXTERNAL,
            priority=0.8,
            proposal_score=0.8,
            urgency=0.9,
            expected_value=0.8,
            feasibility=0.7,
            novelty=0.2,
            evidence_quality=0.6,
        )
        assert candidate.manager_kwargs()["goal_id"] == "c1"
        assert candidate.manager_kwargs()["priority"] == 0.8

    def test_candidate_bounds(self):
        with pytest.raises(ValueError):
            GoalCandidate(
                candidate_id="c1",
                description="x",
                origin=GoalOrigin.EXTERNAL,
                priority=1.1,
                proposal_score=0.5,
                urgency=0.5,
                expected_value=0.5,
                feasibility=0.5,
                novelty=0.5,
                evidence_quality=0.5,
            )

    def test_metadata_is_copied(self):
        metadata = {"x": 1}
        candidate = GoalCandidate(
            "c1", "x", GoalOrigin.EXTERNAL, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, metadata=metadata,
        )
        metadata["y"] = 2
        assert "y" not in candidate.metadata


class TestGoalProposalBatch:
    def test_best_candidate(self):
        candidate = GoalCandidate(
            "c1", "x", GoalOrigin.EXTERNAL, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5,
        )
        batch = GoalProposalBatch((candidate,))
        assert batch.best_candidate == candidate
        assert batch.count == 1

    def test_empty_best_is_none(self):
        assert GoalProposalBatch(()).best_candidate is None

    def test_collections_are_tuples(self):
        batch = GoalProposalBatch([], ["r"], ["d"])
        assert isinstance(batch.candidates, tuple)
        assert batch.rejected_descriptions == ("r",)
        assert batch.duplicate_descriptions == ("d",)


class TestGoalProposalEngine:
    def test_default_weights_produce_expected_score(self):
        engine = GoalProposalEngine()
        signal = GoalSignal(
            "Do work",
            urgency=1,
            expected_value=1,
            feasibility=1,
            novelty=1,
            evidence_quality=1,
        )
        assert engine.score_signal(signal) == 1.0

    def test_zero_scores(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        signal = GoalSignal("Do work", urgency=0, expected_value=0, feasibility=0, novelty=0, evidence_quality=0)
        assert engine.score_signal(signal) == 0.0

    def test_weights_are_normalized(self):
        engine = GoalProposalEngine(weights={k: 2 for k in (
            "urgency", "expected_value", "feasibility", "novelty", "evidence_quality"
        )})
        assert sum(engine.weights.values()) == pytest.approx(1.0)

    def test_missing_weight_rejected(self):
        with pytest.raises(GoalProposalConfigurationError):
            GoalProposalEngine(weights={"urgency": 1})

    def test_negative_weight_rejected(self):
        weights = dict(GoalProposalEngine._DEFAULT_WEIGHTS)
        weights["urgency"] = -1
        with pytest.raises(GoalProposalConfigurationError):
            GoalProposalEngine(weights=weights)

    def test_zero_total_weight_rejected(self):
        with pytest.raises(GoalProposalConfigurationError):
            GoalProposalEngine(weights={k: 0 for k in GoalProposalEngine._DEFAULT_WEIGHTS})

    def test_max_candidates_validation(self):
        with pytest.raises(GoalProposalConfigurationError):
            GoalProposalEngine(max_candidates=0)

    def test_propose_mapping(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        batch = engine.propose([{"description": "Ship feature", "origin": "external"}])
        assert batch.count == 1
        assert batch.best_candidate.description == "Ship feature"

    def test_propose_object(self):
        @dataclass
        class SignalObject:
            description: str
            urgency: float = 0.9

        engine = GoalProposalEngine(minimum_score=0.0)
        batch = engine.propose([SignalObject("Ship feature")])
        assert batch.count == 1

    def test_invalid_object_rejected(self):
        engine = GoalProposalEngine()
        with pytest.raises(GoalProposalInputError):
            engine.propose([object()])

    def test_existing_description_is_duplicate(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        batch = engine.propose([GoalSignal("Ship feature")], existing_descriptions=[" ship   feature "])
        assert batch.count == 0
        assert batch.duplicate_descriptions == ("Ship feature",)

    def test_duplicate_within_batch(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        batch = engine.propose([GoalSignal("Ship feature"), GoalSignal(" ship feature ")])
        assert batch.count == 1
        assert batch.duplicate_descriptions == ("ship feature",)

    def test_existing_candidate_id_is_duplicate(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        batch = engine.propose([GoalSignal("Ship feature", candidate_id="c1")], existing_candidate_ids=["c1"])
        assert batch.count == 0

    def test_low_score_rejected(self):
        engine = GoalProposalEngine(minimum_score=0.9)
        batch = engine.propose([GoalSignal("Low", urgency=0, expected_value=0, feasibility=0, novelty=0, evidence_quality=0)])
        assert batch.count == 0
        assert batch.rejected_descriptions == ("Low",)

    def test_candidates_ranked_by_score(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        signals = [
            GoalSignal("Low", urgency=0.1, expected_value=0.1, feasibility=0.1, novelty=0.1, evidence_quality=0.1),
            GoalSignal("High", urgency=1, expected_value=1, feasibility=1, novelty=1, evidence_quality=1),
        ]
        batch = engine.propose(signals)
        assert batch.candidates[0].description == "High"

    def test_tie_breaks_by_urgency(self):
        engine = GoalProposalEngine(minimum_score=0.0, weights={
            "urgency": 0.5, "expected_value": 0.0, "feasibility": 0.5, "novelty": 0.0, "evidence_quality": 0.0,
        })
        a = GoalSignal("A", urgency=0.5, feasibility=0.5)
        b = GoalSignal("B", urgency=0.7, feasibility=0.3)
        batch = engine.propose([a, b])
        assert batch.candidates[0].description == "B"

    def test_limit_overflow_rejected(self):
        engine = GoalProposalEngine(minimum_score=0.0, max_candidates=1)
        batch = engine.propose([GoalSignal("A"), GoalSignal("B")])
        assert batch.count == 1
        assert len(batch.rejected_descriptions) == 1

    def test_custom_limit(self):
        engine = GoalProposalEngine(minimum_score=0.0, max_candidates=5)
        batch = engine.propose([GoalSignal("A"), GoalSignal("B")], max_candidates=1)
        assert batch.count == 1

    def test_candidate_id_generated(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        candidate = engine.propose([GoalSignal("Improve server health")]).best_candidate
        assert candidate is not None
        assert candidate.candidate_id.startswith("candidate-improve-server-health")

    def test_explicit_candidate_id_preserved(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        candidate = engine.propose([GoalSignal("Improve", candidate_id="goal-x")]).best_candidate
        assert candidate.candidate_id == "goal-x"

    def test_origin_metadata_added(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        candidate = engine.propose([GoalSignal("Improve", origin=GoalOrigin.MEMORY)]).best_candidate
        assert candidate.metadata["goal_origin"] == "memory"

    def test_original_metadata_preserved(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        candidate = engine.propose([GoalSignal("Improve", metadata={"source": "x"})]).best_candidate
        assert candidate.metadata["source"] == "x"

    def test_parent_goal_is_preserved(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        candidate = engine.propose([GoalSignal("Child", parent_goal_id="parent")]).best_candidate
        assert candidate.parent_goal_id == "parent"

    def test_payload_is_preserved(self):
        payload = {"target": "x"}
        engine = GoalProposalEngine(minimum_score=0.0)
        candidate = engine.propose([GoalSignal("Goal", payload=payload)]).best_candidate
        assert candidate.payload == payload

    def test_manager_kwargs_is_safe_copy(self):
        candidate = GoalCandidate(
            "c1", "x", GoalOrigin.EXTERNAL, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5,
            metadata={"x": 1},
        )
        kwargs = candidate.manager_kwargs()
        kwargs["metadata"]["y"] = 2
        assert "y" not in candidate.metadata

    def test_score_is_rounded(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        score = engine.score_signal(GoalSignal("x", urgency=0.333333333333, expected_value=0.444444444444, feasibility=0.555555555555, novelty=0.666666666666, evidence_quality=0.777777777777))
        assert len(str(score).split(".")[1]) <= 12

    def test_metadata_counts_are_deterministic(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        batch = engine.propose([GoalSignal("A"), GoalSignal("B")], max_candidates=1)
        assert batch.metadata["input_count"] == 2
        assert batch.metadata["candidate_count"] == 1
        assert batch.metadata["rejected_count"] == 1

    def test_generator_input_is_supported(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        batch = engine.propose((GoalSignal(x) for x in ["A", "B"]))
        assert batch.count == 2


    def test_generator_input_count_is_accurate(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        batch = engine.propose((GoalSignal(x) for x in ["A", "B"]))
        assert batch.metadata["input_count"] == 2

    def test_custom_minimum_score_is_applied(self):
        engine = GoalProposalEngine(minimum_score=0.75)
        high = GoalSignal("High", urgency=1, expected_value=1, feasibility=1, novelty=0, evidence_quality=1)
        low = GoalSignal("Low", urgency=0.5, expected_value=0.5, feasibility=0.5, novelty=0, evidence_quality=0.5)
        batch = engine.propose([high, low])
        assert [c.description for c in batch.candidates] == ["High"]

    def test_all_supported_origins_round_trip(self):
        engine = GoalProposalEngine(minimum_score=0.0)
        for origin in GoalOrigin:
            signal = GoalSignal(f"Goal {origin.value}", origin=origin)
            candidate = engine.propose([signal]).best_candidate
            assert candidate.origin is origin

    def test_score_signal_accepts_mapping(self):
        engine = GoalProposalEngine()
        assert engine.score_signal({"description": "x", "urgency": 1, "expected_value": 1, "feasibility": 1, "novelty": 1, "evidence_quality": 1}) == 1.0

    def test_score_signal_accepts_object(self):
        @dataclass
        class S:
            description: str = "x"
            urgency: float = 1
            expected_value: float = 1
            feasibility: float = 1
            novelty: float = 1
            evidence_quality: float = 1

        assert GoalProposalEngine().score_signal(S()) == 1.0

    def test_negative_max_candidates_override_rejected(self):
        engine = GoalProposalEngine()
        with pytest.raises(GoalProposalConfigurationError):
            engine.propose([], max_candidates=0)

    def test_candidate_is_immutable(self):
        candidate = GoalCandidate(
            "c1", "x", GoalOrigin.EXTERNAL, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5,
        )
        with pytest.raises(Exception):
            candidate.priority = 0.8  # type: ignore[misc]
