from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.context import AgentContext, AgentContextBuilder


@dataclass
class SampleState:
    status: str
    score: float


@dataclass
class SampleStrategy:
    name: str
    priority: int


def test_builder_creates_complete_context():
    context = AgentContextBuilder().build(
        goal={"id": "g1", "description": "open the box"},
        state={"position": "room_a"},
        world_model={"box": {"state": "closed"}},
        relevant_memory=[{"episode": 1}, {"episode": 2}],
        strategy={"mode": "explore"},
        self_awareness={"confidence": 0.72},
        calibration={"calibrated_confidence": 0.68},
        metacognition={"lesson": "inspect before acting"},
    )

    assert isinstance(context, AgentContext)
    assert context.goal["id"] == "g1"
    assert context.state["position"] == "room_a"
    assert context.world_model["box"]["state"] == "closed"
    assert len(context.relevant_memory) == 2
    assert context.strategy["mode"] == "explore"
    assert context.self_awareness["confidence"] == 0.72
    assert context.calibration["calibrated_confidence"] == 0.68
    assert context.metacognition["lesson"] == "inspect before acting"


def test_dataclasses_and_objects_are_normalized():
    context = AgentContextBuilder().build(
        state=SampleState("ready", 0.5),
        strategy=SampleStrategy("inspect_first", 2),
    )

    assert context.state == {"status": "ready", "score": 0.5}
    assert context.strategy == {"name": "inspect_first", "priority": 2}


def test_memory_is_limited_deterministically():
    builder = AgentContextBuilder(max_memory_items=2)

    context = builder.build(
        relevant_memory=[
            {"episode": 1},
            {"episode": 2},
            {"episode": 3},
        ]
    )

    assert context.relevant_memory == (
        {"episode": 1},
        {"episode": 2},
    )


def test_empty_memory_is_allowed():
    context = AgentContextBuilder().build(relevant_memory=[])

    assert context.relevant_memory == ()


def test_string_values_are_truncated():
    context = AgentContextBuilder(max_string_length=5).build(
        goal={"description": "abcdefghij"}
    )

    assert context.goal["description"] == "abcde…"


def test_deep_structures_are_bounded():
    value = {"a": {"b": {"c": {"d": 1}}}}

    context = AgentContextBuilder(max_depth=2).build(state=value)

    assert context.state["a"]["b"]["c"] == "<max-depth-reached>"


def test_sets_are_normalized_stably():
    context = AgentContextBuilder().build(state={"items": {"b", "a"}})

    assert context.state["items"] == ["a", "b"]


def test_metadata_is_preserved():
    context = AgentContextBuilder().build(
        metadata={
            "episode_id": "ep-7",
            "source": "decision_cycle",
        }
    )

    assert context.metadata == {
        "episode_id": "ep-7",
        "source": "decision_cycle",
    }


def test_non_mapping_metadata_is_rejected_after_normalization():
    with pytest.raises(TypeError, match="metadata must normalize to a mapping"):
        AgentContextBuilder().build(metadata="invalid")


def test_string_memory_is_rejected():
    with pytest.raises(
        TypeError,
        match="relevant_memory must be a sequence of memory items",
    ):
        AgentContextBuilder().build(relevant_memory="memory")


def test_prompt_contains_all_context_sections():
    context = AgentContextBuilder().build(
        goal={"id": "g1"},
        state={"status": "ready"},
        world_model={"door": "locked"},
        relevant_memory=[{"lesson": "check key"}],
        strategy={"mode": "explore"},
        self_awareness={"confidence": 0.4},
        calibration={"calibrated_confidence": 0.3},
        metacognition={"next_step": "inspect key"},
    )

    prompt = context.to_prompt()

    assert "## GOAL" in prompt
    assert "## CURRENT STATE" in prompt
    assert "## WORLD MODEL" in prompt
    assert "## RELEVANT MEMORY" in prompt
    assert "## ACTIVE STRATEGY" in prompt
    assert "## SELF-AWARENESS" in prompt
    assert "## CALIBRATION" in prompt
    assert "## METACOGNITION" in prompt
    assert "inspect key" in prompt


def test_to_dict_is_plain_data():
    context = AgentContextBuilder().build(
        state=SampleState("ready", 1.0),
        relevant_memory=[SampleStrategy("learn", 1)],
    )

    data = context.to_dict()

    assert data["state"] == {"status": "ready", "score": 1.0}
    assert data["relevant_memory"] == [{"name": "learn", "priority": 1}]
    assert isinstance(data["relevant_memory"], list)


def test_builder_configuration_is_validated():
    with pytest.raises(ValueError, match="max_memory_items must be >= 0"):
        AgentContextBuilder(max_memory_items=-1)

    with pytest.raises(ValueError, match="max_string_length must be > 0"):
        AgentContextBuilder(max_string_length=0)

    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        AgentContextBuilder(max_depth=-1)
