from agent.core.action import Action
from agent.learning.strategy import Strategy
from agent.learning.strategy_candidate_validator import (
    StrategyCandidateValidator,
)
from agent.learning.strategy_evolution_selector import (
    StrategyEvolutionPipeline,
)


def make_actions():
    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock door",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={
                "door": "unlocked",
            },
        ),
    ]


def test_pipeline_rejects_invalid_and_selects_valid():
    source = Strategy(
        name="source",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    invalid = Strategy(
        name="invalid",
        goal="Open door",
        steps=[
            "unlock_door",
        ],
    )

    valid = Strategy(
        name="valid",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    # Make valid candidate different from source by giving
    # the source another step that will not be retained.
    source.steps = [
        "open_box",
        "take_key",
        "unlock_door",
        "extra_step",
    ]

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    pipeline = StrategyEvolutionPipeline(
        validator=validator,
    )

    result = pipeline.select(
        source_strategy=source,
        candidates=[
            invalid,
            valid,
        ],
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid_candidates == [
        valid
    ]

    assert result.rejected_candidates == [
        invalid
    ]

    assert result.selection.selected is True
    assert result.selection.strategy is valid


def test_pipeline_with_no_valid_candidates():
    source = Strategy(
        name="source",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    invalid = Strategy(
        name="invalid",
        goal="Open door",
        steps=[
            "unlock_door",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    pipeline = StrategyEvolutionPipeline(
        validator=validator,
    )

    result = pipeline.select(
        source_strategy=source,
        candidates=[
            invalid,
        ],
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid_candidates == [
        ]
    assert result.rejected_candidates == [
        invalid
    ]
    assert result.selection.selected is False
    assert result.selection.strategy is None


def test_pipeline_does_not_register_candidate():
    source = Strategy(
        name="source",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
            "extra_step",
        ],
    )

    candidate = Strategy(
        name="candidate",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    pipeline = StrategyEvolutionPipeline(
        validator=validator,
    )

    result = pipeline.select(
        source_strategy=source,
        candidates=[
            candidate,
        ],
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.selection.strategy is candidate
