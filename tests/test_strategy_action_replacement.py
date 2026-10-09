from types import SimpleNamespace

from agent.core.action import Action
from agent.core.experience import Experience
from agent.learning.strategy import Strategy
from agent.learning.strategy_action_replacement import (
    StrategyActionReplacement,
)
from agent.learning.strategy_candidate_validator import (
    StrategyCandidateValidator,
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
            name="find_spare_key",
            description="Find spare key",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "spare_key": "available",
            },
            effects={
                "key": "agent",
                "spare_key": "found",
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


def make_failed_experience():
    experience = Experience(
        experience_id="exp-replacement",
        goal="Open door",
    )

    experience.add_episode(
        SimpleNamespace(
            action="take_key",
            result={
                "success": False,
            },
            cycle=1,
        )
    )

    experience.fail()

    return experience


def test_find_replacement_action():
    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    replacements = (
        engine.find_replacements(
            "take_key"
        )
    )

    assert len(replacements) == 1
    assert replacements[0].name == (
        "find_spare_key"
    )


def test_replacement_generates_new_candidate():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    experience = (
        make_failed_experience()
    )

    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    result = engine.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.replaced is True
    assert result.failed_action == (
        "take_key"
    )
    assert result.replacement_action == (
        "find_spare_key"
    )

    assert result.candidate is not None

    assert result.candidate.steps == [
        "open_box",
        "find_spare_key",
        "unlock_door",
    ]


def test_source_strategy_is_not_mutated():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    original_steps = list(
        strategy.steps
    )

    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    engine.evolve(
        strategy=strategy,
        experience=make_failed_experience(),
    )

    assert strategy.steps == original_steps


def test_successful_experience_does_not_replace_action():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    experience = Experience(
        experience_id="exp-success",
        goal="Open door",
    )

    experience.add_episode(
        SimpleNamespace(
            action="take_key",
            result={
                "success": True,
            },
            cycle=1,
        )
    )

    experience.complete()

    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    result = engine.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.replaced is False
    assert result.candidate is None


def test_unknown_failed_action_has_no_replacement():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "unknown_action",
            "unlock_door",
        ],
    )

    experience = (
        make_failed_experience()
    )

    # Rewrite failed action to match the
    # unknown strategy action.
    experience.episodes[0].action = (
        "unknown_action"
    )

    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    result = engine.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.replaced is False
    assert result.candidate is None
    assert result.reason == (
        "no compatible replacement action found"
    )


def test_failed_action_not_in_strategy_is_ignored():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    experience = (
        make_failed_experience()
    )

    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    result = engine.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.replaced is False
    assert result.candidate is None


def test_evolve_candidates_returns_replacement():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    experience = (
        make_failed_experience()
    )

    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    candidates = (
        engine.evolve_candidates(
            strategy=strategy,
            experience=experience,
        )
    )

    assert len(candidates) == 1

    assert candidates[0].steps == [
        "open_box",
        "find_spare_key",
        "unlock_door",
    ]


def test_no_effect_action_cannot_be_replacement():
    actions = make_actions()

    actions.append(
        Action(
            name="inspect",
            description="Inspect",
            handler=lambda: {
                "success": True,
            },
            preconditions={},
            effects={},
        )
    )

    engine = StrategyActionReplacement(
        actions=actions
    )

    replacements = (
        engine.find_replacements(
            "inspect"
        )
    )

    assert replacements == []


def test_none_strategy_is_rejected():
    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    experience = (
        make_failed_experience()
    )

    try:
        engine.evolve(
            strategy=None,
            experience=experience,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "strategy cannot be None."
        )


def test_none_experience_is_rejected():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    try:
        engine.evolve(
            strategy=strategy,
            experience=None,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "experience cannot be None."
        )

def test_replacement_candidate_is_compatible_with_validator():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    experience = (
        make_failed_experience()
    )

    engine = StrategyActionReplacement(
        actions=make_actions()
    )

    result = engine.evolve(
        strategy=strategy,
        experience=experience,
    )

    candidate = result.candidate

    assert candidate is not None

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    validation = validator.validate(
        source_strategy=strategy,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": "available",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert validation.valid is True
    assert validation.reason == (
        "candidate is valid"
    )
