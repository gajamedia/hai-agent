from types import SimpleNamespace

from agent.core.action import Action
from agent.core.experience import Experience
from agent.learning.strategy import Strategy
from agent.learning.strategy_evolution import (
    StrategyEvolution,
)


def make_episode(
    action: str,
    success: bool,
    cycle: int = 1,
):
    return SimpleNamespace(
        action=action,
        result={
            "success": success,
        },
        cycle=cycle,
    )


def make_failed_experience() -> Experience:
    experience = Experience(
        experience_id="exp-1",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="open_box",
            success=True,
            cycle=1,
        )
    )

    experience.add_episode(
        make_episode(
            action="take_key",
            success=False,
            cycle=2,
        )
    )

    experience.fail()

    return experience


def make_successful_experience() -> Experience:
    experience = Experience(
        experience_id="exp-success",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="open_box",
            success=True,
            cycle=1,
        )
    )

    experience.complete()

    return experience


def test_failed_experience_generates_candidate():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    experience = make_failed_experience()

    evolution = StrategyEvolution()

    result = evolution.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.evolved is True
    assert result.candidate is not None

    assert result.candidate.goal == "Open door"

    assert result.candidate.steps == [
        "open_box",
        "unlock_door",
    ]


def test_evolution_does_not_mutate_source_strategy():
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

    experience = make_failed_experience()

    evolution = StrategyEvolution()

    evolution.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert strategy.steps == original_steps


def test_successful_experience_does_not_trigger_evolution():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    experience = make_successful_experience()

    evolution = StrategyEvolution()

    result = evolution.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.evolved is False
    assert result.candidate is None


def test_unfailed_experience_does_not_trigger_evolution():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    experience = Experience(
        experience_id="exp-2",
        goal="Open door",
    )

    evolution = StrategyEvolution()

    result = evolution.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.evolved is False
    assert result.candidate is None


def test_no_failed_action_returns_no_candidate():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    experience = Experience(
        experience_id="exp-3",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="open_box",
            success=True,
            cycle=1,
        )
    )

    experience.fail()

    evolution = StrategyEvolution()

    result = evolution.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.evolved is False
    assert result.candidate is None

    assert (
        result.reason
        == "no failed actions found"
    )


def test_evolve_candidates_returns_multiple_candidates():
    strategy = Strategy(
        name="open_door",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "find_spare_key",
            "unlock_door",
        ],
    )

    experience = Experience(
        experience_id="exp-4",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="take_key",
            success=False,
            cycle=1,
        )
    )

    experience.add_episode(
        make_episode(
            action="find_spare_key",
            success=False,
            cycle=2,
        )
    )

    experience.fail()

    evolution = StrategyEvolution(
        max_candidates=3,
    )

    candidates = evolution.evolve_candidates(
        strategy=strategy,
        experience=experience,
    )

    assert len(candidates) == 2

    assert candidates[0].steps == [
        "open_box",
        "find_spare_key",
        "unlock_door",
    ]

    assert candidates[1].steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]


def test_none_strategy_is_rejected():
    evolution = StrategyEvolution()

    experience = make_failed_experience()

    try:
        evolution.evolve(
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
            "unlock_door",
        ],
    )

    evolution = StrategyEvolution()

    try:
        evolution.evolve(
            strategy=strategy,
            experience=None,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "experience cannot be None."
        )

def test_unified_evolution_generates_removal_and_replacement():
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
        experience_id="exp-unified",
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

    actions = [
        Action(
            name="take_key",
            description="Take key",
            handler=lambda: {
                "success": True,
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
            effects={
                "key": "agent",
            },
        ),
    ]

    evolution = StrategyEvolution(
        actions=actions,
        max_candidates=3,
    )

    candidates = (
        evolution.evolve_candidates(
            strategy=strategy,
            experience=experience,
        )
    )

    assert [
        "open_box",
        "unlock_door",
    ] in [
        candidate.steps
        for candidate in candidates
    ]

    assert [
        "open_box",
        "find_spare_key",
        "unlock_door",
    ] in [
        candidate.steps
        for candidate in candidates
    ]

def test_evolve_prefers_removal_candidate_for_backward_compatibility():
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
        experience_id="exp-priority",
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

    actions = [
        Action(
            name="take_key",
            description="Take key",
            handler=lambda: {
                "success": True,
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
            effects={
                "key": "agent",
            },
        ),
    ]

    evolution = StrategyEvolution(
        actions=actions,
    )

    result = evolution.evolve(
        strategy=strategy,
        experience=experience,
    )

    assert result.evolved is True

    assert result.reason == (
        "candidate generated by removing "
        "a failed action"
    )

    assert result.candidate is not None

    assert result.candidate.steps == [
        "open_box",
        "unlock_door",
    ]
