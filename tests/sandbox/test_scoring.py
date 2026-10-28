from dataclasses import FrozenInstanceError

import pytest

from src.simulation.rules.scoring import ScoreState, apply_score_events
from src.simulation.track.state import ScorePickupCollected


def test_score_state_defaults_to_zero_and_validates_non_negative_integer() -> None:
    assert ScoreState() == ScoreState(0)
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            ScoreState(value)  # type: ignore[arg-type]


def test_score_state_is_immutable() -> None:
    score = ScoreState(10)
    with pytest.raises(FrozenInstanceError):
        score.points = 20  # type: ignore[misc]


def test_apply_score_events_adds_one_or_multiple_pickups() -> None:
    score = ScoreState(5)
    result = apply_score_events(score, (ScorePickupCollected("a", 10), ScorePickupCollected("b", 25)))
    assert result == ScoreState(40)
    assert score == ScoreState(5)


def test_apply_score_events_empty_tuple_is_value_noop() -> None:
    score = ScoreState(12)
    assert apply_score_events(score, ()) == score
