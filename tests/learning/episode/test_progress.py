from dataclasses import FrozenInstanceError

import pytest

from src.rl.episode import (
    RaceCheckpoint,
    RaceFinish,
    RaceProgressResult,
    RaceProgressState,
    RaceRoute,
    advance_race_progress,
)
from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from src.simulation.track.state import ScorePickupCollected


def _route() -> RaceRoute:
    return RaceRoute(
        (RaceCheckpoint("cp-0", 0, Vec3(1.0, 0.0, 0.0)), RaceCheckpoint("cp-1", 1, Vec3(2.0, 0.0, 0.0))),
        RaceFinish("finish", Vec3(3.0, 0.0, 0.0)),
    )


@pytest.mark.parametrize("index", [True, -1, 1.0, "0"])
def test_progress_state_rejects_invalid_index(index: object) -> None:
    with pytest.raises(ValueError):
        RaceProgressState(index, False)


@pytest.mark.parametrize("finished", [0, "false", None])
def test_progress_state_requires_bool_finished(finished: object) -> None:
    with pytest.raises(ValueError):
        RaceProgressState(0, finished)


def test_progress_values_validate_and_are_immutable() -> None:
    state = RaceProgressState(0, False)
    result = RaceProgressResult(state, Vec3(1.0, 0.0, 0.0), None, False)

    with pytest.raises(ValueError):
        RaceProgressResult("state", None, None, False)
    with pytest.raises(ValueError):
        RaceProgressResult(state, "target", None, False)
    with pytest.raises(ValueError):
        RaceProgressResult(state, None, "", False)
    with pytest.raises(ValueError):
        RaceProgressResult(state, None, None, 0)
    with pytest.raises(FrozenInstanceError):
        result.finished_now = True


@pytest.mark.parametrize("state", [RaceProgressState(3, False), RaceProgressState(1, True)])
def test_advance_rejects_route_incompatible_state(state: RaceProgressState) -> None:
    with pytest.raises(ValueError):
        advance_race_progress(_route(), state, ())


def test_expected_checkpoint_advances_once_and_returns_post_transition_target() -> None:
    route = _route()

    result = advance_race_progress(route, RaceProgressState(0, False), (CheckpointEntered("cp-0", 0),))

    assert result.state == RaceProgressState(1, False)
    assert result.completed_checkpoint_id == "cp-0"
    assert result.finished_now is False
    assert result.target_position == route.target_for(result.state)


@pytest.mark.parametrize(
    "events",
    [
        (CheckpointEntered("cp-1", 1),),
        (CheckpointEntered("cp-0", 1),),
        (CheckpointEntered("wrong", 0),),
        (ObstacleHit("wall"),),
        (ScorePickupCollected("pickup", 1),),
        (FinishLineEntered("finish"),),
        (FinishLineEntered("wrong-finish"),),
    ],
)
def test_invalid_events_leave_progress_unchanged(events: tuple[object, ...]) -> None:
    route = _route()
    state = RaceProgressState(0, False)

    result = advance_race_progress(route, state, events)

    assert result == RaceProgressResult(state, route.target_for(state), None, False)


def test_one_transition_cannot_skip_multiple_checkpoints() -> None:
    route = _route()

    result = advance_race_progress(route, RaceProgressState(0, False), (CheckpointEntered("cp-1", 1), CheckpointEntered("cp-0", 0), CheckpointEntered("cp-1", 1)))

    assert result.state == RaceProgressState(1, False)
    assert result.completed_checkpoint_id == "cp-0"


def test_finish_requires_all_checkpoints_but_zero_checkpoint_route_finishes_directly() -> None:
    route = _route()
    zero_route = RaceRoute((), RaceFinish("finish", Vec3(3.0, 0.0, 0.0)))

    assert advance_race_progress(route, RaceProgressState(1, False), (FinishLineEntered("finish"),)).state == RaceProgressState(1, False)
    assert advance_race_progress(zero_route, RaceProgressState(0, False), (FinishLineEntered("finish"),)) == RaceProgressResult(RaceProgressState(0, True), None, None, True)


@pytest.mark.parametrize(
    "events",
    [
        (CheckpointEntered("cp-1", 1), FinishLineEntered("finish")),
        (FinishLineEntered("finish"), CheckpointEntered("cp-1", 1)),
    ],
)
def test_final_checkpoint_and_finish_complete_in_same_transition_regardless_of_event_order(events: tuple[object, ...]) -> None:
    route = _route()

    result = advance_race_progress(route, RaceProgressState(1, False), events)

    assert result == RaceProgressResult(RaceProgressState(2, True), None, "cp-1", True)


def test_finished_progress_ignores_later_events() -> None:
    route = _route()
    state = RaceProgressState(2, True)

    assert advance_race_progress(route, state, (CheckpointEntered("cp-0", 0), FinishLineEntered("finish"))) == RaceProgressResult(state, None, None, False)
